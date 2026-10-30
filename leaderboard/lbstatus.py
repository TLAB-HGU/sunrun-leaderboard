"""Post an experiment's status (대기/진행/완료/실패/중단) to the leaderboard Space before and while it runs.

    python lbstatus.py set <experiment> --status 대기 --desc "what is tested"   # register before running
    python lbstatus.py run <experiment> -- <command ...>                         # 진행 while it runs (and until submitted)
    python lbstatus.py sync                                                      # after submit.py: clear it, restore rows
    python lbstatus.py set <experiment> --status 실패 --note "why"               # end a run without a leaderboard result
    python lbstatus.py list                                                      # print entries in the terminal
    python lbstatus.py rm <experiment>                                           # drop a mistaken entry

The Space keeps 대기/진행 entries in status.json and finished ones in status-history.json. Each 대기/진행 entry is
a row at the top of the ranking table in index.html, shown as "<experiment> (진행)" with dashes for the metrics.
A finished experiment leaves those rows by appearing in the ranking: every write marks an active entry 완료 once
the dataset holds a result for the same (member, experiment) submitted after the entry was registered, and moves
it to the history. A successful `run` therefore stays 진행 until its result is submitted.
Each write also bolds the best and underlines the second-best value of every skill metric column (mark_leaders).
Every publisher (submit.py and its per-study copies) rewrites index.html and drops the rows and the marks, so
every set/rm, `sync`, and a running `run` (every --sync-every seconds) re-apply them; they commit only when
something changed.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

SPACE = "tlabtlab/sunrun-leaderboard"
DATASET = "tlabtlab/sunrun-lb-store"
# Format 2 stores entries under "active"/"finished", never "entries": `run` processes started on 2026-10-28 with
# the first version read status.json["entries"], so they now fail with a warning instead of re-inserting the
# separate status table that this version replaced.
STATUS_FILE, HISTORY_FILE = "status.json", "status-history.json"
ROWS_BEGIN, ROWS_END = "<!-- sunrun-status-rows:begin -->", "<!-- sunrun-status-rows:end -->"
ROWS = re.compile(re.escape(ROWS_BEGIN) + ".*?" + re.escape(ROWS_END), re.S)
LEGACY_BLOCK = re.compile(r"<!-- sunrun-status:begin -->.*?<!-- sunrun-status:end -->", re.S)
STATUSES = ("대기", "진행", "완료", "실패", "중단")
ACTIVE = ("진행", "대기")
ALIASES = {"queued": "대기", "waiting": "대기", "pending": "대기", "running": "진행", "done": "완료",
           "failed": "실패", "cancelled": "중단", "canceled": "중단"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
DEFAULT_MEMBER = os.environ.get("SUNRUN_MEMBER", "seongeun")
FINISHED_KEPT = 300
HIGHER_IS_BETTER = re.compile(r"^(CSI|skill vs naive)$")
LOWER_IS_BETTER = re.compile(r"MSE|MAE|≥\d+|^(low|high)_(low|high)_(low|high)$")  # sec/fold etc. are cost, not skill
LEADER_MARK = re.compile(r'<(b|u) class="lb-[12]">(.*?)</\1>', re.S)
CELL = re.compile(r"<td(\s[^>]*)?>(.*?)</td>", re.S)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def normalize_status(s: str) -> str:
    s = ALIASES.get(s.strip().lower(), s.strip())
    if s not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES} (or {sorted(ALIASES)}), got {s!r}")
    return s


def detect_agent() -> str:
    if os.environ.get("SUNRUN_AGENT"):
        return os.environ["SUNRUN_AGENT"]
    if os.environ.get("CLAUDECODE"):
        return "claude-code"
    if any(k.startswith("CODEX_") for k in os.environ):
        return "codex"
    return ""


def upsert(entries: list[dict], member: str, experiment: str, status: str, at: str | None = None,
           **fields) -> list[dict]:
    """Return a new list with (member, experiment) set to `status`; empty fields keep their previous value."""
    status, at = normalize_status(status), at or now()
    for name in (member, experiment):
        if not SAFE_NAME.fullmatch(name):
            raise ValueError(f"member/experiment must match {SAFE_NAME.pattern}, got {name!r}")
    out = [dict(e) for e in entries]
    e = next((e for e in out if (e["member"], e["experiment"]) == (member, experiment)), None)
    if e is None:
        e = {"member": member, "experiment": experiment, "created_utc": at, "history": []}
        out.append(e)
    elif e["status"] not in ACTIVE and status in ACTIVE:  # a re-run: older leaderboard results must not clear it
        for k in ("finished_utc", "note", "started_utc"):
            e.pop(k, None)
        e["created_utc"] = at
    e.update({k: v for k, v in fields.items() if v})
    if status == "진행" and e.get("status") != "진행":
        e["started_utc"] = at
    if status in ("완료", "실패", "중단"):
        e["finished_utc"] = at
    if e.get("status") != status:
        e["history"] = [*e.get("history", []), [status, at]]
    e["status"], e["updated_utc"] = status, at
    return out


def remove(entries: list[dict], member: str, experiment: str) -> list[dict]:
    return [e for e in entries if (e["member"], e["experiment"]) != (member, experiment)]


def _digits(t: str) -> str:
    """'20261028T130424123Z' (submit.py) and '2026-10-28T13:04:24Z' (here) -> '20261028130424' for comparison."""
    return re.sub(r"\D", "", t)[:14]


def on_leaderboard(entries: list[dict], submitted: dict[tuple[str, str], str]) -> list[dict]:
    """Mark an active entry 완료 when `submitted[(member, experiment)]` is not older than its registration."""
    out = entries
    for e in entries:
        t = submitted.get((e["member"], e["experiment"]))
        if e["status"] in ACTIVE and t and _digits(t) >= _digits(e["created_utc"]):
            out = upsert(out, e["member"], e["experiment"], "완료", note="on the leaderboard")
    return out


def leaderboard_submissions(entries: list[dict], dataset: str = DATASET) -> dict[tuple[str, str], str]:
    """Latest submitted_utc per active (member, experiment) among the dataset's results/<member>/<experiment>/*.json."""
    from huggingface_hub import HfApi, hf_hub_download
    active = {(e["member"], e["experiment"]) for e in entries if e["status"] in ACTIVE}
    if not active:
        return {}
    latest = {}
    for f in HfApi().list_repo_files(dataset, repo_type="dataset"):
        parts = f.split("/")
        if len(parts) == 4 and parts[0] == "results" and (parts[1], parts[2]) in active:
            with open(hf_hub_download(dataset, f, repo_type="dataset"), encoding="utf-8") as fh:
                t = json.load(fh)["submitted_utc"]
            latest[(parts[1], parts[2])] = max(latest.get((parts[1], parts[2]), ""), t)
    return latest


def _ranking(page: str) -> tuple[int, int, list[tuple[str, str]]] | None:
    """(start, stop, [(header label, header class)]) of the ranking table, the first <table> on the page."""
    start = page.find("<table>")
    stop = page.find("</table>", start)
    if start < 0 or stop < 0 or "<tbody>" not in page[start:stop]:
        return None
    head = page[start:stop].split("<tbody>", 1)[0]
    cols = []
    for attrs, inner in re.findall(r"<th(\s[^>]*)?>(.*?)</th>", head, re.S):
        cls = re.search(r'class="([^"]*)"', attrs or "")
        cols.append((html.unescape(re.sub(r"<[^>]+>", "", inner)).strip(), cls.group(1) if cls else ""))
    return start, stop, cols


def trim_ranking_columns(page: str) -> str:
    """Keep skill vs naive as the final column, including pages from older publishers."""
    ranking = _ranking(page)
    if ranking is None:
        return page
    start, stop, cols = ranking
    labels = [label for label, _ in cols]
    if "skill vs naive" not in labels:
        return page
    count = labels.index("skill vs naive") + 1
    if count == len(cols):
        return page

    def trim_row(match):
        row = match.group(0)
        cells = list(re.finditer(r"<t[hd]\b[^>]*>.*?</t[hd]>", row, re.S))
        if len(cells) <= count:
            return row
        return row[:cells[count].start()] + row[cells[-1].end():]

    table = re.sub(r"<tr\b[^>]*>.*?</tr>", trim_row, page[start:stop], flags=re.S)
    return page[:start] + table + page[stop:]


def hide_ranking_rows(page: str, visibility: dict) -> str:
    """Hide exact member/experiment identities without changing stored results or active runs."""
    hidden = {(e["member"], e["experiment"]) for e in visibility.get("hidden", [])}
    found = _ranking(page)
    if not hidden or not found:
        return page
    start, stop, cols = found
    labels = [label for label, _ in cols]
    if "experiment" not in labels or "member" not in labels:
        return page
    experiment_col, member_col = labels.index("experiment"), labels.index("member")
    rank = 0

    def filter_row(match):
        nonlocal rank
        row = match.group(0)
        cells = list(CELL.finditer(row))
        if len(cells) != len(cols):
            return row
        values = [html.unescape(re.sub(r"<[^>]+>", "", c.group(2))).strip() for c in cells]
        experiment = re.sub(r" \((?:진행|대기)\)$", "", values[experiment_col])
        if (values[member_col], experiment) in hidden:
            return ""
        if values[0].isdigit():
            rank += 1
            cell = cells[0]
            row = row[:cell.start(2)] + str(rank) + row[cell.end(2):]
        return row

    table = re.sub(r"<tr\b[^>]*>.*?</tr>", filter_row, page[start:stop], flags=re.S)
    return page[:start] + table + page[stop:]


def bold_experiment_names(page: str, visibility: dict) -> str:
    """Emphasize configured experiment names, independent of their current ranks."""
    selected = {(e["member"], e["experiment"]) for e in visibility.get("bold", [])}
    found = _ranking(page)
    if not found:
        return page
    start, stop, cols = found
    labels = [label for label, _ in cols]
    if "experiment" not in labels or "member" not in labels:
        return page
    experiment_col, member_col = labels.index("experiment"), labels.index("member")

    def emphasize(match):
        row = match.group(0)
        cells = list(CELL.finditer(row))
        if len(cells) != len(cols):
            return row
        text = lambda cell: html.unescape(re.sub(r"<[^>]+>", "", cell.group(2))).strip()
        cell = cells[experiment_col]
        inner = re.sub(r'<strong class="lb-experiment">(.*?)</strong>', r"\1", cell.group(2), flags=re.S)
        if (text(cells[member_col]), text(cell)) in selected:
            inner = f'<strong class="lb-experiment">{inner}</strong>'
        return row[:cell.start(2)] + inner + row[cell.end(2):]

    table = re.sub(r"<tr\b[^>]*>.*?</tr>", emphasize, page[start:stop], flags=re.S)
    return page[:start] + table + page[stop:]


def apply_architecture_links(page: str, mapping: dict) -> str:
    """Replace code links only for an exact member, experiment, and full config SHA."""
    models = {(e["member"], e["experiment"], e["config_sha256"]): e["url"]
              for e in mapping.get("models", [])}
    found = _ranking(page)
    if not models or not found:
        return page
    start, stop, cols = found
    labels = [label for label, _ in cols]
    if not all(label in labels for label in ("experiment", "member", "code")):
        return page
    experiment_col, member_col, code_col = (labels.index(label) for label in ("experiment", "member", "code"))

    def replace_link(match):
        row = match.group(0)
        cells = list(CELL.finditer(row))
        if len(cells) != len(cols):
            return row
        text = lambda cell: html.unescape(re.sub(r"<[^>]+>", "", cell.group(2))).strip()
        member, experiment = text(cells[member_col]), text(cells[experiment_col])
        config = None
        if "config" in labels:
            title = re.search(r'title="([0-9a-f]{64})"', cells[labels.index("config")].group(1) or "")
            if title:
                config = title.group(1)
        cell = cells[code_col]
        if config is None:
            href = re.search(r'href="([^"]+)"', cell.group(2))
            if href:
                path = rf"/scripts/{re.escape(member)}/{re.escape(experiment)}/([0-9a-f]{{64}})\.py(?:[?#]|$)"
                script = re.search(path, html.unescape(href.group(1)))
                if script:
                    config = script.group(1)
        url = models.get((member, experiment, config))
        if url is None:
            return row
        link = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">architecture</a>'
        return row[:cell.start(2)] + link + row[cell.end(2):]

    table = re.sub(r"<tr\b[^>]*>.*?</tr>", replace_link, page[start:stop], flags=re.S)
    return page[:start] + table + page[stop:]


def status_rows(entries: list[dict], cols: list[tuple[str, str]]) -> str:
    """One row per 대기/진행 entry, 진행 first: "<experiment> (진행)" (details in its tooltip), the member, and
    dashes in every other column, between ROWS markers."""
    active = sorted((e for e in entries if e["status"] in ACTIVE),
                    key=lambda e: (ACTIVE.index(e["status"]), e.get("started_utc") or e["created_utc"]))
    rows = []
    for e in active:
        since = (e.get("started_utc") if e["status"] == "진행" else None) or e["created_utc"]
        tip = " · ".join(x for x in (e.get("description"), f'{e["status"]} since {since[:16].replace("T", " ")} UTC',
                                     e.get("agent"), e.get("host"), e.get("note")) if x)
        cells = []
        for label, cls in cols:
            c = f' class="{cls}"' if cls else ""
            if label == "experiment":
                cells.append(f'<td{c} title="{html.escape(tip, quote=True)}">'
                             f'{html.escape(e["experiment"])} ({e["status"]})</td>')
            elif label == "member":
                cells.append(f'<td{c}>{html.escape(e["member"])}</td>')
            elif label == "#":
                cells.append("<td></td>")
            else:
                cells.append(f'<td{c} data-v=""><span class="dash">–</span></td>')
        rows.append(f'<tr style="color:var(--mut)">{"".join(cells)}</tr>')
    return ROWS_BEGIN + "".join(rows) + ROWS_END


def inject(page: str, entries: list[dict]) -> str:
    """Put the 대기/진행 rows at the top of the ranking table, replacing earlier ones; drop the legacy status table."""
    page = ROWS.sub("", LEGACY_BLOCK.sub("", page))
    found = _ranking(page)
    if not found:
        return page
    i = page.index("<tbody>", found[0]) + len("<tbody>")
    return page[:i] + status_rows(entries, found[2]) + page[i:]


def _cell_value(attrs: str, inner: str) -> float | None:
    m = re.search(r'data-v="([^"]*)"', attrs or "")
    text = html.unescape(m.group(1) if m else re.sub(r"<[^>]+>", "", inner)).replace(",", "").replace("%", "").strip()
    try:
        v = float(text)
    except ValueError:
        return None
    return v if v == v and abs(v) != float("inf") else None


def mark_leaders(page: str) -> str:
    """Bold the best and underline the second-best value of each skill metric column in the ranking table.
    Ties share the mark. Idempotent: old marks are removed first."""
    page = LEADER_MARK.sub(r"\2", page)
    found = _ranking(page)
    if not found:
        return page
    start, stop, cols = found
    head, body = page[start:stop].split("<tbody>", 1)
    rows = list(re.finditer(r"<tr(?:\s[^>]*)?>.*?</tr>", body, re.S))
    cells = [list(CELL.finditer(r.group(0))) for r in rows]
    marks = {}  # (row, column) -> tag
    for col, (label, _) in enumerate(cols):
        sign = -1 if HIGHER_IS_BETTER.search(label) else 1 if LOWER_IS_BETTER.search(label) else 0
        if not sign:
            continue
        values = {i: v for i, cs in enumerate(cells) if col < len(cs)
                  and (v := _cell_value(cs[col].group(1), cs[col].group(2))) is not None}
        ranks = sorted({sign * v for v in values.values()})
        for i, v in values.items():
            if sign * v in ranks[:2]:
                marks[i, col] = "b" if sign * v == ranks[0] else "u"
    out, pos = [], 0
    for i, r in enumerate(rows):
        row = r.group(0)
        for col, m in reversed(list(enumerate(cells[i]))):
            if (i, col) in marks:
                tag = marks[i, col]
                row = row[:m.start(2)] + f'<{tag} class="lb-{1 if tag == "b" else 2}">{m.group(2)}</{tag}>' + row[m.end(2):]
        out += [body[pos:r.start()], row]
        pos = r.end()
    return page[:start] + head + "<tbody>" + "".join(out) + body[pos:] + page[stop:]


def _space_file(repo: str, name: str, revision: str) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        with open(hf_hub_download(repo, name, repo_type="space", revision=revision), encoding="utf-8") as f:
            return f.read()
    except EntryNotFoundError:
        return None


def entries_of(raw: str | None, key: str) -> list[dict]:
    """Entries of a status file: format 2 keeps them under `key`, the first version under "entries"."""
    data = json.loads(raw) if raw else {}
    return data.get(key, data.get("entries", []))


def dump(key: str, entries: list[dict]) -> str:
    return json.dumps({"format": 2, key: entries}, ensure_ascii=False, indent=1) + "\n"


def load(space: str = SPACE) -> tuple[str | None, str | None, str | None, str]:
    """Raw (status.json, status-history.json, index.html) at the Space's head, and that revision."""
    from huggingface_hub import HfApi
    revision = HfApi().repo_info(space, repo_type="space").sha
    return (*(_space_file(space, n, revision) for n in (STATUS_FILE, HISTORY_FILE, "index.html")), revision)


def split(entries: list[dict], history: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep 대기/진행 entries; append finished ones to the bounded history."""
    finished = [e for e in entries if e["status"] not in ACTIVE]
    return [e for e in entries if e["status"] in ACTIVE], (history + finished)[-FINISHED_KEPT:]


def commit(mutate, space: str = SPACE, message: str = "Update experiment status", attempts: int = 5,
           dataset: str = DATASET) -> bool:
    """Apply `mutate(entries) -> entries`, retire entries now on the leaderboard and refresh the rows and marks in
    one commit; retry when another publish lands first."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.utils import HfHubHTTPError
    api = HfApi()
    for attempt in range(attempts):
        status_raw, history_raw, page, parent = load(space)
        new = mutate(entries_of(status_raw, "active"))
        new, history = split(on_leaderboard(new, leaderboard_submissions(new, dataset)),
                             entries_of(history_raw, "finished"))
        ops = [CommitOperationAdd(name, body.encode())
               for name, body, raw in ((STATUS_FILE, dump("active", new), status_raw),
                                       (HISTORY_FILE, dump("finished", history), history_raw)) if body != raw]
        if page is not None:
            visibility = json.loads(_space_file(space, "visibility.json", parent) or "{}")
            architecture = json.loads(_space_file(space, "architecture-code.json", parent) or "{}")
            new_page = bold_experiment_names(
                mark_leaders(hide_ranking_rows(inject(trim_ranking_columns(page), new), visibility)), visibility)
            new_page = apply_architecture_links(new_page, architecture)
            if new_page != page:
                ops.append(CommitOperationAdd("index.html", new_page.encode()))
        if not ops:
            return False
        try:
            api.create_commit(repo_id=space, repo_type="space", commit_message=message, operations=ops,
                              parent_commit=parent)
            return True
        except HfHubHTTPError as exc:
            if getattr(exc.response, "status_code", None) not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("unreachable")


def _print(entries: list[dict]) -> None:
    rows = sorted(entries, key=lambda e: (e["status"] not in ACTIVE, ACTIVE.index(e["status"]) if e["status"] in ACTIVE else 0,
                                          e["updated_utc"]))
    print(f"{'status':6} {'experiment':32} {'member':12} {'agent':12} {'updated (UTC)':20} description")
    for e in rows:
        print(f"{e['status']:6} {e['experiment'][:32]:32} {e['member'][:12]:12} {(e.get('agent') or '-')[:12]:12} "
              f"{e['updated_utc']:20} {e.get('description') or ''}{'  [' + e['note'] + ']' if e.get('note') else ''}")


def _run(a) -> int:
    """Mark 진행 and run the command. rc 0 keeps 진행 until the result is on the leaderboard; otherwise 중단
    (SIGTERM/SIGINT) or 실패. Posting errors only warn."""
    def post(status, **fields):
        try:
            commit(lambda es: upsert(es, a.member, a.experiment, status, **fields), a.space,
                   f"{status}: {a.member}/{a.experiment}")
        except Exception as exc:  # never lose a run to a Hub outage; the next set/sync repairs the page
            print(f"lbstatus: WARNING could not post {status} for {a.experiment}: {exc}", file=sys.stderr)

    post("진행", description=a.desc, agent=a.agent or detect_agent(), host=socket.gethostname(),
         command=shlex.join(a.cmd)[:500], note=a.note)
    try:
        child = subprocess.Popen(a.cmd)
    except OSError as exc:
        post("실패", note=f"launch failed: {exc}")
        raise
    stopped = []

    def forward(sig, _frame):
        stopped.append(sig)
        child.send_signal(sig)

    def keep_rows():  # other publishers drop the rows while this runs; sync commits only if they are missing/stale
        while not done.wait(a.sync_every):
            try:
                commit(lambda es: es, a.space, "Restore experiment status rows")
            except Exception as exc:
                print(f"lbstatus: WARNING sync failed: {exc}", file=sys.stderr)
    done = threading.Event()
    if a.sync_every > 0:
        threading.Thread(target=keep_rows, daemon=True).start()
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        rc = child.wait()
    finally:
        done.set()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if rc == 0:
        post("진행", note="run finished (rc=0); awaiting leaderboard submission" + (f"; {a.note}" if a.note else ""))
    else:
        post("중단" if stopped else "실패", note=f"rc={rc}" + (f"; {a.note}" if a.note else ""))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lbstatus", description=__doc__.split("\n\n")[0])
    ap.add_argument("--space", default=SPACE)
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("set", "run", "rm"):
        p = sub.add_parser(name)
        p.add_argument("experiment")
        p.add_argument("--member", default=DEFAULT_MEMBER, help="leaderboard team_member (env SUNRUN_MEMBER)")
        if name != "rm":
            p.add_argument("--desc", default="", help="one line: what the run tests")
            p.add_argument("--note", default="", help="e.g. result, log path, failure reason")
            p.add_argument("--agent", default="", help="codex / claude-code (auto-detected from the environment)")
    sub.choices["set"].add_argument("--status", required=True, help="대기 | 진행 | 완료 | 실패 | 중단")
    sub.choices["run"].add_argument("--sync-every", type=int, default=300,
                                    help="seconds between row checks while running (0 = off)")
    sub.add_parser("list")
    sub.add_parser("sync")
    argv = sys.argv[1:] if argv is None else list(argv)
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []  # split before argparse so run's options stay ours
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    if a.action == "run":
        if not cmd:
            ap.error("run needs a command after --")
        a.cmd = cmd
        return _run(a)
    try:
        if a.action == "list":
            status_raw, history_raw = load(a.space)[:2]
            _print(entries_of(status_raw, "active") + entries_of(history_raw, "finished")[-20:])
            return 0
        if a.action == "sync":
            changed = commit(lambda es: es, a.space, "Restore experiment status rows")
        elif a.action == "rm":
            changed = commit(lambda es: remove(es, a.member, a.experiment), a.space, f"Remove {a.member}/{a.experiment}")
        else:
            status = normalize_status(a.status)
            fields = {"description": a.desc, "agent": a.agent or detect_agent(), "note": a.note}
            if status in ACTIVE:
                fields["host"] = socket.gethostname()
            changed = commit(lambda es: upsert(es, a.member, a.experiment, status, **fields), a.space,
                             f"{status}: {a.member}/{a.experiment}")
    except ValueError as exc:
        print(f"lbstatus: {exc}", file=sys.stderr)
        return 2
    print(f"lbstatus: {a.action} {'committed' if changed else 'no change'} -> https://huggingface.co/spaces/{a.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

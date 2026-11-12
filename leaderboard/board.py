"""Static leaderboard: render results (from the private HF dataset) to one HTML page for a Static Space.

Every render marks the best value of each skill metric column in bold and the second best underlined, so the page a
publisher writes keeps its marks; lbstatus.py shares the same column rule (`mark_sign`) to re-apply them to pages
published by older clients or rewritten after rows or columns changed."""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

REGIMES = [f"{t}_{s}_{f}" for t in ("low", "high") for s in ("low", "high") for f in ("low", "high")]

HIGHER_IS_BETTER = re.compile(r"^(CSI|skill vs naive)$")
# The regime columns of older pages were skill.
LOWER_IS_BETTER = re.compile(r"MSE|MAE|≥\d+|^(low|high)_(low|high)_(low|high)$")
COMPETITION_VERSION = "lead-timse-6h-v1"
METRIC_COLUMNS = [f"{scope} {metric}" for metric in ("S", "TIMSE", "RMSE")
                  for scope in ("Total", "F1", "F2", "F3")]
BOARD_COLUMNS = ["#", "experiment", *METRIC_COLUMNS, "lead curves", "visualization", "member", "code"]
# Link columns open a page on GitHub; they are not sortable.
LINK_COLUMNS = {"lead curves", "visualization"}
VISUALIZATION_FILE = "visualizations.json"
# Pictures live in research/<experiment>/visualization since 2026-11-12; links pinned to older commits keep the
# former docs/visualizations/<folder> path.
VISUALIZATION_URL = re.compile(r"https://github\.com/TLAB-HGU/sunrun-leaderboard/(tree|blob)/[0-9a-f]{40}/"
                               r"(docs/visualizations/[^?#\s\"<>]+|research/[^/?#\s\"<>]+/visualization)")
CELL = re.compile(r"<td(\s[^>]*)?>(.*?)</td>", re.S)
# Name tags dropped from the experiment cell text (user instruction 2026-11-10). The stored identity is unchanged and
# kept in the cell's data-experiment attribute, which every page edit reads through experiment_id().
DISPLAY_DROP_SUFFIXES = ("_mean3",)


def display_name(experiment: str) -> str:
    for suffix in DISPLAY_DROP_SUFFIXES:
        if experiment.endswith(suffix):
            return experiment[:-len(suffix)]
    return experiment


def experiment_id(cell: re.Match) -> str:
    """The stored experiment name of a ranking cell: its data-experiment attribute, else its text."""
    tag = re.search(r'data-experiment="([^"]*)"', cell.group(1) or "")
    if tag:
        return html.unescape(tag.group(1))
    return html.unescape(re.sub(r"<[^>]+>", "", cell.group(2))).strip()

CSS = """
:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--line:#e3e3e6;--acc:#2b6cb0;--ok:#1a7f4b;--bad:#b3261e;--head:#f6f6f8}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececee;--mut:#9a9aa1;--line:#2d2d31;--acc:#7fb2f0;--ok:#4cc38a;--bad:#f0837b;--head:#1c1c20}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding:24px 16px}
main{max-width:1200px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--head);cursor:pointer;position:sticky;top:0;user-select:none}th:hover{color:var(--acc)}
td.l,th.l{text-align:left}tr:last-child td{border-bottom:0}.ok{color:var(--ok);font-weight:600}.no{color:var(--mut)}
.dash{color:var(--mut)}footer{color:var(--mut);font-size:12.5px;margin-top:14px}code{font-size:12.5px}
"""

JS = """
document.querySelectorAll('th').forEach((th,i)=>th.addEventListener('click',()=>{
 if(th.dataset.sortable==='0')return;
 const tb=th.closest('table').tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 const v=r=>{const t=(r.cells[i].dataset.v??r.cells[i].textContent).trim();
  if(!t||t==='–')return null;const n=Number(t);return Number.isFinite(n)?n:t};
 rows.sort((a,b)=>{const ap=a.cells[0].textContent.trim()==='',bp=b.cells[0].textContent.trim()==='';
  if(ap!==bp)return ap?-1:1;const x=v(a),y=v(b);if(x===null||y===null)return x===y?0:x===null?1:-1;
  return (x>y?1:x<y?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r))}));
"""


def ranking(page: str) -> tuple[int, int, list[tuple[str, str]]] | None:
    """(start, stop, [(header label, header class)]) of the ranking table: the first <table> of a published page.
    lbstatus.py edits the rows and columns through it, so both sides agree on what the ranking table is."""
    opening = re.search(r"<table(?:\s[^>]*)?>", page)
    if opening is None:
        return None
    start = opening.start()
    stop = page.find("</table>", opening.end())
    if stop < 0 or "<tbody>" not in page[start:stop]:
        return None
    head = page[start:stop].split("<tbody>", 1)[0]
    cols = []
    for attrs, inner in re.findall(r"<th(\s[^>]*)?>(.*?)</th>", head, re.S):
        cls = re.search(r'class="([^"]*)"', attrs or "")
        cols.append((html.unescape(re.sub(r"<[^>]+>", "", inner)).strip(), cls.group(1) if cls else ""))
    return start, stop, cols


def apply_architecture_links(page: str, mapping: dict) -> str:
    """Replace code links only for an exact member, experiment, and full config SHA.

    The SHA comes from the code cell's `data-config` attribute (the board no longer publishes a config column) or, on
    a page an older client rendered, from that column; a row still linking its replay script is identified by its path."""
    models = {(e["member"], e["experiment"], e["config_sha256"]): e["url"]
              for e in mapping.get("models", [])}
    found = ranking(page)
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
        member, experiment = text(cells[member_col]), experiment_id(cells[experiment_col])
        cell = cells[code_col]
        tag = re.search(r'data-config="([0-9a-f]{64})"', cell.group(1) or "")
        config = tag.group(1) if tag else None
        if config is None and "config" in labels:  # a page published before the config column was retired
            title = re.search(r'title="([0-9a-f]{64})"', cells[labels.index("config")].group(1) or "")
            if title:
                config = title.group(1)
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


def script_cells(page: str) -> list[tuple[str, str]]:
    """(experiment, link or "") of every ranking row whose code cell is not an architecture link: the cells that
    board.render labels "script" (or leaves empty), listed so a publish can report and repair them."""
    found = ranking(page)
    if found is None:
        return []
    start, stop, cols = found
    labels = [label for label, _ in cols]
    if not all(label in labels for label in ("experiment", "code")):
        return []
    experiment_col, code_col = labels.index("experiment"), labels.index("code")
    out = []
    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", page[start:stop], flags=re.S):
        cells = list(CELL.finditer(row))
        if len(cells) != len(cols):
            continue
        text = lambda cell: html.unescape(re.sub(r"<[^>]+>", "", cell.group(2))).strip()
        experiment = experiment_id(cells[experiment_col])
        if re.search(r"\((진행|대기|완료|실패|중단)\)$", experiment):  # a status row carries dashes, not a code link
            continue
        if text(cells[code_col]) == "architecture":
            continue
        href = re.search(r'href="([^"]+)"', cells[code_col].group(2))
        out.append((experiment, html.unescape(href.group(1)) if href else ""))
    return out


def _latest(results: list[dict]) -> list[dict]:
    """One row per experiment identity; collapse duplicate legacy records only."""
    by = {}
    for r in sorted(results, key=lambda r: r["submitted_utc"]):
        k = (r["team_member"], r["experiment"], r.get("config_sha256", "legacy"))
        by[k] = r
    return list(by.values())


def is_legacy(r: dict) -> bool:
    """Scored before the 2026-10-31 re-split (F1 = 03..05, F2 = 06..09): no F3 value, so its F1/F2 columns mean other windows."""
    return r.get("mse_f3") is None


def has_f1(r: dict) -> bool:
    """Scored on the complete F1 + F2 + F3 (Jan-Sep). Results rescored from predictions that start in March (the
    pre-2026-10-31 evaluation) carry a partial F1 (`coverage_f1` < 1) and are ranked below the complete ones."""
    return (not is_legacy(r) and r.get("mse_f1") is not None
            and all(r.get(f"coverage_{p}", 1.0) == 1.0 for p in ("f1", "f2", "f3")))


def f23_mse(r: dict):
    """Pooled F2 + F3 MSE, the ranking value of results without a complete F1 (None for legacy records)."""
    if is_legacy(r) or r.get("mse_f2") is None:
        return None
    n2, n3 = r.get("n_pairs_f2"), r.get("n_pairs_f3")
    if not n2 or not n3:
        return r["mse_f3"] if not n2 else r["mse_f2"]
    return (r["mse_f2"] * n2 + r["mse_f3"] * n3) / (n2 + n3)


def event_value(r: dict, key: str):
    """HSS event metric shown on the board: over F1 + F2 + F3, or the F2 value when F1 is incomplete (None if not scored)."""
    if is_legacy(r):
        return None
    return r.get(key) if has_f1(r) else r.get(f"{key}_f2", r.get(key))


def competition_metrics(r: dict) -> dict | None:
    """Only complete, finite scores from the current metric version enter this ranking."""
    metrics = r.get("competition")
    if not isinstance(metrics, dict) or metrics.get("version") != COMPETITION_VERSION:
        return None
    for scope in ("total", "f1", "f2", "f3"):
        values = metrics.get(scope)
        if not isinstance(values, dict):
            return None
        for key in ("s", "timse", "rmse"):
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return None
    return metrics


def ranked(results: list[dict]) -> list[dict]:
    """Current six-hour competition results by Total S; unscored submissions follow without a rank."""
    rows = _latest(results)
    return sorted(rows, key=lambda r: (competition_metrics(r) is None,
                                       (competition_metrics(r) or {}).get("total", {}).get("s", math.inf)))


def _num(x, d=1):
    return '<span class="dash">–</span>' if x is None else f"{x:,.{d}f}"


def mark_sign(label: str) -> int:
    """+1 for a lower-is-better skill column, -1 for a higher-is-better one, 0 for rank, identity and cost columns.
    `leader_marks` ranks a column by `sign * value` ascending, so its first two distinct values are the best two."""
    if label in METRIC_COLUMNS:
        return 1
    if HIGHER_IS_BETTER.search(label):
        return -1
    return 1 if LOWER_IS_BETTER.search(label) else 0


def leader_marks(values: list[list[float | None]], head: list[str]) -> dict[tuple[int, int], int]:
    """(row, column) -> 1 for the best and 2 for the second-best value of every skill column; ties share the mark.
    Columns without a skill direction are left unmarked, and so are the dashes of unscored cells."""
    marks: dict[tuple[int, int], int] = {}
    for col, label in enumerate(head):
        sign = mark_sign(label)
        if not sign:
            continue
        column = {i: vs[col] for i, vs in enumerate(values) if col < len(vs) and vs[col] is not None}
        best = sorted({sign * v for v in column.values()})[:2]
        for i, v in column.items():
            if sign * v in best:
                marks[i, col] = 1 if sign * v == best[0] else 2
    return marks


def architecture_mapping(architecture: dict | None = None) -> dict:
    """`architecture-code.json` (member, experiment, full config SHA) -> architecture source, or the given mapping."""
    if architecture is not None:
        return architecture
    architecture_path = Path(__file__).resolve().parent / "architecture-code.json"
    if not architecture_path.exists():
        architecture_path = architecture_path.parent.parent / "architecture-code.json"
    return json.loads(architecture_path.read_text()) if architecture_path.exists() else {}


def architecture_urls(architecture: dict | None = None) -> dict[tuple[str, str, str], str]:
    return {(e["member"], e["experiment"], e["config_sha256"]): e["url"]
            for e in architecture_mapping(architecture).get("models", [])}


def code_link(r: dict, architecture: dict | None = None) -> tuple[str | None, str]:
    """(url, label) of the `code` cell: the study code that defines the architecture, never the submitted replay
    script when an architecture link is known. The mapping wins over the result's own `architecture_url`; a row with
    neither falls back to its recorded `code_url`, labelled script."""
    url = architecture_urls(architecture).get((r["team_member"], r["experiment"], r.get("config_sha256")))
    url = url or r.get("architecture_url")
    return (url, "architecture") if url else (r.get("code_url"), "script")


def visualization_mapping(visualizations: dict | None = None) -> dict:
    """`visualizations.json` {"models": [{member, experiment, submitted_utc, url}]}: the prediction visualisation of
    one stored submission, or the given mapping. Keyed by submission time, so a resubmission never inherits the
    pictures of an older forecast."""
    if visualizations is not None:
        return visualizations
    path = Path(__file__).resolve().parent / VISUALIZATION_FILE
    if not path.exists():
        path = path.parent.parent / VISUALIZATION_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def visualization_url(r: dict, visualizations: dict | None = None) -> str | None:
    """The pinned GitHub folder that visualises exactly this submission, if one is recorded."""
    for e in visualization_mapping(visualizations).get("models", []):
        if ((e.get("member"), e.get("experiment"), e.get("submitted_utc"))
                == (r["team_member"], r["experiment"], r.get("submitted_utc"))
                and VISUALIZATION_URL.fullmatch(e.get("url", ""))):
            return e["url"]
    return None


def upsert_visualization(mapping: dict, entry: dict) -> dict:
    """Mapping with `entry` replacing any entry for the same submission, newest first."""
    key = (entry["member"], entry["experiment"], entry["submitted_utc"])
    if not VISUALIZATION_URL.fullmatch(entry.get("url", "")):
        raise ValueError("visualization url must be a pinned research/<experiment>/visualization path on GitHub: "
                         f"{entry.get('url')}")
    kept = [e for e in mapping.get("models", []) if (e["member"], e["experiment"], e["submitted_utc"]) != key]
    return {**mapping, "models": [dict(entry)] + kept}


def visibility_policy() -> dict:
    """`visibility.json`: the rows hidden from the ranking (hidden) and the experiment names shown in bold. Every
    publish renders with it, and lbstatus.py re-applies it when it rewrites a page."""
    policy_path = Path(__file__).resolve().parent / "visibility.json"
    if not policy_path.exists():
        policy_path = policy_path.parent.parent / "visibility.json"
    return json.loads(policy_path.read_text()) if policy_path.exists() else {}


def hidden_before(member: str, submitted_utc: str | None, policy: dict) -> bool:
    """Hide dated results only; undated rows (including active runs) survive.

    Rules are {"member": ..., "submitted_utc": ...} entries in hidden_before.
    Submission timestamps use the store's compact UTC format, with optional
    fractional seconds. Invalid or absent timestamps cannot establish age.
    """
    def timestamp(value):
        match = re.fullmatch(r"(\d{8}T\d{6})(\d{0,6})Z", value or "")
        return (match[1], match[2].ljust(6, "0")) if match else None

    submitted = timestamp(submitted_utc)
    if submitted is None:
        return False
    return any(rule.get("member") == member
               and (cutoff := timestamp(rule.get("submitted_utc"))) is not None
               and submitted < cutoff for rule in policy.get("hidden_before", []))


def render(results: list[dict], note: str = "", architecture: dict | None = None,
           visualizations: dict | None = None) -> str:
    policy = visibility_policy()
    architecture = architecture_mapping(architecture)
    visualizations = visualization_mapping(visualizations)
    hidden = {(e["member"], e["experiment"]) for e in policy.get("hidden", [])}
    bold = {(e["member"], e["experiment"]) for e in policy.get("bold", [])}
    rows = [r for r in ranked(results) if (r["team_member"], r["experiment"]) not in hidden
            and not hidden_before(r["team_member"], r.get("submitted_utc"), policy)]
    head = BOARD_COLUMNS
    left = {"experiment", "member", "code", *LINK_COLUMNS}
    th = "".join(f'<th class="{"l" if h in left else ""}"'
                 + (' data-sortable="0"' if h in LINK_COLUMNS else '')
                 + f'>{html.escape(h)}</th>' for h in head)
    table: list[list[tuple[str, str, float | None]]] = []
    rank_no = 0
    for r in rows:
        metrics = competition_metrics(r)
        if metrics is not None:
            rank_no += 1
            rank, rank_attrs = str(rank_no), ""
        else:
            rank = "unscored"
            rank_attrs = ' data-v="999999" title="No complete score under the current six-hour evaluation metric"'
        experiment_name = html.escape(display_name(r["experiment"]))
        if (r["team_member"], r["experiment"]) in bold:
            experiment_name = f'<strong class="lb-experiment">{experiment_name}</strong>'
        experiment_attrs = f' class="l" data-experiment="{html.escape(r["experiment"], quote=True)}"'
        cells = [(rank_attrs, rank, None), (experiment_attrs, experiment_name, None)]
        for key in ("s", "timse", "rmse"):
            for scope in ("total", "f1", "f2", "f3"):
                values = metrics[scope] if metrics else {}
                tip = []
                if values.get("replacement_fraction") is not None:
                    tip.append(f'Persistence replacements: {values["replacement_fraction"]:.2%} of valid pairs')
                if values.get("bias") is not None:
                    tip.append(f'Mean forecast bias: {values["bias"]:+.2f} km/s')
                title = html.escape("; ".join(tip), quote=True)
                value = values.get(key)
                attrs = f' data-v="{"" if value is None else value}" title="{title}"'
                cells.append((attrs, _num(value, 2), value))
        plot_url = r.get("metric_plot_url") if metrics else None
        # Every published image is immutable and hosted alongside the reproducible source.
        if plot_url and re.fullmatch(r"https://raw\.githubusercontent\.com/TLAB-HGU/sunrun-leaderboard/[0-9a-f]{40}/leaderboard/assets/metrics/[^?#]+\.png", plot_url):
            github_url = plot_url.replace("https://raw.githubusercontent.com/", "https://github.com/", 1)
            github_url = github_url.replace("/sunrun-leaderboard/", "/sunrun-leaderboard/blob/", 1)
            url = html.escape(github_url, quote=True)
            plot = (f'<a class="lead-curves-link" href="{url}" target="_blank" rel="noopener" '
                    'title="View S, TIMSE and RMSE curves on GitHub">GitHub</a>')
        else:
            plot = '<span class="dash">–</span>'
        viz_url = visualization_url(r, visualizations)
        viz = (f'<a class="visualization-link" href="{html.escape(viz_url, quote=True)}" target="_blank" '
               'rel="noopener" title="Monthly forecast GIFs and lead curves of this submission on GitHub">GitHub</a>'
               if viz_url else '<span class="dash">–</span>')
        code_url, code_label = code_link(r, architecture)
        code = (f'<a href="{html.escape(code_url, quote=True)}" target="_blank" rel="noopener">{code_label}</a>'
                if code_url else '<span class="dash">–</span>')
        cells.extend([
            (' class="l"', plot, None),
            (' class="l"', viz, None),
            (' class="l"', html.escape(r["team_member"]), None),
            (f' data-config="{html.escape(r.get("config_sha256") or "")}" class="l"', code, None),
        ])
        table.append(cells)
    marks = leader_marks([[value for _, _, value in cells] for cells in table], head)
    body = []
    for i, cells in enumerate(table):
        tds = []
        for col, (attrs, inner, _) in enumerate(cells):
            if place := marks.get((i, col)):
                tag = "b" if place == 1 else "u"
                inner = f'<{tag} data-metric-version="{COMPETITION_VERSION}" class="lb-{place}">{inner}</{tag}>'
            tds.append(f"<td{attrs}>{inner}</td>")
        submitted = html.escape(rows[i].get("submitted_utc") or "", quote=True)
        body.append(f'<tr data-submitted-utc="{submitted}">' + "".join(tds) + "</tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solar wind 72h leaderboard</title><style>{CSS}</style></head><body><main>
<h1>Solar wind 72h leaderboard</h1>
<p class="sub">Forecasts issued every 6 hours at 00, 06, 12 and 18 UTC; leads 1–72h.
<b>F1</b>: targets 2026-01-01 .. 03-31; <b>F2</b>: 2026-04-01 .. 06-30; <b>F3</b>: 2026-07-01 .. 09-29.
Only forecasts whose full 72h window lies within one period are evaluated. Rank = <b>Total S</b>, lower is better. {html.escape(note)}</p>
<div class="wrap"><table data-metric-version="{COMPETITION_VERSION}"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<footer>Each displayed S, TIMSE and RMSE is the arithmetic mean of its 72 lead-specific values.
Total pools F1–F3 observations separately at each lead before computing the metrics.
S = √(RMSE × √TIMSE); S and RMSE are in km/s, TIMSE in (km/s)².
TIMSE uses 100 observation-percentile intervals with velocity-width weighting, recomputed at each lead.
Missing observations are excluded without filling. Missing or invalid predictions use the latest valid observation at or before issue time;
hover over metrics for persistence replacement rates and mean forecast bias.
Predictions for F1, F2 and F3 use training data before 2025-12-29, 2026-03-29 and 2026-06-29 00:00 UTC respectively.
Legacy submissions without a complete current score are marked <b>unscored</b> and excluded from the ranking.
The lead curves column links to the image on GitHub; images are not embedded in this table.
The visualization column links to that submission's monthly forecast GIFs and lead curves on GitHub, re-scored from the stored forecast.
In each metric column the best value is <b>bold</b> and the second best <u>underlined</u>; ties share marks.
Click a numeric or identity header to sort.<br>Generated from {len(results)} scored submissions.</footer>
</main><script>{JS}</script></body></html>"""


def load_results(root) -> list[dict]:
    """Load legacy flat and keyed nested result files for local previews."""
    from pathlib import Path
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("*.json"))]


if __name__ == "__main__":  # quick local preview: python board.py results_dir > index.html
    import sys
    print(render(load_results(sys.argv[1])))

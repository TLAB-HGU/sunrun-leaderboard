"""Static leaderboard: render results (from the private HF dataset) to one HTML page for a Static Space.

Every render marks the best value of each skill metric column in bold and the second best underlined, so the page a
publisher writes keeps its marks; lbstatus.py shares the same column rule (`mark_sign`) to re-apply them to pages
published by older clients or rewritten after rows or columns changed."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

REGIMES = [f"{t}_{s}_{f}" for t in ("low", "high") for s in ("low", "high") for f in ("low", "high")]

HIGHER_IS_BETTER = re.compile(r"^(CSI|skill vs naive)$")
# sec/fold, end-to-end (h) and config are cost or identity, not skill; the regime columns of older pages were skill.
LOWER_IS_BETTER = re.compile(r"MSE|MAE|≥\d+|^(low|high)_(low|high)_(low|high)$")

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
 const tb=th.closest('table').tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 const v=r=>{const t=r.cells[i].dataset.v??r.cells[i].textContent;const n=parseFloat(t);return isNaN(n)?t:n};
 rows.sort((a,b)=>{const x=v(a),y=v(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r))}));
"""


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


def end_to_end_hours(r: dict):
    """Optional end-to-end compute time (training through the submitted F1 + F2 + F3 forecasts), from result extras."""
    seconds = (r.get("extras") or {}).get("end_to_end_seconds")
    return None if seconds is None else seconds / 3600


def ranked(results: list[dict]) -> list[dict]:
    """Complete F1+F2+F3 results by overall MSE, then results without a complete F1 by pooled F2+F3 MSE, then legacy ones."""
    rows = _latest(results)
    partial = [r for r in rows if not has_f1(r) and not is_legacy(r)]
    return (sorted([r for r in rows if has_f1(r)], key=lambda r: r["mse"])
            + sorted(partial, key=f23_mse)
            + sorted([r for r in rows if is_legacy(r)], key=lambda r: r["mse"]))


def _num(x, d=1):
    return '<span class="dash">–</span>' if x is None else f"{x:,.{d}f}"


def mark_sign(label: str) -> int:
    """+1 for a lower-is-better skill column, -1 for a higher-is-better one, 0 for rank, identity and cost columns.
    `leader_marks` ranks a column by `sign * value` ascending, so its first two distinct values are the best two."""
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


def render(results: list[dict], note: str = "") -> str:
    policy_path = Path(__file__).resolve().parent / "visibility.json"
    if not policy_path.exists():
        policy_path = policy_path.parent.parent / "visibility.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    architecture_path = Path(__file__).resolve().parent / "architecture-code.json"
    if not architecture_path.exists():
        architecture_path = architecture_path.parent.parent / "architecture-code.json"
    architecture = json.loads(architecture_path.read_text()) if architecture_path.exists() else {}
    architecture_urls = {(e["member"], e["experiment"], e["config_sha256"]): e["url"]
                         for e in architecture.get("models", [])}
    hidden = {(e["member"], e["experiment"]) for e in policy.get("hidden", [])}
    bold = {(e["member"], e["experiment"]) for e in policy.get("bold", [])}
    rows = [r for r in ranked(results) if (r["team_member"], r["experiment"]) not in hidden]
    head = ["#", "experiment",
            "MSE", "F1 MSE", "F2 MSE", "F3 MSE", "MSE ≥550", "F1 ≥550", "F2 ≥550", "F3 ≥550",
            "CSI", "timing MAE (h)", "MSE obs.", "skill vs naive",
            "member", "sec/fold", "config", "end-to-end (h)", "code"]
    th = "".join(f'<th class="{"l" if i in (1, 14, 16, 18) else ""}">{html.escape(h)}</th>' for i, h in enumerate(head))
    table: list[list[tuple[str, str, float | None]]] = []  # (cell attributes, cell html, value for the leader marks)
    rank_no = 0
    for r in rows:
        full, legacy = has_f1(r), is_legacy(r)
        rank_no += full
        if full:
            rank, rank_attrs = str(rank_no), ""
        elif legacy:
            rank = "legacy"
            rank_attrs = ' data-v="99999" title="scored on the pre-2026-10-31 windows (F1 03..05, F2 06..09); resubmit to be ranked on F1-F3"'
        else:
            rank = "no F1"
            rank_attrs = ' data-v="9999" title="rescored from earlier predictions: F1 (2026-01..03) is mostly missing and F2 lacks the origins of 05-29..05-31, so F1 and the overall values are not shown; listed by pooled F2+F3 MSE; resubmit complete F1-F3 predictions to be ranked"'
        keys = ("mse", "mse_f1", "mse_f2", "mse_f3", "mse_ge550", "mse_f1_ge550", "mse_f2_ge550", "mse_f3_ge550")
        metrics = [None if legacy else r.get(k) for k in keys]
        if not full and not legacy:  # an incomplete F1 and the overall values that include it are not shown
            for i in (0, 1, 4, 5):
                metrics[i] = None
        csi, tmae, bias = (event_value(r, k) for k in ("csi", "timing_mae_h", "timing_bias_h"))
        bias_title = "" if bias is None else f"mean timing error {bias:+.1f} h (+ = forecast peak late)"
        e2e = end_to_end_hours(r)
        e2e_title = html.escape((r.get("extras") or {}).get("end_to_end_note", ""), quote=True)
        skill = None if legacy else (r.get("skill_vs_naive") or 0)
        experiment_name = html.escape(r["experiment"])
        if (r["team_member"], r["experiment"]) in bold:
            experiment_name = f'<strong class="lb-experiment">{experiment_name}</strong>'
        config_sha = r.get("config_sha256")
        config_label = config_sha[:12] if config_sha else "legacy"
        seconds = r.get("inference_seconds_per_fold")
        architecture_url = architecture_urls.get((r["team_member"], r["experiment"], config_sha))
        code_url = architecture_url or r.get("code_url")
        code_label = "architecture" if architecture_url else "script"
        code = (f'<a href="{html.escape(code_url, quote=True)}" target="_blank" rel="noopener">{code_label}</a>'
                if code_url else '<span class="dash">–</span>')
        table.append([
            (rank_attrs, rank, None),
            (' class="l"', experiment_name, None),
            *((f' data-v="{"" if x is None else x}"', _num(x), x) for x in metrics),
            (f' data-v="{"" if csi is None else csi}"', _num(csi, 3), csi),
            (f' data-v="{"" if tmae is None else tmae}" title="{bias_title}"', _num(tmae), tmae),
            ("", _num(r.get("mse_observed")), r.get("mse_observed")) if not legacy else ("", _num(None), None),
            (f' data-v="{"" if skill is None else skill}"', '<span class="dash">–</span>' if skill is None else f"{skill:+.1%}", skill),
            (' class="l"', html.escape(r["team_member"]), None),
            (f' data-v="{seconds if seconds is not None else ""}"', _num(seconds, 6), None),
            (f' class="l" title="{html.escape(config_sha or "legacy")}"', f"<code>{config_label}</code>", None),
            (f' data-v="{"" if e2e is None else e2e}" title="{e2e_title}"', _num(e2e, 2), None),
            (' class="l"', code, None),
        ])
    marks = leader_marks([[value for _, _, value in cells] for cells in table], head)
    body = []
    for i, cells in enumerate(table):
        tds = []
        for col, (attrs, inner, _) in enumerate(cells):
            if place := marks.get((i, col)):  # best column value in bold, second best underlined
                tag = "b" if place == 1 else "u"
                inner = f'<{tag} class="lb-{place}">{inner}</{tag}>'
            tds.append(f"<td{attrs}>{inner}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solar wind 72h leaderboard</title><style>{CSS}</style></head><body><main>
<h1>Solar wind 72h leaderboard</h1>
<p class="sub">Hourly origins, 72h windows. <b>F1</b> targets 2026-01-01 00:00 .. 2026-03-31 23:00 UTC (2,089 folds),
<b>F2</b> targets 2026-04-01 00:00 .. 2026-06-30 23:00 UTC (2,113 folds), <b>F3</b> targets 2026-07-01 00:00 .. 2026-09-29 23:00 UTC (2,113 folds).
Rank = MSE over F1 + F2 + F3. Lower MSE (km/s)² is better. {html.escape(note)}</p>
<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<footer>Rank = overall MSE over F1 + F2 + F3 (counts forward-filled targets, same as the baseline study). <b>≥550</b> columns score only targets whose true speed is ≥ 550 km/s.
Predictions for F1, F2 and F3 must come from models trained on data before 2025-12-29, 2026-03-29 and 2026-06-29 00:00 UTC respectively.
<b>no F1</b> rows were rescored from earlier predictions that start in March and skip the origins of 2026-05-29..05-31, so their F1 is incomplete and F2 misses those 3 days: F1 and the overall values are not shown, they are listed below the ranking by pooled F2+F3 MSE,
and their CSI / timing are F2 values. <b>legacy</b> rows could not be rescored and keep no values. <b>MSE obs.</b> scores only originally observed targets.
<b>CSI</b> (threat score) = TP / (TP + FP + FN) for HSS events in each 72h window: a run of hours ≥ 550 km/s (quiet gaps ≤ 6h merged; runs touching
hour 1 or 72 are not counted), forecast and true peaks matched one-to-one within 12h, summed over all windows. <b>timing MAE</b> = mean |forecast − true
peak hour| of matched events (hover: signed bias, + = late). Both are diagnostic; rows without a complete F1 show F2 values.
<b>end-to-end (h)</b> = wall-clock compute from training through the submitted F1 + F2 + F3 forecasts, as reported by the submitter
(optional, shown only where provided; hover for its scope). In every skill column the best value is <b>bold</b> and the
second best <u>underlined</u> (ties share the mark); sec/fold, end-to-end (h) and the ranking columns are not marked.
Click a header to sort.<br>Generated from {len(results)} scored submissions.</footer>
</main><script>{JS}</script></body></html>"""


def load_results(root) -> list[dict]:
    """Load legacy flat and keyed nested result files for local previews."""
    from pathlib import Path
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("*.json"))]


if __name__ == "__main__":  # quick local preview: python board.py results_dir > index.html
    import sys
    print(render(load_results(sys.argv[1])))

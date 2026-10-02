"""Static leaderboard: render results (from the private HF dataset) to one HTML page for a Static Space."""
from __future__ import annotations

import html
import json

REGIMES = [f"{t}_{s}_{f}" for t in ("low", "high") for s in ("low", "high") for f in ("low", "high")]

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


def _num(x, d=1):
    return '<span class="dash">–</span>' if x is None else f"{x:,.{d}f}"


def render(results: list[dict], note: str = "") -> str:
    rows = sorted(_latest(results), key=lambda r: r["mse"])
    head = ["#", "experiment", "member", "config", "sec/fold", "code", "MSE", "MSE obs.", "skill vs naive", "beats naive (95% CI)", "regime-bal. MSE", "submitted (UTC)"] + REGIMES
    th = "".join(f'<th class="{"l" if i in (1, 2, 3, 5, 11) else ""}">{html.escape(h)}</th>' for i, h in enumerate(head))
    body = []
    for i, r in enumerate(rows, 1):
        vs = r.get("vs_naive") or {}
        beat = vs.get("beats_reference")
        config_sha = r.get("config_sha256")
        config_label = config_sha[:12] if config_sha else "legacy"
        seconds = r.get("inference_seconds_per_fold")
        code_url = r.get("code_url")
        code = (f'<a href="{html.escape(code_url, quote=True)}" target="_blank" rel="noopener">script</a>'
                if code_url else '<span class="dash">–</span>')
        cells = [
            f"<td>{i}</td>", f'<td class="l">{html.escape(r["experiment"])}</td>', f'<td class="l">{html.escape(r["team_member"])}</td>',
            f'<td class="l" title="{html.escape(config_sha or "legacy")}"><code>{config_label}</code></td>',
            f'<td data-v="{seconds if seconds is not None else ""}">{_num(seconds, 6)}</td>', f'<td class="l">{code}</td>',
            f'<td data-v="{r["mse"]}">{r["mse"]:,.1f}</td>', f'<td>{_num(r.get("mse_observed"))}</td>',
            f'<td data-v="{r.get("skill_vs_naive") or 0}">{(r.get("skill_vs_naive") or 0):+.1%}</td>',
            f'<td class="{"ok" if beat else "no"}">{"yes" if beat else "no"}</td>',
            f'<td>{_num(r.get("regime_balanced_mse"))}</td>',
            f'<td class="l">{html.escape(r["submitted_utc"][:15])}</td>',
        ]
        for g in REGIMES:
            c = r["by_regime"][g]
            cells.append(f'<td>{_num(c["mse"])}</td>' if c["reliable"] else '<td class="dash" title="fewer than 3 independent 72h blocks">–</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solar wind 72h leaderboard</title><style>{CSS}</style></head><body><main>
<h1>Solar wind 72h leaderboard</h1>
<p class="sub">Hourly origins, 72h windows, targets 2026-06-01 00:00 .. 2026-09-29 23:00 UTC (2,833 folds; fixed). Lower MSE (km/s)² is better. {html.escape(note)}</p>
<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>
<footer>Rank = overall MSE (fills forward-filled targets, same as the baseline study). <b>MSE obs.</b> scores only originally observed targets.
<b>beats naive</b> = block-bootstrap (72h blocks) 95% CI of the paired MSE difference is below 0. Regime columns (trend/seasonal-27d/forecastability, low/high) are
diagnostic and hidden when a cell has fewer than 3 non-overlapping 72h blocks. Click a header to sort.<br>Generated from {len(results)} scored submissions.</footer>
</main><script>{JS}</script></body></html>"""


def load_results(root) -> list[dict]:
    """Load legacy flat and keyed nested result files for local previews."""
    from pathlib import Path
    return [json.loads(p.read_text()) for p in sorted(Path(root).rglob("*.json"))]


if __name__ == "__main__":  # quick local preview: python board.py results_dir > index.html
    import sys
    print(render(load_results(sys.argv[1])))

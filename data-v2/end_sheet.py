"""Write the latest actual data time (reports/end-dates-<date>.json) into SolarWind DB column '끝'.

Run inside the research-sheet wrapper env:
    sh -c '. /dev/stdin; python3 end_sheet.py' < <(sed -e '/^exec/d' -e '/^#!/d' ~/.local/bin/research-sheet)
"""
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".local/share/research-sheet"))
import research_sheet as R  # noqa: E402

SID = "1KWgNQPlajHWw7FLUMz3lr4rO2z7sxuxXGstD0-YjAw4"
TAB = "데이터셋 manifest"
ROOT = Path(__file__).resolve().parent
E = json.loads(sorted((ROOT / "reports").glob("end-dates-*.json"))[-1].read_text())
LB = Path("/home/t-lab01/sunrun/leaderboard")


def fmt(iso, copy):
    t = dt.datetime.fromisoformat(iso).astimezone(dt.timezone.utc)
    return f"{t:%Y-%m-%d %H:%M} UTC ({copy})"


def gong_manifest_end(path):
    m = json.loads(Path(path).read_text())
    vals = [x.get("date_obs_utc") or x.get("observation_time_utc") for x in m.get("maps", []) if isinstance(x, dict)]
    vals = [v for v in vals if v]
    return max(vals) if vals else None


hcs = gong_manifest_end(LB / "experiments/cmehcs72_20261014/gong_manifest.json")
pfc = gong_manifest_end(LB / "experiments/pfssconn_20261014/gong_manifest.json")
VALUES = {sw: fmt(v["latest_utc"], v["copy"]) for sw, v in E.items() if v.get("latest_utc")}
VALUES.update({
    "SW-017": "2017-07-25 (ACE SWEPAM L3 strahl 원천 종료; 실험 피처는 모두 결측)",
    "SW-019": (fmt(pfc, "v1 실험 창") if pfc else "2026-09-26 (v1 실험 창)") + "; v2 생성 중",
    "SW-020": (fmt(hcs, "v1 실험 창") if hcs else "2026-09-30 (v1 실험 창)") + "; v2 계산 시간 초과로 중단",
    "SW-021": (fmt(hcs, "v1 실험 창") if hcs else "2026-09-30 (v1 실험 창)") + "; v2 생성 중",
    "SW-022": (fmt(hcs, "v1 실험 창") if hcs else "2026-09-30 (v1 실험 창)") + "; v2는 CH v2 완료 후 생성",
    "SW-023": (fmt(hcs, "v1 실험 창") if hcs else "2026-09-30 (v1 실험 창)") + "; v2 생성 중",
    "SW-026": fmt(E["SW-009"]["latest_utc"], "원천 SW-009 v2 기준"),
    "SW-055": "- (미수집)",
    "SW-058": (fmt(hcs, "v1 실험 창") if hcs else "2026-09-30 (v1 실험 창)") + "; 원천 GONG v2는 SW-012",
})

vals = R.gws(["values"], "get", {"spreadsheetId": SID, "range": f"'{TAB}'!A1:Z200"}).get("values", [])
head = vals[0]
col = head.index("끝")
letter = chr(65 + col)
ids = [r[0] if r else "" for r in vals]
missing = [i for i in ids if i.startswith("SW-") and i not in VALUES]
if missing:
    raise SystemExit(f"no end value for {missing}")
(ROOT / "reports/sheet-before-end.json").write_text(json.dumps(vals, ensure_ascii=False, indent=1))
column = [[head[col]]] + [[VALUES[i] if i in VALUES else (r[col] if len(r) > col else "")] for i, r in zip(ids[1:], vals[1:])]
rng = f"'{TAB}'!{letter}1:{letter}{len(column)}"
R.gws(["values"], "update", {"spreadsheetId": SID, "range": rng, "valueInputOption": "RAW"},
      {"range": rng, "majorDimension": "ROWS", "values": column})
back = R.gws(["values"], "get", {"spreadsheetId": SID, "range": f"'{TAB}'!A1:Z200"}).get("values", [])
for old, new in zip(vals, back):
    o = old + [""] * (len(head) - len(old)); n = new + [""] * (len(head) - len(new))
    assert o[:col] == n[:col] and o[col + 1:] == n[col + 1:], o[0]
    if o[0] in VALUES:
        assert n[col] == VALUES[o[0]], o[0]
print(json.dumps({"column": letter, "rows": len(VALUES), "verified": True}, ensure_ascii=False))
for i in ids[1:]:
    if i in VALUES:
        print(i, VALUES[i])

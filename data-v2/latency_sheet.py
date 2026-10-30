"""Insert the '입수지연' column right after '인과(가용 시각) 규칙' in SolarWind DB and fill it per SW ID.

Values combine measured latency (reports/latency-<date>.json, built by latency_report.py) with the
research rule. Run inside the research-sheet wrapper env (gws credentials):

    sh -c '. /dev/stdin; python3 latency_sheet.py [--dry-run]' < <(sed -e '/^exec/d' -e '/^#!/d' ~/.local/bin/research-sheet)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".local/share/research-sheet"))
import research_sheet as R  # noqa: E402

SID = "1KWgNQPlajHWw7FLUMz3lr4rO2z7sxuxXGstD0-YjAw4"
TAB = "데이터셋 manifest"
ROOT = Path(__file__).resolve().parent
L = json.loads(sorted((ROOT / "reports").glob("latency-*.json"))[-1].read_text())


def h(x):
    """hours -> short Korean text."""
    if x < 1:
        return f"{max(round(x * 60), 1)}분"
    if x < 48:
        return f"{x:.1f}h"
    return f"{x / 24:.1f}일"


def meas(key, what):
    d = L[key]
    return f"실측 중앙 {h(d['median_h'])} (p90 {h(d['p90_h'])}, n={d['n']:,}, {d.get('period', '')}, {what})"


ACE = ("실측: daily list가 일 끝+11분에 생성(n=499, 2025-06~2026-10), 실시간 2-hour list는 2~3분 "
       "/ 규칙 관측+2h")
SUVI = f"{meas('suvi195', 'S3 LastModified−obs_end')} / 규칙 obs_end+1h 및 S3 LastModified≤origin"
GONG = f"{meas('gong', 'Last-Modified−지도 시각')} / 규칙 max(관측+2h, Last-Modified)"
ENLIL = f"{meas('donki_enlil', 'modelCompletionTime−입력 CME 시작')} / 규칙 modelCompletionTime≤origin"
VALUES = {
    "SW-001": ACE, "SW-002": ACE, "SW-003": ACE, "SW-004": ACE,
    "SW-005": "원천 SW-008과 같음 (SUVI 실측 약 1분) / 규칙 SUVI proxy(obs_end+1h)",
    "SW-006": SUVI, "SW-007": SUVI, "SW-008": SUVI,
    "SW-009": ENLIL, "SW-010": ENLIL,
    "SW-011": f"원천 SW-012와 같음: {GONG}", "SW-012": GONG,
    "SW-013": L["rules"]["k1"], "SW-014": L["rules"]["l2"],
    "SW-015": "해당 없음 (ACE 공개 시각 증거 자료)",
    "SW-016": f"원천 ACE와 같음: {ACE}", "SW-017": f"원천 ACE와 같음: {ACE}",
    "SW-018": f"원천 SW-012와 같음: {GONG}", "SW-019": f"원천 SW-012와 같음: {GONG}",
    "SW-020": f"원천 SW-012와 같음: {GONG}", "SW-021": f"원천 SW-012·SW-025와 같음: {GONG}",
    "SW-022": f"원천 SW-012·SW-008과 같음: {GONG}", "SW-023": f"원천 SW-012·SW-008·SW-025와 같음: {GONG}",
    "SW-024": (f"{meas('enlil_ambient', 'archive Last-Modified−init 00:00')} / 규칙 init+12h 또는 Last-Modified "
               "— init+12h는 실측보다 짧으므로 Last-Modified 기준이 안전"),
    "SW-025": (f"실측 CMEAnalysis 중앙 {h(L['donki_cmeanalysis']['median_h'])} (p90 {h(L['donki_cmeanalysis']['p90_h'])}, "
               f"n={L['donki_cmeanalysis']['n']:,}, time21_5 기준), CME 기록 중앙 {h(L['donki_cme']['median_h'])} "
               f"(p90 {h(L['donki_cme']['p90_h'])}, n={L['donki_cme']['n']:,}, startTime 기준), 2026-06~09 "
               "/ 규칙 submissionTime≤origin"),
    "SW-026": f"원천 SW-009와 같음: {ENLIL}",
    "SW-027": f"{meas('ccmc', 'submissionTime−CME 시각')} / 규칙 submissionTime≤origin",
    "SW-028": f"{meas('lasco', 'Last-Modified−프레임 시각')} / 규칙 Last-Modified(약 관측+1h)",
    "SW-029": L["rules"]["cactus"],
    "SW-030": f"{meas('hi_1', 'Last-Modified−DATE-OBS')} / 규칙 DATE-END+1h (2026-10-15 실측 p90의 약 2배)",
    "SW-031": f"{meas('hi_2', 'Last-Modified−DATE-OBS')} / 규칙 DATE-END+1h (2026-10-15 실측 p90의 약 2배)",
    "SW-032": ("실측: 파일이 매일 약 17:20 UT에 재생성(10-13 17:20, 10-14 17:16), 관측은 20:00~09:00 UT라 "
               f"관측→입수 약 8~22h (마지막 관측 기준 중앙 {h(L['isee']['median_h'])}) "
               "/ 규칙 관측 시각 이후 처음 오는 18:00 UT"),
    "SW-033": (f"{meas('punch', 'Last-Modified−관측')}; 2026-06 초기분은 07-28 일괄 공개(수 주) "
               "/ 규칙 max(FITS DATE, Last-Modified, DATE-END+1h)"),
    "SW-034": (f"실측 NGDC 아카이브 일 끝+{h(L['swpc_events']['median_h'])} (n={L['swpc_events']['n']}, 2026-06~09), "
               "실시간 텍스트는 수 분 / 규칙 보고서 생성 시각"),
    "SW-035": L["rules"]["xrs"],
    "SW-036": ("실측 첫 공개 일 끝+4.5h, 같은 파일명으로 +2.2일·+6.2일 재발행(최신본 기준이면 약 6일) "
               "/ 규칙 manifest의 버전별 Last-Modified"),
    "SW-037": L["rules"]["hmi"], "SW-038": L["rules"]["nmdb"],
    "SW-039": f"{meas('suvi171', 'S3 LastModified−obs_end')} / 규칙 SUVI proxy(obs_end+1h)",
    "SW-040": "원천 SW-008과 같음 (SUVI 실측 약 1분) / 규칙 SUVI proxy",
    "SW-041": "원천 SW-008과 같음 (SUVI 실측 약 1분) / 규칙 SUVI proxy",
    "SW-042": f"실측 발행 즉시 (NGDC Last-Modified−발행 {h(max(L['discussion']['median_h'], 0.001))}, n={L['discussion']['n']}) / 규칙 발행 시각≤origin",
    "SW-055": "-", "SW-057": f"원천 SW-012와 같음: {GONG}", "SW-058": f"원천 SW-012와 같음: {GONG}",
}


def main(dry):
    vals = R.gws(["values"], "get", {"spreadsheetId": SID, "range": f"'{TAB}'!A1:Z200"}).get("values", [])
    head = vals[0]
    if "입수지연" in head:
        raise SystemExit("입수지연 column already present")
    rule_col = head.index("인과(가용 시각) 규칙")
    ids = [r[0] if r else "" for r in vals]
    sw = [i for i in ids if i.startswith("SW-")]
    missing = [i for i in sw if i not in VALUES]
    if missing:
        raise SystemExit(f"no latency value for {missing}")
    (ROOT / "reports/sheet-before-latency.json").write_text(json.dumps(vals, ensure_ascii=False, indent=1))
    col = rule_col + 1
    column = [["입수지연"]] + [[VALUES.get(i, "")] for i in ids[1:]]
    if dry:
        print(f"insert at column index {col}; {len(sw)} SW rows"); return
    meta = R.gws([], "get", {"spreadsheetId": SID, "fields": "sheets.properties"})
    sheet_id = next(s["properties"]["sheetId"] for s in meta["sheets"] if s["properties"]["title"] == TAB)
    R.gws([], "batchUpdate", {"spreadsheetId": SID}, {"requests": [
        {"insertDimension": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": col, "endIndex": col + 1},
                             "inheritFromBefore": True}},
        {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": col, "endIndex": col + 1},
                                       "properties": {"pixelSize": 300}, "fields": "pixelSize"}}]})
    letter = chr(65 + col)
    rng = f"'{TAB}'!{letter}1:{letter}{len(column)}"
    R.gws(["values"], "update", {"spreadsheetId": SID, "range": rng, "valueInputOption": "RAW"},
          {"range": rng, "majorDimension": "ROWS", "values": column})
    back = R.gws(["values"], "get", {"spreadsheetId": SID, "range": f"'{TAB}'!A1:Z200"}).get("values", [])
    assert back[0][col] == "입수지연" and back[0][col - 1] == "인과(가용 시각) 규칙", back[0]
    assert back[0][col + 1] == head[col], (back[0][col + 1], head[col])
    for old, new in zip(vals[1:], back[1:]):
        o = old + [""] * (len(head) - len(old)); n = new + [""] * (len(head) + 1 - len(new))
        assert n[:col] == o[:col] and n[col + 1:len(head) + 1] == o[col:], (o[0], "shifted cells differ")
        if o[0].startswith("SW-"):
            assert n[col] == VALUES[o[0]], o[0]
    print(json.dumps({"inserted": letter, "rows": len(sw), "header": back[0], "verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main("--dry-run" in sys.argv)

"""Mirror verified v2 state into the existing dataset sheet and harness table."""
import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
import sys

from paths import ROOT, STORE, DATASETS
SHEET='1KWgNQPlajHWw7FLUMz3lr4rO2z7sxuxXGstD0-YjAw4'
TAB='데이터셋 manifest'
HEADERS=['v2 경로','v2 관측 최신 시각','v2 as_of (UTC)','v2 검증 상태','v2 manifest']

def api(resource,method,params=None,body=None):
    os.environ.setdefault('GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE',str(Path.home()/'.config/research-sheet-gws/credentials.json'))
    os.environ.setdefault('GOOGLE_WORKSPACE_CLI_CONFIG_DIR',str(Path.home()/'.config/research-sheet-gws'))
    os.environ.setdefault('GOOGLE_WORKSPACE_CLI_KEYRING_BACKEND','file')
    sys.path.insert(0,str(Path.home()/'.local/share/research-sheet'))
    from research_sheet import gws
    return gws(resource,method,params,body)

def mapping():
    result={}
    for name,ids in DATASETS.items():
        for id in ids:result.setdefault(id,[]).append(STORE/name)
    for subdir in ('derived','suvi-derived'):
        path=ROOT/subdir/'registry.json'
        if not path.exists():continue
        records=json.loads(path.read_text())
        if isinstance(records,dict):records=records.get('jobs',[])
        for record in records:
            for id in record.get('ids',[]):
                id=re.sub(r'^SW(\d{3})$',r'SW-\1',id)
                p=Path(record['output'])
                if p not in result.setdefault(id,[]):result[id].append(p)
    result.setdefault('SW-015',[]).append(STORE/'ace-release')
    return result

def observation(doc,folder,id):
    for key in ('observation_max','origin_max','latest_origin','latest_origin_utc','latest_obs_utc','observation_date_max','latest_init'):
        if doc.get(key):return str(doc[key])
    index=folder/'FILES.json'
    if index.exists():
        entries=json.loads(index.read_text())
        wanted={'SW-004':'ace.parquet','SW-005':'ch-hourly.parquet','SW-006':'suvi-hourly-v2.parquet'}.get(id)
        edges=[e['coverage']['end_utc'] for e in entries if e.get('coverage') and (wanted is None or Path(e['file']).name==wanted)]
        if edges:return max(edges)
    cov=doc.get('coverage',{})
    if isinstance(cov,dict):
        for key in ('end_utc','origin_max','max_utc','end'):
            if cov.get(key):return str(cov[key])
    return '관측 끝 미검증'

def fields(id,folders):
    paths=[];edges=[];times=[];statuses=[];manifests=[]
    for folder in folders:
        paths.append(str(folder))
        path=folder/'MANIFEST.json';manifests.append(str(path))
        if not path.exists():
            statuses.append('미완료 / manifest 없음');continue
        doc=json.loads(path.read_text())
        edges.append(observation(doc,folder,id))
        times.append(str(doc.get('as_of_utc',doc.get('created_utc',doc.get('generated_at_utc','as_of 미기록')))))
        state=str(doc.get('status','생성됨; 관측 끝/실험 사용 별도 검증'))
        if doc.get('file_index_sha256'):state+='; SHA256 index verified'
        statuses.append(folder.name+': '+state)
    return ['\n'.join(paths),'\n'.join(edges),'\n'.join(times),'\n'.join(statuses),'\n'.join(manifests)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args()
    meta=api([],'get',{'spreadsheetId':SHEET,'fields':'spreadsheetId,spreadsheetUrl,sheets.properties'})
    tab=next(s['properties'] for s in meta['sheets'] if s['properties']['title']==TAB)
    values=api(['values'],'get',{'spreadsheetId':SHEET,'range':f"'{TAB}'!A1:Y100",'valueRenderOption':'FORMULA'}).get('values',[])
    if not values or values[0][:2]!=['ID','데이터셋']:raise ValueError('unexpected dataset header')
    # Locate columns by header name (2026-10-15: '입수지연' was inserted after the causal-rule column).
    head=values[0];BASE=head.index('확인일')+1 if '확인일' in head else len(head)
    # 2026-10-15: the user trimmed the sheet (no v2 columns / 확인일). Then only the harness mirror is refreshed.
    SHEET_V2=HEADERS[0] in head and '확인일' in head
    V2=head.index(HEADERS[0]) if HEADERS[0] in head else BASE;V2E=V2+len(HEADERS)
    LAT=head.index('입수지연') if '입수지연' in head else None
    col=lambda i:chr(65+i)
    if any(c for row in values for c in row[V2:V2E]) and values[0][V2:V2E]!=HEADERS:
        raise ValueError('destination columns already contain unrelated data')
    original_length=len(values)
    known={row[0] for row in values[1:] if row}
    additions=[]
    # 2026-10-15: rows are never re-added; the user decides which datasets the sheet lists.
    maps=mapping();updates=[];report=[]
    for idx,row in enumerate(values[1:],1):
        if not row:continue
        id=row[0];folders=maps.get(id,[])
        f=fields(id,folders) if folders else ['','','','기존 원천 / 증거 / 갱신 대상 아님 (이력 보존)','']
        report.append({'id':id,'name':row[1],'fields':f})
        updates.append({'updateCells':{'range':{'sheetId':tab['sheetId'],'startRowIndex':idx,'endRowIndex':idx+1,'startColumnIndex':V2,'endColumnIndex':V2E},
            'rows':[{'values':[{'userEnteredValue':{'stringValue':v}} for v in f]}],'fields':'userEnteredValue'}})
    header={'updateCells':{'range':{'sheetId':tab['sheetId'],'startRowIndex':0,'endRowIndex':1,'startColumnIndex':V2,'endColumnIndex':V2E},
                'rows':[{'values':[{'userEnteredValue':{'stringValue':v}} for v in HEADERS]}],'fields':'userEnteredValue'}}
    ROOT.joinpath('reports/catalog.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if args.write and not SHEET_V2:
        print('sheet has no v2 columns; skipping sheet write (harness mirror only)')
    if args.write and SHEET_V2:
        requests=[header,*updates,
            {'copyPaste':{'source':{'sheetId':tab['sheetId'],'startRowIndex':0,'endRowIndex':1,'startColumnIndex':BASE-1,'endColumnIndex':BASE},
                          'destination':{'sheetId':tab['sheetId'],'startRowIndex':0,'endRowIndex':1,'startColumnIndex':V2,'endColumnIndex':V2E},'pasteType':'PASTE_FORMAT'}},
            {'repeatCell':{'range':{'sheetId':tab['sheetId'],'startRowIndex':0,'endRowIndex':len(values),'startColumnIndex':V2,'endColumnIndex':V2E},
                           'cell':{'userEnteredFormat':{'wrapStrategy':'WRAP'}},'fields':'userEnteredFormat.wrapStrategy'}}]
        for start,end,width in [(V2,V2+1,420),(V2+1,V2+3,190),(V2+3,V2+4,280),(V2+4,V2+5,420)]:
            requests.append({'updateDimensionProperties':{'range':{'sheetId':tab['sheetId'],'dimension':'COLUMNS','startIndex':start,'endIndex':end},'properties':{'pixelSize':width},'fields':'pixelSize'}})
        if additions:
            requests.extend([
                {'copyPaste':{'source':{'sheetId':tab['sheetId'],'startRowIndex':original_length-1,'endRowIndex':original_length,'startColumnIndex':0,'endColumnIndex':BASE},
                              'destination':{'sheetId':tab['sheetId'],'startRowIndex':original_length,'endRowIndex':len(values),'startColumnIndex':0,'endColumnIndex':BASE},'pasteType':'PASTE_FORMAT'}},
                {'updateCells':{'range':{'sheetId':tab['sheetId'],'startRowIndex':original_length,'endRowIndex':len(values),'startColumnIndex':0,'endColumnIndex':BASE},
                    'rows':[{'values':[{'userEnteredValue':{'stringValue':v}} for v in row]} for row in additions],'fields':'userEnteredValue'}},
            ])
        api([],'batchUpdate',{'spreadsheetId':SHEET},{'requests':requests})
        check=api(['values'],'get',{'spreadsheetId':SHEET,'range':f"'{TAB}'!{col(V2)}1:{col(V2E-1)}{len(values)}",'valueRenderOption':'FORMULA'}).get('values',[])
        expected=[HEADERS,*[r['fields'] for r in report]]
        check=[row+['']*(5-len(row)) for row in check]
        if check!=expected:raise RuntimeError('sheet readback differs from authored v2 columns')
        ROOT.joinpath('reports/sheet-verified.json').write_text(json.dumps({'url':meta['spreadsheetUrl'],'rows':len(report),'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'columns':f'{col(V2)}:{col(V2E-1)}'},indent=2)+'\n')
    h=ROOT.parent/'research/chatgpt-skill/solar-queue-harness/references/datasets.md'
    text=h.read_text();marker='<!-- DATA-V2-TABLE:START -->'
    if marker in text:text=text.split(marker)[0].rstrip()+'\n'
    lines=['','<!-- DATA-V2-TABLE:START -->','## v2 최신화 상태','',
        '위 표는 v1 재현 이력이다. 아래 상태는 SolarWind DB의 v2 열(`v2 경로`부터 5열)과 함께 갱신한다. `ready` 전환 조건은 서버의 `data-v2/reports/defaults.json`이며, 수집·해시 검사와 실험 사용 검증은 별개다. 실제 관측 끝이 미검증이거나 일부 생성/전송이 남은 항목을 최신화 완료로 취급하지 않는다.','',
        '| ID | 데이터셋 | v2 경로 | 관측 최신 | 검증 상태 |','|---|---|---|---|---|']
    for r in report:
        f=r['fields'];escape=lambda s:s.replace('|','\\|').replace('\n','<br>')
        lines.append('| '+ ' | '.join(escape(str(s)) for s in [r['id'],r['name'],f[0],f[1],f[3]])+' |')
    lines+=['','<!-- DATA-V2-TABLE:END -->','']
    h.write_text(text+'\n'.join(lines))
    print(json.dumps({'write':args.write,'rows':len(report),'sheet_verified':args.write,'catalog':str(ROOT/'reports/catalog.json')}))

if __name__=='__main__':main()

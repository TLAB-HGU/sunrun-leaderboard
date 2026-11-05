"""Publish the user's local CPU scheduling policy for this rerun queue."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'leaderboard'))
import lbstatus

names=set(json.loads((ROOT/'registry.json').read_text()))
records={p.parent.name:json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/launch.json')}
lbstatus.leaderboard_submissions=lambda entries,dataset:{}

def update(entries):
    for entry in entries:
        name=entry.get('experiment')
        if entry.get('member')!='seongeun' or name not in names:
            continue
        note=entry.get('note','')
        for host in ('golab2 CPU','local CPU'):
            note=note.replace(host,'현재 로컬 CPU')
        if 'golab2 미사용' not in note:
            note+='; 앞으로 현재 로컬 CPU에서 실행, golab2 미사용'
        if records.get(name,{}).get('migration'):
            entry['status']='대기'
            note='golab2 체크포인트 회수 완료; 현재 로컬 CPU 자원 대기; 입력 피처·학습 y ffill'
        entry['note']=note
    return entries

lbstatus.commit(update,message='Run all remaining fresh-df experiments on local CPU only')
print('Local CPU policy published',flush=True)

import migrate_local


def test_rebase_paths_and_proven_input_identity_without_changing_model_settings():
    old={'path':'/remote/input.parquet','bytes':42,'mtime_ns':100}
    new={'path':'/local/input.parquet','bytes':42,'mtime_ns':126}
    source={'data':old,'models':['/remote/runs/x.ubj'],'params':{'nthread':2,'seed':42},
            'other':'/remoteish/unrelated','note':'original /remote data'}
    actual=migrate_local.rebase(source,'/remote','/local',old,new)
    assert actual['data']==new
    assert actual['models']==['/local/runs/x.ubj']
    assert actual['params']==source['params']
    assert actual['other']==source['other'] and actual['note']==source['note']
    assert source['data']==old


def test_identity_with_different_bytes_is_not_attested():
    old={'path':'/remote/input.parquet','bytes':42,'mtime_ns':100}
    wrong=dict(old,bytes=99)
    new={'path':'/local/input.parquet','bytes':42,'mtime_ns':126}
    actual=migrate_local.rebase(wrong,'/remote','/local',old,new)
    assert actual['bytes']==99 and actual['mtime_ns']==100

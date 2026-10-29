"""Data-root resolution shared by the reproduction pipelines.

All inputs live under one directory, ``$SUNRUN_REPRO_DATA`` (default ``reproduce/reproduce_data``),
which ``fetch_data.py`` fills from the Hugging Face dataset. Layout::

    store/ch-v1/ace.parquet
    store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet
    grids/{train_complete_causal,official_causal,features_f1}.parquet, grids/features_v3.json
    reference/pointweight4_cache/{train,f1,official}.npz, names.json, speed.parquet   (verification only)
    lb-store/{truth,folds,naive}.parquet
    e1604/...
"""
import hashlib
import os
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('SUNRUN_REPRO_DATA', REPRO / 'reproduce_data')).resolve()
OUTPUT_ROOT = Path(os.environ.get('SUNRUN_REPRO_OUT', REPRO / 'outputs')).resolve()


def data(*parts):
    path = DATA_ROOT.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run `python reproduce/fetch_data.py` or set SUNRUN_REPRO_DATA')
    return path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

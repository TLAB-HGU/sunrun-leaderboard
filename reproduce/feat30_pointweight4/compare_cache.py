"""Fail unless a rebuilt feature cache equals the frozen reference cache array by array.

    python reproduce/feat30_pointweight4/compare_cache.py <rebuilt cache dir> <reference cache dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(new, ref):
    new, ref = Path(new), Path(ref)
    for name in ('train', 'f1', 'official'):
        a, b = np.load(new / f'{name}.npz'), np.load(ref / f'{name}.npz')
        assert sorted(a.files) == sorted(b.files), name
        for key in b.files:
            assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == 'f'), f'{name}/{key} differs'
    assert json.loads((new / 'names.json').read_text()) == json.loads((ref / 'names.json').read_text())
    assert pd.read_parquet(new / 'speed.parquet').equals(pd.read_parquet(ref / 'speed.parquet'))
    return True


if __name__ == '__main__':
    compare(*sys.argv[1:3])
    print('rebuilt cache is identical to the frozen reference')

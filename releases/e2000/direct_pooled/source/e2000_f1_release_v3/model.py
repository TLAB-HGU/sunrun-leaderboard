"""Memory-bounded pooled input expansion; identical float32 values and XGB fit."""
import mmap
from pathlib import Path
import tempfile
import numpy as np
from experiments.e2000_njobs_all.model import PooledRegressor as OriginalPooledRegressor


class DiskBackedPooledRegressor(OriginalPooledRegressor):
    def _expand(self, X, Y, mask):
        rows, horizons = np.nonzero(mask)
        path = self._expansion_root / f'{len(self._expansion_maps)}.npy'
        flat = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                        shape=(len(rows), X.shape[1] + 1))
        self._expansion_maps.append(flat)
        for start in range(0, len(rows), 4096):
            end = min(start + 4096, len(rows))
            flat[start:end, :-1] = X[rows[start:end]]
            flat[start:end, -1] = horizons[start:end] + 1
        flat.flush()
        # Written pages are backed by disk and can be reloaded on demand.
        if hasattr(flat._mmap, 'madvise'):
            flat._mmap.madvise(mmap.MADV_DONTNEED)
        return flat, Y[rows, horizons]

    def fit(self, X, Y, *, observed_mask=None, validation=None, output_dir, context,
            validation_role='held_out'):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pooled-expansion-', dir=directory) as temporary:
            self._expansion_root = Path(temporary)
            self._expansion_maps = []
            try:
                return super().fit(X, Y, observed_mask=observed_mask, validation=validation,
                                   output_dir=output_dir, context=context, validation_role=validation_role)
            finally:
                for array in self._expansion_maps:
                    array._mmap.close()
                del self._expansion_maps
                del self._expansion_root

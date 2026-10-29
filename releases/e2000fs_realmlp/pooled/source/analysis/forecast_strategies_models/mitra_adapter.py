"""Fail closed until the official Mitra-v2 fit has reusable, verified artifacts.

Mitra-v2 DOES support regression. The blocked capability is this campaign's
fit-once/predict-many contract. The pinned public API retrains in every predict,
then disposes of its bagged wrapper. Treating that API as an ordinary regressor
would silently repeat fine-tuning for every recursive horizon/query batch.

This module deliberately imports no optional ML runtime and changes no packages.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

MITRA_CODE_REVISION = 'b4701e8148dc33b00ed15d7086ff59816957cde4'
MITRA_WEIGHT_REVISION = '246c7d310fff04296a2388df5f448e3ffe809283'
CODE_BASE = f'https://huggingface.co/autogluon/mitra-finetune/blob/{MITRA_CODE_REVISION}'

SOURCE_EVIDENCE = {
    'api.py': {
        'url': f'{CODE_BASE}/src/mitra_finetune/api.py',
        'sha256': 'c92c4fb0029329383a336168d3ce2c8be45186eed3f650dc59100d7cd2d3128c',
        'facts': [
            'Line 212: fit retains X/y in _fit_data; it does not fine-tune.',
            'Line 392: regression predict calls _execute_view, which calls run_view.',
        ],
    },
    'runner.py': {
        'url': f'{CODE_BASE}/src/mitra_finetune/runner.py',
        'sha256': 'fbf912da747348a549ad33a2f33fd75d5d0a1982eccc863bc616345d1d0d761e',
        'facts': [
            'Line 771: upstream notes that reconstructing the wrapper protocol was not equivalent.',
            'Line 1076: child_main calls wrapper.cleanup() before returning.',
            'Line 1155: run_view executes in a TemporaryDirectory and returns prediction arrays.',
        ],
    },
    'hooks.py': {
        'url': f'{CODE_BASE}/src/mitra_finetune/hooks.py',
        'sha256': 'c957740ee9c758f981cc5796ad454b77715a8a4a14ca296e0fc9cb51c4ed2653',
        'facts': ['Line 10: official hooks are process-global and each fit runs in its own process.'],
    },
}

_STATUS = {
    'model': 'mitra_v2',
    'status': 'unsupported_persistent_inference',
    'regression_supported': True,
    'scientific_result': False,
    'code_revision': MITRA_CODE_REVISION,
    'weight_repository': 'autogluon/mitra-regressor-2',
    'weight_revision': MITRA_WEIGHT_REVISION,
    'requested_recipe': {
        'package': 'mitra-finetune==0.3.0', 'fine_tune_steps': 50,
        'num_bag_folds': 8, 'time_limit_seconds': 3600,
        'random_state': 42, 'physical_gpu': 1,
        'point_prediction': 'mean of the official 1000-bin distribution',
    },
    'reason': (
        'The pinned official API fine-tunes on each predict call and destroys its '
        'bagged wrapper afterward. No supported retained-model inference interface '
        'has been verified. Repeated prediction would violate the approved '
        'fit-once-per-stage/head protocol and contaminate inference timing.'
    ),
    'required_recovery': [
        'Provide a pinned retained-artifact path preserving the official wrapper, '
        'preprocessing, eight bag children, support composition and bin-mean decode.',
        'Prove batched and repeated predictions match an official single-call '
        'reference on GPU1, and prove predict invokes no training.',
        'Preserve native train/validation fine-tuning history and ensure F1/F2 '
        'selection uses only training-side data, never overlapping validation labels.',
        'After preceding campaigns complete, resolve and lock the shared environment; '
        'verify the resulting CUDA build with driver 561.17 before changing packages.',
    ],
    'dependency_constraints': {
        'python': '>=3.11; campaign uses 3.12',
        'mitra_finetune': '0.3.0 at the pinned official source revision',
        'autogluon_tabular': '1.6.3[mitra] metadata inspected; not an installed lock',
        'torch': '>=2.10,<2.14 (Linux AutoGluon Mitra extra)',
        'huggingface_hub': '>=1.3,<2.0 (AutoGluon Mitra extra)',
        'tabarena': '>=0.1.0; exact version requires protocol verification',
        'cuda_build': 'unresolved; must be tested against host driver 561.17',
        'active_environment_change_allowed': False,
    },
    'source_evidence': SOURCE_EVIDENCE,
}


class UnsupportedRecipe(RuntimeError):
    """A required model execution contract has no verified implementation."""


def recipe_status():
    """Return a detached receipt; callers cannot mutate the canonical evidence."""
    return copy.deepcopy(_STATUS)


def _write_receipt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class MitraAdapter:
    """Explicit unavailable adapter with the campaign's scalar model interface."""

    def __init__(self, config=None, stage='select', selection_dir=None):
        self.config = dict(config or {})
        self.stage = stage
        self.selection_dir = selection_dir
        self.metadata = recipe_status()
        self.metadata['stage'] = stage

    def fit(self, X, y, Xv, yv, directory):
        # Never serialize training arrays into a failure receipt.
        receipt = recipe_status()
        receipt.update(stage=self.stage, operation='fit', fit_started=False)
        _write_receipt(Path(directory) / 'unsupported_recipe.json', receipt)
        raise UnsupportedRecipe(receipt['reason'])

    def predict(self, X):
        raise UnsupportedRecipe(_STATUS['reason'])

    def clear(self):
        """No runtime/model was acquired; cleanup is intentionally idempotent."""


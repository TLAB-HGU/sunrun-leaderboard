"""Maintainer tool: write data_manifest.json from a populated data root.

``--root`` holds the complete layout. Files that already live in this repository (``--repo-file``) or come from the
pinned lb-store (the three evaluation tables) are recorded with their source; everything else is expected on the HF
dataset under the same relative path (upload the ``hf`` entries with `hf upload`; see README).

    python reproduce/build_manifest.py --root <full data root> --hf-revision <sha>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402

REPO_FILES = {'e1604/prepared_inputs.npz': 'releases/e1604/prepared_inputs.npz',
              'reference/pointweight4_forecast.npz': 'releases/e2000fs_feat30_pointweight4/forecast.npz',
              'reference/extreme4_forecast.npz': 'releases/e2000fs_feat30_lgbm_extreme4/forecast.npz',
              'reference/dirmo_24_feat30_forecast.npz': 'releases/e2000fs/dirmo_24_feat30/forecast.npz',
              'reference/dirrecmo_24_forecast.npz': 'releases/e2000fs/dirrecmo_24/forecast.npz',
              'e1604/hsspeak_f1_inputs.npz': 'releases/f1-20261026/hsspeak_f1_inputs.npz'}
LB_FILES = {'lb-store/truth.parquet', 'lb-store/folds.parquet', 'lb-store/naive.parquet'}
REFERENCE_PREFIXES = ('reference/',)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--hf-repo', default='tlabtlab/sunrun-repro-data')
    parser.add_argument('--hf-revision', default='main')
    parser.add_argument('--lb-repo', default='tlabtlab/sunrun-lb-store')
    parser.add_argument('--lb-revision', default='5c274be8a920ce98832d14f7e56f66f341bc0833')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'data_manifest.json'))
    args = parser.parse_args()
    root = Path(args.root)
    files = []
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        rel = path.relative_to(root).as_posix()
        entry = {'path': rel, 'sha256': paths.sha256(path), 'size': path.stat().st_size,
                 'group': 'reference' if rel.startswith(REFERENCE_PREFIXES) else 'core'}
        if rel in REPO_FILES:
            entry.update(source='repo', repo_path=REPO_FILES[rel])
        elif rel in LB_FILES:
            entry.update(source='lb')
        else:
            entry.update(source='hf')
        files.append(entry)
    manifest = {'hf_repo': args.hf_repo, 'hf_revision': args.hf_revision, 'lb_repo': args.lb_repo,
                'lb_revision': args.lb_revision, 'files': files}
    Path(args.out).write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{len(files)} files, {sum(f["size"] for f in files) / 1e6:.1f} MB -> {args.out}')


if __name__ == '__main__':
    main()

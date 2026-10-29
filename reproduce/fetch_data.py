"""Download and verify the reproduction inputs into ``$SUNRUN_REPRO_DATA`` (default reproduce/reproduce_data).

Sources listed in ``data_manifest.json``:
    repo  - already in this git repository (releases/...), copied
    hf    - Hugging Face dataset ``tlabtlab/sunrun-repro-data`` (private; needs read access, HF_TOKEN or `hf auth login`)
    lb    - ``tlabtlab/sunrun-lb-store`` truth/folds/naive at the pinned revision (private, same access rule)

    python reproduce/fetch_data.py                 # core inputs for both experiments
    python reproduce/fetch_data.py --reference     # also the frozen outputs used to verify the rerun
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import paths  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / 'data_manifest.json'
REPO = Path(__file__).resolve().parents[1]


def fetch(entry, manifest, target):
    from huggingface_hub import hf_hub_download
    source = entry['source']
    if source == 'repo':
        shutil.copyfile(REPO / entry['repo_path'], target)
    elif source == 'hf':
        cached = hf_hub_download(manifest['hf_repo'], entry['path'], repo_type='dataset', revision=manifest['hf_revision'])
        shutil.copyfile(cached, target)
    elif source == 'lb':
        cached = hf_hub_download(manifest['lb_repo'], Path(entry['path']).name, repo_type='dataset',
                                 revision=manifest['lb_revision'])
        shutil.copyfile(cached, target)
    else:
        raise ValueError(f'unknown source {source!r} for {entry["path"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', action='store_true', help='also fetch frozen reference outputs')
    parser.add_argument('--root', default=str(paths.DATA_ROOT))
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    root = Path(args.root)
    wanted = [e for e in manifest['files'] if args.reference or e['group'] == 'core']
    for entry in wanted:
        target = root / entry['path']
        if target.exists() and paths.sha256(target) == entry['sha256']:
            print('ok      ', entry['path'])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        fetch(entry, manifest, target)
        if paths.sha256(target) != entry['sha256']:
            target.unlink()
            raise SystemExit(f'sha256 mismatch after download: {entry["path"]}')
        print('fetched ', entry['path'])
    print(f'{len(wanted)} files verified under {root}')


if __name__ == '__main__':
    main()

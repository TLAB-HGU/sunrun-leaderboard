"""Render and publish the stored, equal-weight three-seed forecast visualization."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
EXPERIMENT = 'e2000fs_calib_offset_roll_drop_t035_equatorial_mean3'
OUT = ROOT / 'docs/visualizations' / EXPERIMENT
WORK = CAMPAIGN / 'lb_publish_drop035/ensemble'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submitted-utc', required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / 'docs/visualizations/code/submission_viz.py'),
                    '--member', 'seongeun', '--experiment', EXPERIMENT,
                    '--submitted-utc', args.submitted_utc, '--out', str(OUT)], check=True)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['verification']['complete']
    assert manifest['verification']['all_equal_to_record']
    assert manifest['verification']['rendered_frames'] == 1052
    assert len(manifest['outputs']) == 9
    sys.path.insert(0, str(ROOT / 'leaderboard'))
    spec = importlib.util.spec_from_file_location(
        'ablation_source_publisher', ROOT / 'research/offset_roll_ablation2_20261105/publish_lb.py')
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    files = {str(p.relative_to(ROOT)): p.read_bytes() for p in OUT.iterdir() if p.is_file()}
    revision = publisher.publish_sources(files)
    (WORK / 'visualization-source-revision.txt').write_text(revision + '\n')
    print(json.dumps({'visualization_revision': revision, 'files': len(files)}), flush=True)
    subprocess.run([sys.executable, str(ROOT / 'docs/visualizations/code/publish_links.py'),
                    '--commit', revision,
                    f'seongeun/{EXPERIMENT}/{args.submitted_utc}={EXPERIMENT}'], check=True)
    verify(args.submitted_utc, revision, len(files))


def verify(submitted_utc, revision, file_count):
    import submit
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    head = api.repo_info(submit.SPACE, repo_type='space').sha
    mapping = submit.space_file(submit.SPACE, 'visualizations.json', head)
    url = f'https://github.com/TLAB-HGU/sunrun-leaderboard/tree/{revision}/docs/visualizations/{EXPERIMENT}'
    assert any(e['experiment'] == EXPERIMENT and e['submitted_utc'] == submitted_utc
               and e['url'] == url for e in mapping['models'])
    page = Path(hf_hub_download(submit.SPACE, 'index.html', repo_type='space', revision=head)).read_text()
    rows = re.findall(r'<tr\b[^>]*>.*?</tr>', page, re.S)
    matching = [row for row in rows if EXPERIMENT in row]
    assert len(matching) == 1 and url in matching[0] and 'visualization-link' in matching[0]
    raw = f'https://raw.githubusercontent.com/TLAB-HGU/sunrun-leaderboard/{revision}/docs/visualizations/{EXPERIMENT}/README.md'
    with urllib.request.urlopen(raw, timeout=60) as response:
        assert response.status == 200 and EXPERIMENT in response.read().decode()
    evidence = {'experiment': EXPERIMENT, 'submitted_utc': submitted_utc,
                'url': url, 'space_revision': head, 'files': file_count,
                'monthly_gifs': 9, 'rendered_frames': 1052, 'official_score_match': True,
                'space_visualization_link_verified': True, 'github_readme_http_status': 200}
    (WORK / 'visualization-verified.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()

"""Resume full official-submission visualization and link it on the leaderboard."""
import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from publish import EXPERIMENT, MEMBER, REPO, WORK, source_publisher, write

OUT = REPO / 'docs/visualizations' / EXPERIMENT


def valid_manifest(submitted_utc):
    path = OUT / 'manifest.json'
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    check = manifest.get('verification', {})
    return (manifest.get('experiment') == EXPERIMENT
            and manifest.get('submitted_utc') == submitted_utc
            and check.get('complete') and check.get('all_equal_to_record')
            and check.get('rendered_frames') == 1052
            and len(manifest.get('outputs', [])) == 9
            and all((OUT / item['file']).is_file()
                    and (OUT / item['file']).stat().st_size == item['bytes']
                    for item in manifest['outputs'])
            and all((OUT / name).is_file() for name in ('README.md', 'leaderboard-result.json', 'lead_curves.png')))


def verify(submitted_utc, revision):
    import submit
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    head = api.repo_info(submit.SPACE, repo_type='space').sha
    mapping = submit.space_file(submit.SPACE, 'visualizations.json', head)
    url = f'https://github.com/TLAB-HGU/sunrun-leaderboard/tree/{revision}/docs/visualizations/{EXPERIMENT}'
    assert any(e['experiment'] == EXPERIMENT and e['submitted_utc'] == submitted_utc
               and e['url'] == url for e in mapping.get('models', []))
    page = Path(hf_hub_download(submit.SPACE, 'index.html', repo_type='space', revision=head)).read_text()
    rows = re.findall(r'<tr\b[^>]*>.*?</tr>', page, re.DOTALL)
    matching = [row for row in rows if EXPERIMENT in row]
    assert len(matching) == 1 and url in matching[0] and 'visualization-link' in matching[0]
    raw = f'https://raw.githubusercontent.com/TLAB-HGU/sunrun-leaderboard/{revision}/docs/visualizations/{EXPERIMENT}/README.md'
    with urllib.request.urlopen(raw, timeout=60) as response:
        assert response.status == 200 and EXPERIMENT in response.read().decode()
    evidence = {'experiment': EXPERIMENT, 'submitted_utc': submitted_utc, 'url': url, 'space_revision': head,
                    'monthly_gifs': 9, 'rendered_frames': 1052, 'official_score_match': True,
                    'space_visualization_link_verified': True, 'github_readme_http_status': 200}
    write(WORK / 'visualization-verified.json', evidence)
    print(json.dumps(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submitted-utc', required=True)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if not valid_manifest(args.submitted_utc):
        subprocess.run([sys.executable, str(REPO / 'docs/visualizations/code/submission_viz.py'),
                        '--member', MEMBER, '--experiment', EXPERIMENT,
                        '--submitted-utc', args.submitted_utc, '--out', str(OUT)], check=True)
    if not valid_manifest(args.submitted_utc):
        raise RuntimeError('Visualization manifest or one of its required artifacts is incomplete')
    sys.path.insert(0, str(REPO / 'leaderboard'))
    revision_file = WORK / 'visualization-source-revision.txt'
    if revision_file.is_file():
        revision = revision_file.read_text().strip()
    else:
        files = {str(p.relative_to(REPO)): p.read_bytes() for p in OUT.iterdir() if p.is_file()}
        revision = source_publisher()(files)
        revision_file.write_text(revision + '\n')
    subprocess.run([sys.executable, str(REPO / 'docs/visualizations/code/publish_links.py'),
                    '--commit', revision,
                    f'{MEMBER}/{EXPERIMENT}/{args.submitted_utc}={EXPERIMENT}'], check=True)
    verify(args.submitted_utc, revision)


if __name__ == '__main__':
    main()

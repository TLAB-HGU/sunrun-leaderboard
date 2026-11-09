"""Keep the actual remote CPU scheduler foregrounded on SSH/WSL."""
import argparse
from pathlib import Path
from remote_services import foreground_scheduler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    raise SystemExit(foreground_scheduler(args.root.resolve()))

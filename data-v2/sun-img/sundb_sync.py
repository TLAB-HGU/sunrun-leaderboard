#!/usr/bin/env python3
"""Retired external publication entrypoint. Raw data is managed by the internal RawDB service."""


def _retired(*args, **kwargs):
    raise RuntimeError("Google Drive publication is retired; use swcollect rawdb export for internal snapshots")


sync_ace = _retired
sync_suvi = _retired
sync_images = _retired
sync_live = _retired
rclone_copy = _retired
push_series = _retired


def main():
    raise SystemExit("Google Drive publication is retired; raw data stays internal")


if __name__ == '__main__':
    main()

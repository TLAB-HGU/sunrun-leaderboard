"""Sunrun-visible entrypoint for the no-fit output-sharing diagnostic service."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'analysis/w50_audit_20261108/output_sharing_probe_admission_controller.py'), run_name='__main__')

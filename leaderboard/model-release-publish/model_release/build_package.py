"""Assemble the standalone inference ZIP; fitted PKLs are distributed separately."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

RUNTIME_FILES = (
    "__init__.py", "__main__.py", "api.py", "features.py", "numeric.py", "strips.py",
    "regular_runtime.py", "analog_runtime.py", "test_release.py",
)


def build_package(output_directory):
    output = Path(output_directory)
    source = Path(__file__).resolve().parent
    for name in RUNTIME_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    artifact = output / "sunrun_inference.zip"
    with ZipFile(artifact, "w", ZIP_DEFLATED) as archive:
        for name in RUNTIME_FILES:
            archive.write(source / name, "model_release/" + name)
        for name in ("README.md", "requirements.txt"):
            archive.write(source / name, name)
        for name in ("suvi_availability.parquet", "identity-audit.json"):
            archive.write(output / name, name)
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    print(build_package(args.output_directory))

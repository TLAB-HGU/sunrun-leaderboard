"""Publish only verified release artifacts; preserve unrelated Drive/Git files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

FOLDER = "1-VWoE6LeutGnC_31Kzc0ni5Q-x-4TWCm"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def verify_gate(root, models, verification):
    if len(models) != 10 or not verification.get("passed"):
        raise ValueError("The complete ten-model integration gate must pass before publication")
    expected = {model["experiment"]: model["config_sha256"] for model in models}
    verified = verification.get("models", [])
    if len(expected) != 10 or len(verified) != 10 or len({row["experiment"] for row in verified}) != 10:
        raise ValueError("The verification receipt must cover exactly ten unique models")
    if {row["experiment"]: row["config_sha256"] for row in verified} != expected:
        raise ValueError("Verification model/config coverage differs from selected submissions")
    for row in verified:
        path = root / (row["experiment"] + "_seongeun.pkl")
        if not row.get("passed") or row.get("rows") != 2833 * 72 or sha256(path) != row["sha256"]:
            raise ValueError(f"Current model bytes have not passed integration: {path.name}")
    if sha256(root / "sunrun_inference.zip") != verification.get("package_sha256"):
        raise ValueError("Current inference package bytes have not passed integration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--publish-worktree", type=Path, required=True)
    args = parser.parse_args()
    root = args.release_dir.resolve()
    models = json.loads((root / "selection.json").read_text())
    verification = json.loads((root / "integration-verification.json").read_text())
    verify_gate(root, models, verification)
    files = [root / f"{model['experiment']}_seongeun.pkl" for model in models]
    files += [root / "sunrun_inference.zip"]
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing artifact: {path}")
    remote = "sundb:"
    options = ("--drive-root-folder-id", FOLDER, "--timeout", "30s", "--contimeout", "10s")
    existing = json.loads(command("rclone", "lsjson", remote, *options, "--max-depth", "1", "--hash"))
    by_name = {item["Name"]: item for item in existing}
    receipts = []
    for path in files:
        old = by_name.get(path.name)
        if old:
            with path.open("rb") as stream:
                md5 = hashlib.file_digest(stream, "md5").hexdigest()
            if old.get("Hashes", {}).get("md5") != md5:
                raise ValueError(f"Existing Drive artifact has different bytes: {path.name}")
        else:
            subprocess.run(["rclone", "copyto", str(path), remote + path.name, *options,
                            "--checksum", "--tpslimit", "3"], check=True)
        link = command("rclone", "link", remote + path.name, *options).splitlines()[-1]
        if not link.startswith("https://drive.google.com/"):
            raise ValueError(f"Unexpected public link for {path.name}: {link}")
        receipts.append({"file_name": path.name, "size": path.stat().st_size,
                         "sha256": sha256(path), "url": link})
        (root / "upload-receipts.json").write_text(json.dumps(receipts, indent=2))
        print(f"Uploaded and linked {path.name}", flush=True)
    links = {row["file_name"]: row for row in receipts}
    package = links["sunrun_inference.zip"]["url"]
    for model in models:
        relative = Path("scripts/seongeun") / model["experiment"] / (model["config_sha256"] + ".py")
        script = args.publish_worktree / relative
        text = script.read_text()
        marker = "# MODEL RELEASE (seongeun, 2026-10-10)"
        if marker in text:
            raise ValueError(f"Release header already exists; inspect instead of duplicating: {relative}")
        receipt = links[f"{model['experiment']}_seongeun.pkl"]
        header = (f"{marker}\n# Fitted model: {receipt['file_name']}\n"
                  f"# Google Drive: {receipt['url']}\n"
                  f"# SHA-256: {receipt['sha256']}\n"
                  f"# Standalone inference package: {package}\n"
                  "# Access: anyone with the link. See package README for numeric inputs and usage.\n\n")
        if text.startswith("#!"):
            first, rest = text.split("\n", 1)
            text = first + "\n" + header + rest
        else:
            text = header + text
        script.write_text(text)
    manifest = {"folder_url": f"https://drive.google.com/drive/folders/{FOLDER}",
                "models": [{"experiment": m["experiment"], "member": "seongeun",
                            "config_sha256": m["config_sha256"], "mse": m["mse"],
                            **links[f"{m['experiment']}_seongeun.pkl"]} for m in models],
                "inference_package": links["sunrun_inference.zip"], "verification": verification}
    (root / "release-manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Drive files uploaded; script headers prepared. Verify readback before Git publication.")


if __name__ == "__main__":
    main()

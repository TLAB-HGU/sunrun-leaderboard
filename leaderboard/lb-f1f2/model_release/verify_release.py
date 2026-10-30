"""Verify every released PKL through the isolated public inference interface."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--experiment", help="Verify one available model without opening the publication gate")
    parser.add_argument("--available", action="store_true", help="Verify all currently completed model files")
    parser.add_argument("--resume", action="store_true", help="Reuse evidence only for identical package, inputs and PKL bytes")
    args = parser.parse_args()
    release, source = args.release_dir.resolve(), args.source_root.resolve()
    records = json.loads((release / "selection.json").read_text())
    if args.experiment:
        records = [record for record in records if record["experiment"] == args.experiment]
        if not records:
            raise ValueError("Requested experiment is not in the selected top ten")
    if args.available:
        records = [record for record in records if (release / (record["experiment"] + "_seongeun.pkl")).is_file()]
    donki = release / "validation-inputs" / "donki"
    donki.mkdir(parents=True, exist_ok=True)
    for directory in (source / "store/headroom-audit/donki", source / "store/donki-live"):
        for path in sorted(directory.glob("WSAEnlilSimulations_*.json")):
            if directory.name == "donki" and path.name >= "WSAEnlilSimulations_2026-06":
                continue
            link = donki / path.name
            if not link.exists():
                link.symlink_to(path)
    inputs = {"raw_ace_dir": str(source / "store/ch-breakthrough-v2/upload/raw-ace-snapshot"),
              "speed_parquet": str(source / "store/ch-v1/ace.parquet"),
              "ch_parquet": str(source / "store/ch-v1/ch-hourly.parquet"),
              "suvi_parquet": str(source / "store/suvi-fusion/suvi-hourly-v2.parquet"),
              "suvi_manifest": str(release / "suvi_availability.parquet"), "donki_dir": str(donki)}
    with tempfile.TemporaryDirectory(prefix="sunrun-isolated-inference-") as directory:
        isolated = Path(directory)
        with ZipFile(release / "sunrun_inference.zip") as archive:
            archive.extractall(isolated)
        with (release / "sunrun_inference.zip").open("rb") as stream:
            package_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        input_files = [Path(inputs[key]) for key in ("speed_parquet", "ch_parquet", "suvi_parquet", "suvi_manifest")]
        input_files += sorted(Path(inputs["raw_ace_dir"]).glob("year=*/part.parquet"))
        input_files += sorted(donki.glob("*.json"))
        fingerprint = hashlib.sha256()
        for path in input_files:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            fingerprint.update((str(path) + "\t" + digest + "\n").encode())
        input_sha = fingerprint.hexdigest()
        prior_reports = []
        previous = release / "partial-integration-verification.json"
        if args.resume and previous.is_file():
            prior = json.loads(previous.read_text())
            if prior.get("package_sha256") == package_sha and prior.get("input_sha256") == input_sha:
                for record in records:
                    matches = [row for row in prior.get("models", []) if row["experiment"] == record["experiment"]]
                    if len(matches) == 1 and matches[0].get("passed") and matches[0].get("config_sha256") == record["config_sha256"]:
                        with (release / (record["experiment"] + "_seongeun.pkl")).open("rb") as stream:
                            current = hashlib.file_digest(stream, "sha256").hexdigest()
                        if current == matches[0]["sha256"]:
                            prior_reports.append(matches[0])
        completed = {row["experiment"] for row in prior_reports}
        records = [record for record in records if record["experiment"] not in completed]
        payload = {"release": str(release), "records": records, "inputs": inputs,
                   "package_sha256": package_sha, "input_sha256": input_sha,
                   "prior_reports": prior_reports}
        (isolated / "verification-inputs.json").write_text(json.dumps(payload))
        program = '''import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from model_release import load_model

payload=json.loads(Path("verification-inputs.json").read_text())
root=Path(payload["release"])
reports=payload["prior_reports"]
for row in payload["records"]:
    name=row["experiment"]
    path=root/(name+"_seongeun.pkl")
    reference=pd.read_parquet(root/"references"/(name+".parquet"))
    reference["origin_last_input_utc"]=pd.to_datetime(reference["origin_last_input_utc"],utc=True)
    origins=pd.DatetimeIndex(reference["origin_last_input_utc"].unique()).sort_values()
    model=load_model(path)
    if model.bundle["config_sha256"]!=row["config_sha256"]:
        raise ValueError("Config identity mismatch: "+name)
    predicted=model.predict(origins,payload["inputs"])
    predicted["origin_last_input_utc"]=pd.to_datetime(predicted["origin_last_input_utc"],utc=True)
    keys=["origin_last_input_utc","horizon_hours"]
    joined=predicted.merge(reference,on=keys,validate="one_to_one",suffixes=("_new","_old"))
    assert len(joined)==len(predicted)==len(reference)==2833*72
    delta=float(np.abs(joined.pred_kms_new-joined.pred_kms_old).max())
    assert np.isfinite(predicted.pred_kms).all() and delta<=1e-3,(name,delta)
    with path.open("rb") as stream:
        digest=hashlib.file_digest(stream,"sha256").hexdigest()
    report={"experiment":name,"config_sha256":row["config_sha256"],"rows":len(predicted),"submission_max_abs":delta,"sha256":digest,"passed":True}
    reports.append(report)
    (root/(name+".isolated-verification.json")).write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    del model,predicted,reference,joined
foreign=[name for name in sys.modules if name=="experiments" or name.startswith("experiments.") or name=="expos" or name.startswith("expos.")]
assert not foreign,foreign
report={"passed":True,"models":reports,"rows_per_model":2833*72,"isolated_runtime":True,"experiment_imports":foreign,"package_sha256":payload["package_sha256"],"input_sha256":payload["input_sha256"]}
target="integration-verification.json" if len(reports)==10 else "partial-integration-verification.json"
(root/target).write_text(json.dumps(report,indent=2))
'''
        (isolated / "verify.py").write_text(program)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        environment.pop("PYTHONPATH", None)
        subprocess.run([sys.executable, str(isolated / "verify.py")], cwd=isolated,
                       env=environment, check=True)


if __name__ == "__main__":
    main()

"""Report the frozen two-feature combination without fitting or selecting models."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)
PARENT = ROOT.parent / "offset_roll_complete_ablation_20261106"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"per_seed": values.tolist(), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)), "worst": float(values.max())}


def analyze(root=ROOT, replicates=2000):
    scorer = module(root/"competition.py", "two_removal_scorer")
    stats = module(PARENT/"analyze_results.py", "two_removal_statistics")
    assignment = json.loads((root/"assignment.global.json").read_text())
    context = scorer.prepare_context(scorer.load_observations(assignment["observations"]))
    arrays, scores = [], {}
    for group in ("baseline", "two_removal"):
        scores[group] = []
        for seed in SEEDS:
            out = PARENT/"runs"/f"baseline_s{seed}" if group == "baseline" else root/"runs"/f"drop035_v650_s{seed}"
            if (out/"train.exit").read_text().strip() != "0":
                raise ValueError("Only successful fresh training runs may enter analysis")
            if group == "two_removal" and not (out/"training-validation.json").is_file():
                raise ValueError("Fresh run validation missing")
            frame = pd.read_parquet(out/"predictions.parquet")
            actual = scorer.score_competition(frame, context)
            expected = json.loads((out/"local_score.json").read_text())["competition"]
            array = scorer.align_predictions(frame, context)
            for scope in ("total", "f1", "f2", "f3"):
                if actual[scope]["n_replaced"]:
                    raise ValueError("Missing official forecasts")
                for metric in ("s", "rmse", "trmse", "timse"):
                    if not np.isclose(actual[scope][metric], expected[scope][metric], atol=1e-8, rtol=1e-10):
                        raise ValueError("Stored score does not reproduce")
                mask = np.ones(len(array), bool) if scope == "total" else context["period"] == scope.upper()
                actual[scope]["mae"] = float(np.nanmean(np.abs(array[mask]-context["truth"][mask])))
            arrays.append(array)
            scores[group].append(actual)
    exact, draws, _ = stats.bootstrap_scores(context, np.stack(arrays), replicates, 168)
    delta = exact["s"][3:]-exact["s"][:3]
    uncertainty = stats.distribution((draws["s"][:,3:]-draws["s"][:,:3]).mean(1), float(delta.mean()))
    summaries = {group: {scope:{metric:describe([row[scope][metric] for row in rows])
                           for metric in ("s", "rmse", "trmse", "timse", "mae")}
                        for scope in ("total", "f1", "f2", "f3")}
                 for group,rows in scores.items()}
    base = summaries["baseline"]
    treatment = summaries["two_removal"]
    improvement = 1-treatment["total"]["s"]["mean"]/base["total"]["s"]["mean"]
    passed = bool(replicates >= 2000 and improvement >= .01 and max(delta)<0
                  and uncertainty["ci95"][1]<0
                  and all(treatment["total"][m]["mean"]<=1.01*base["total"][m]["mean"] for m in ("rmse","mae"))
                  and all(treatment[p]["s"]["mean"]<=1.02*base[p]["s"]["mean"] for p in ("f1","f2","f3")))
    result = {"status":"complete", "seeds":list(SEEDS), "baseline":"original A, no B1",
        "summaries":summaries, "mean_delta_s":float(delta.mean()), "per_seed_delta_s":delta.tolist(),
        "improvement_fraction":float(improvement), **uncertainty,
        "bootstrap":{"replicates":replicates,"block_hours":168,"rng_seed":20261104},
        "candidate_gate_passed":passed, "automatic_baseline_promotion":False,
        "publication":"explicit user request irrespective of performance gate",
        "classification":"exploratory user-selected combination after viewing official LOCO results",
        "limitations":["Official periods reused; not independent generalization confirmation.",
                       "Single fixed joint contrast, not additive attribution or per-feature significance.",
                       "Seed scalar-score average is distinct from fixed prediction ensemble score."]}
    incumbent = ROOT.parent/"offset_roll_ablation3_20261105/remote_results/runs"
    incumbent_scores = []
    for seed in SEEDS:
        frame = pd.read_parquet(incumbent/f"b1_short_persist_blend6_s{seed}"/"predictions.parquet")
        score = scorer.score_competition(frame, context)
        if score["total"]["n_replaced"]:
            raise ValueError("Incomplete adopted-incumbent reference")
        incumbent_scores.append(score["total"]["s"])
    result["adopted_A_plus_B1_reference"] = dict(describe(incumbent_scores),
        per_seed_delta_s=(exact["s"][3:]-incumbent_scores).tolist(),
        independent_promotion=False, method="same-observation official re-score of adopted incumbent")
    sota = json.loads((ROOT.parent/"post_ablation_five_axes_20261108/sota-reference.json").read_text())
    if sota.get("state") != "verified" or sota["scorer_version"] != scorer.VERSION:
        raise ValueError("External reference has not passed official reproduction")
    result["external_SOTA_reference"] = dict(sota,
        candidate_mean_improvement=1-treatment["total"]["s"]["mean"]/sota["total_s"],
        target_2pct_met=treatment["total"]["s"]["mean"]<=.98*sota["total_s"])
    dest=root/"analysis";dest.mkdir(exist_ok=True)
    (dest/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    lines=["# Two-feature removal confirmation", "", "|Seed|Original A S|Fresh two-removal S|Delta S|",
           "|---|---:|---:|---:|"]
    for index,seed in enumerate(SEEDS):
        lines.append(f"|{seed}|{scores['baseline'][index]['total']['s']:.6f}|{scores['two_removal'][index]['total']['s']:.6f}|{delta[index]:+.6f}|")
    lines += ["",f"Mean delta S: {delta.mean():+.6f}; relative improvement: {improvement:.3%}.",
              f"95% paired168h CI: {uncertainty['ci95']}; candidate gate: {passed}.",
              f"Adopted A+B1 mean S: {np.mean(incumbent_scores):.6f}; external frozen board SOTA: {sota['total_s']:.6f}; 2% target: {sota['target_s']:.6f}.",
              "", "No automatic adoption, tuning or seed selection. Publication is explicitly requested."]
    (dest/"report.md").write_text("\n".join(lines)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--bootstrap",type=int,default=2000)
    args=parser.parse_args()
    result=analyze(args.root,args.bootstrap)
    print(json.dumps({k:result[k] for k in ("mean_delta_s","improvement_fraction","candidate_gate_passed")}))

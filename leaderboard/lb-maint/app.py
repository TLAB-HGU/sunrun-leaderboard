"""Gradio leaderboard: Submit tab validates+scores, Leaderboard tab lists runs.

Storage: LB_STORE=<local dir> (default ./store) or LB_STORE=hf://<org>/<private-dataset>
(needs HF_TOKEN). Truth/folds live in <store>/truth.parquet and <store>/folds.parquet.
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "scorer"))
from scoring import REGIMES, score, validate  # noqa: E402

STORE = os.environ.get("LB_STORE", str(Path(__file__).parent / "store"))
HF = STORE.startswith("hf://")


def _read(name: str) -> Path:
    if HF:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(STORE[5:], name, repo_type="dataset", token=os.environ["HF_TOKEN"]))
    return Path(STORE) / name


def _write(name: str, data: bytes) -> None:
    if HF:
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=data, path_in_repo=name, repo_id=STORE[5:], repo_type="dataset")
    else:
        p = Path(STORE) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _results() -> list[dict]:
    if HF:
        from huggingface_hub import HfApi
        files = [f for f in HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(STORE[5:], repo_type="dataset") if f.startswith("results/")]
        return [json.loads(_read(f).read_text()) for f in files]
    return [json.loads(p.read_text()) for p in sorted((Path(STORE) / "results").glob("*.json"))]


def submit(pred_file, meta_file, key=""):
    required = os.environ.get("SUBMIT_KEY")
    if required and not hmac.compare_digest(str(key or ""), required):
        return "제출 키가 올바르지 않습니다."
    if pred_file is None or meta_file is None:
        return "predictions 파일과 meta.yaml을 모두 올려주세요."
    meta = yaml.safe_load(Path(meta_file).read_text())
    for k in ("team_member", "experiment", "description", "no_future_leakage"):
        if not meta.get(k):
            return f"meta.yaml에 `{k}` 필드가 필요합니다(no_future_leakage는 true여야 함)."
    path = Path(pred_file)
    pred = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    folds, truth = pd.read_parquet(_read("folds.parquet")), pd.read_parquet(_read("truth.parquet"))
    errs = validate(pred, folds)
    if errs:
        return "제출 거부:\n- " + "\n- ".join(errs)
    naive = pd.read_parquet(_read("naive.parquet"))
    res = score(pred, truth, folds, naive)
    now = time.time()
    ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{int(now % 1 * 1000):03d}Z"  # ms suffix: no same-second collisions
    tag = f"{ts}_{meta['team_member']}_{meta['experiment']}"
    rec = {"submitted_utc": ts, **{k: meta[k] for k in ("team_member", "experiment", "description")}, **res}
    try:
        _write(f"submissions/{tag}{path.suffix}", path.read_bytes())  # raw file first so a scored row always has its source
        _write(f"results/{tag}.json", json.dumps(rec, ensure_ascii=False).encode())
    except Exception as e:  # surface storage failures to the user instead of a bare traceback
        return f"저장 실패(채점은 완료됐지만 기록되지 않음, 다시 제출하세요): {type(e).__name__}: {e}"
    vs = res.get("vs_naive", {})
    return (f"채점 완료: MSE={res['mse']:.1f}, MSE(관측만)={res['mse_observed']:.1f}, skill vs naive={res.get('skill_vs_naive', float('nan')):+.3f}, "
            f"Naive를 95% CI로 이김={vs.get('beats_reference')} (차이 CI {vs.get('ci95')})")


def leaderboard(sort_by: str):
    rows = []
    for r in _results():
        vs = r.get("vs_naive") or {}
        row = {"experiment": r["experiment"], "member": r["team_member"], "MSE": r["mse"], "MSE_obs": r.get("mse_observed"),
               "RMSE": r["rmse"], "regime_bal_MSE": r["regime_balanced_mse"], "skill_vs_naive": r.get("skill_vs_naive"),
               "beats_naive_CI": vs.get("beats_reference"), "submitted": r["submitted_utc"]}
        for g in REGIMES:
            c = r["by_regime"][g]
            row[g] = c["mse"] if c["reliable"] else None  # N 부족 셀은 비워 둔다
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("submitted").groupby(["member", "experiment"]).tail(1)  # 같은 실험은 최신 버전
    return df.sort_values(sort_by).round(3).reset_index(drop=True)


with gr.Blocks(title="Solar wind 72h leaderboard") as demo:
    gr.Markdown("# Solar wind 72h — 2026-06~09 leaderboard\n낮은 MSE가 위. 레짐 열은 서로 겹치지 않는 72h 블록이 3개 미만이면 비웁니다(진단용, 순위는 전체 MSE 기준). beats_naive_CI는 72h 블록 부트스트랩 95% CI로 Naive보다 낮은지 여부입니다.")
    with gr.Tab("Leaderboard"):
        sort = gr.Dropdown(["MSE", "regime_bal_MSE", "RMSE"], value="MSE", label="정렬")
        table = gr.Dataframe(leaderboard("MSE"))
        sort.change(leaderboard, sort, table)
        gr.Button("새로고침").click(leaderboard, sort, table)
    with gr.Tab("Submit"):
        pf, mf = gr.File(label="predictions (.parquet/.csv)"), gr.File(label="meta.yaml")
        key, out = gr.Textbox(label="팀 제출 키", type="password"), gr.Textbox(label="결과")
        gr.Button("제출").click(submit, [pf, mf, key], out, api_name="submit")

if __name__ == "__main__":
    demo.launch()

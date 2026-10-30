"""Command line for the Experiment OS. Run from leaderboard/: `.venv-fusion/bin/python -m expos <command> ...`"""
import argparse
import json
import sys

from . import archive, axes, core, evaluate, experience, rocv


def _running(x):
    r = x.get("running")
    if not r:
        return ""
    return f"  {'STALE (process gone, no run record)' if r.get('stale') else 'RUNNING'} gpu={r.get('gpu')}"


def _board():
    s = core.state()
    runs = sorted((r for x in s["experiments"].values() for r in x["runs"]), key=lambda r: r["seq"])[-20:]
    fails = sum(r["rc"] != 0 for r in runs)
    print(f"directions: {len(s['directions'])}  experiments: {len(s['experiments'])}  "
          f"frozen: {s['frozen']['id'] if s['frozen'] else '-'}  official: {'done' if s['official'] else '-'}  "
          f"rc!=0 in last {len(runs)} runs: {fails}")
    for d in s["directions"].values():
        print(f"  {d['id']:10s} [{d.get('state')}] parent={d.get('parent') or '-'} {d.get('title')}")
    print(f"{'id':6s} {'dir':8s} {'kind':9s} {'owner':12s} {'dev':>8s} {'sel':>8s} {'aud':>8s} {'meanrel':>8s} "
          f"{'F1':>6s} gate poison reviews flags")
    for x in s["experiments"].values():
        ev = x["evals"][-1] if x["evals"] else {}
        sm, g = ev.get("summary", {}), ev.get("gate") or {}
        mr = g.get("mean_relative")
        f1 = sm.get("pooled_f1")
        print(f"{x['id']:6s} {x['direction']:8s} {x['kind']:9s} {x['owner'][:12]:12s} "
              f"{sm.get('dev', float('nan')):8.1f} {sm.get('selection', float('nan')):8.1f} "
              f"{sm.get('audit', float('nan')):8.1f} {mr if mr is not None else float('nan'):+8.4f} "
              f"{f1 if f1 is not None else float('nan'):6.3f} {'Y' if g.get('pass') else 'n':4s} "
              f"{'Y' if ev.get('poison_ok') else 'n':6s} {','.join(r['verdict'][0] for r in x['reviews']) or '-':7s} "
              f"{len(x['flags'])}{_running(x)}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="expos")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    sub.add_parser("verify")
    sub.add_parser("board")
    sub.add_parser("pareto")
    p = sub.add_parser("rocv-eval"); p.add_argument("id"); p.add_argument("--reference", required=True)
    sub.add_parser("rocv-folds")
    p = sub.add_parser("axes"); p.add_argument("--last", type=int, help="only the last N experiments")
    p = sub.add_parser("show"); p.add_argument("id")
    p = sub.add_parser("direction"); p.add_argument("id"); p.add_argument("--title"); p.add_argument("--rationale", default="")
    p.add_argument("--parent"); p.add_argument("--state"); p.add_argument("--reason", default="")
    p = sub.add_parser("register"); p.add_argument("--direction", required=True); p.add_argument("--owner", required=True)
    p.add_argument("--hypothesis", required=True); p.add_argument("--cmd", required=True)
    p.add_argument("--kind", default="full", choices=core.KINDS); p.add_argument("--code", nargs="*", default=[])
    p.add_argument("--reference"); p.add_argument("--experience", nargs="*", default=[])
    p = sub.add_parser("run"); p.add_argument("id"); p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--gpu", action="store_true"); p.add_argument("--gpus", type=int, default=1, choices=(1, 2))
    p.add_argument("--prefer-gpu", type=int, choices=(0, 1)); p.add_argument("--background", action="store_true")
    p = sub.add_parser("evaluate"); p.add_argument("id"); p.add_argument("--reference")
    p = sub.add_parser("review"); p.add_argument("id"); p.add_argument("--reviewer", required=True)
    p.add_argument("--verdict", required=True, choices=core.VERDICTS); p.add_argument("--notes", required=True)
    p = sub.add_parser("freeze"); p.add_argument("id"); p.add_argument("--final")
    sub.add_parser("official-score")
    p = sub.add_parser("exp-add"); p.add_argument("--record", required=True); p.add_argument("--applicability", required=True)
    p.add_argument("--finding", required=True); p.add_argument("--implication", required=True)
    p.add_argument("--tags", nargs="+", required=True); p.add_argument("--author", required=True)
    p = sub.add_parser("exp-query"); p.add_argument("--text", default=""); p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("-k", type=int, default=8)
    a = ap.parse_args(argv)
    try:
        if a.action == "init":
            r = core.init()
        elif a.action == "verify":
            r = {"lines": core.verify(), "protected_changed": core.check_protected()}
        elif a.action == "board":
            return _board()
        elif a.action == "rocv-eval":
            r = rocv.evaluate_rocv(a.id, a.reference)
            r = {k: r[k] for k in ("id", "gate")}
        elif a.action == "rocv-folds":
            r = {"folds": rocv.FOLDS, "rule": rocv.__doc__}
        elif a.action == "pareto":
            r = archive.pareto()
        elif a.action == "axes":
            r = axes.report(a.last)
            print(f"design-space coverage over {r['n']} experiments (window={r['window']}); LOCKED-IN axes (<15% non-default): {r['locked_axes']}")
            for x in r["axes"]:
                print(f"{'LOCKED ' if x['locked_in'] else '       '}{x['axis']:7s} default={x['default']:16s} non-default {x['non_default_share']*100:5.1f}%  "
                      f"used={x['values_used']}  never={x['never_tried']}")
            return 0
        elif a.action == "show":
            r = core.state()["experiments"][a.id]
        elif a.action == "direction":
            r = (core.set_direction(a.id, a.state, a.reason) if a.state else
                 core.add_direction(a.id, a.title or a.id, a.rationale, a.parent))
        elif a.action == "register":
            r = core.register(a.direction, a.owner, a.hypothesis, a.cmd, a.kind, a.code, a.reference, a.experience)
        elif a.action == "run":
            r = (core.run_background if a.background else core.run)(a.id, a.timeout, a.gpu, a.gpus, a.prefer_gpu)
        elif a.action == "evaluate":
            r = evaluate.evaluate(a.id, a.reference)
            r = {k: r[k] for k in ("id", "summary", "gate", "poison_ok")}
        elif a.action == "review":
            r = core.review(a.id, a.reviewer, a.verdict, a.notes)
        elif a.action == "freeze":
            r = core.freeze(a.id, a.final)
        elif a.action == "official-score":
            r = evaluate.official_score()
        elif a.action == "exp-add":
            r = experience.add(a.record, a.applicability, a.finding, a.implication, a.tags, a.author)
        elif a.action == "exp-query":
            r = experience.query(a.text, a.tags, a.k)
    except core.ExposError as e:
        print(f"expos: {e}", file=sys.stderr)
        return 2
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:6000])
    return 0

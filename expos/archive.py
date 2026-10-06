"""Quality-diversity view of the experiment graph (GEAR / MAP-Elites style): Pareto front and niche elites.

Objectives (vs reference E0003, common rows): minimise mean relative MSE, maximise pooled unique-event F1.
Niche = top-level lineage (direct child of D0) x kind. Read-only: nothing is appended to the graph.
"""
from . import core

REFERENCE = "E0003"


def lineage_root(directions, d):
    while d in directions and directions[d].get("parent") not in (None, "D0"):
        d = directions[d]["parent"]
    return d


def points(s=None, reference=REFERENCE):
    s = s or core.state()
    out = []
    for x in s["experiments"].values():
        if not x["evals"] or x["flags"]:
            continue
        g = x["evals"][-1].get("gate") or {}
        if g.get("reference") != reference or g.get("mean_relative") is None or g.get("pooled_f1") is None:
            continue
        out.append({"id": x["id"], "direction": x["direction"], "kind": x["kind"],
                    "niche": f"{lineage_root(s['directions'], x['direction'])}/{x['kind']}",
                    "mean_rel": round(g["mean_relative"], 4), "f1": round(g["pooled_f1"], 4),
                    "n_improved": g.get("n_improved")})
    return out


def dominated(a, b):
    """True if b dominates a (b no worse on both objectives and strictly better on one)."""
    return (b["mean_rel"] <= a["mean_rel"] and b["f1"] >= a["f1"]
            and (b["mean_rel"] < a["mean_rel"] or b["f1"] > a["f1"]))


def pareto(s=None, reference=REFERENCE):
    pts = points(s, reference)
    front = sorted((p for p in pts if not any(dominated(p, q) for q in pts)), key=lambda p: p["mean_rel"])
    niches = {}
    for p in pts:
        n = niches.setdefault(p["niche"], {"mse_elite": p, "f1_elite": p, "count": 0})
        n["count"] += 1
        if p["mean_rel"] < n["mse_elite"]["mean_rel"]:
            n["mse_elite"] = p
        if p["f1"] > n["f1_elite"]["f1"]:
            n["f1_elite"] = p
    return {"reference": reference, "n_points": len(pts), "front": front,
            "niches": dict(sorted(niches.items()))}

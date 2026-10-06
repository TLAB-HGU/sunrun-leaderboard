"""Design-space coverage (read-only): which axes of the research design have actually been varied?

Each experiment is described by values on ten axes. Values come from an explicit `[axes: loss=huber, sel=permutation]` tag
in the hypothesis (preferred) and, for untagged experiments, from keyword detection on the hypothesis and the registered code.
An axis whose experiments all sit on its default value is a lock-in on that axis, however diverse the directions look.
"""
import re

from . import core

# axis -> (default, {value: regex}); values not listed for an axis can still be set through an explicit tag.
AXES = {
    "feat": ("ace+ch+strips", {
        "suvi-hires": r"hi-?res|high.?res|raw suvi|1.?2.?deg", "dinov3/embedding": r"dino|embedding|v-?jepa|pca",
        "dimming/diff": r"dimming|brightness.?change|difference image|diff series", "precursor(epam/imf)": r"epam|imf|precursor|shock sig|icme",
        "analog/recurrence-ext": r"analog|replay|rotation", "spectral/wavelet": r"wavelet|fourier|spectr|fft",
        "physics-derived": r"alfv|mach|dynamic.?pressure|ram pressure|plasma.?beta|geometry|b0|asymmetr",
        "learned-image": r"cnn|conv|lonmap|backprop"}),
    "sel": ("none(all cols)", {
        "importance-prune": r"prune|drop.*(feature|col)|top-?k feature", "permutation": r"permutation", "shap": r"\bshap\b",
        "group-ablation": r"ablat|leave.?one.?group", "l1/lasso": r"lasso|l1 penalty|elastic", "stability/boruta": r"boruta|stability select|mrmr"}),
    "model": ("xgb-gbdt", {
        "lgbm": r"lightgbm|lgbm", "catboost": r"catboost", "rf/extratrees": r"random.?forest|extra.?trees|xgbrf|\[family=rf",
        "dart": r"\bdart\b", "mlp/tcn": r"\btcn\b|\bmlp\b|neural|\[family=nn", "rnn": r"\bgru\b|\blstm\b",
        "transformer": r"transformer|attention", "linear/poly": r"lasso|polynom|cubic|ridge|\[family=linear"}),
    "loss": ("mse", {
        "huber": r"huber", "absolute/quantile": r"quantile|pinball|absolute.?error|\bmae\b", "tweedie/gamma/log-link": r"tweedie|gamma|log.?link",
        "weighted-mse": r"weighted (loss|mse)|sample.?weight|riseweight|bandweight|\bweights?\b.*(loss|fit)",
        "shape/temporal(dilate,tilde)": r"dilate|tilde|soft.?dtw|shape loss", "focal/event-bce": r"focal|binary:logistic|event.*bce|rise.?head"}),
    "opt": ("fixed-params", {
        "hpo": r"\bhpo\b|optuna|tpe|hyperparam search", "lr-schedule": r"lr.?schedule|cosine|one.?cycle|warmup", "early-stop": r"early.?stop|patience",
        "adamw/sgd": r"adamw|\badam\b|\bsgd\b", "swa/ema": r"\bswa\b|\bema\b|weight averaging", "regularisation-sweep": r"reg_lambda|min_child|colsample|subsample sweep"}),
    "target": ("direct-level", {
        "residual": r"residual|vs persistence|on top of", "level+shape": r"level.*shape|mean.?level|decompos", "log/rank/quantile-tf": r"log1p|rank|quantile transform|standardi[sz]e target",
        "event-gated": r"event gate|gate|hard.?switch|branch", "multi-task": r"multi.?task|auxiliary"}),
    "data": ("expanding-window", {
        "rolling-window": r"rolling window|last \d+ years|recent.?only", "recency-weight": r"recency|time.?decay", "event-oversample": r"oversampl|reweight|event.?weight|surge.?weight",
        "regime-specific": r"regime.?(specific|conditional)|per.?regime", "augmentation": r"augment|jitter|mixup", "denser-grid": r"3h grid|full density|1h grid"}),
    "ens": ("single-model", {
        "seed-mean": r"seed.?mean|seed.?avg|\d.?seeds?\b|(average|mean|vote)\w* (\w+ )?seeds|multi.?seed", "bagging": r"bagg|bootstrap aggregat", "stacking": r"stack|meta.?learner",
        "router/gate": r"router|routing|gate", "oof-blend": r"oof.?blend|blend", "crossover": r"crossover|parent"}),
    "post": ("none", {
        "distribution-map": r"box.?cox|quantile map|distribution map", "smoothing": r"smooth", "clipping/monotone": r"clip|monoton",
        "calibration": r"calibrat|isotonic", "hard-switch": r"hard.?switch|hard branch"}),
    "val": ("fixed-3-periods", {
        "rolling-origin": r"rolling.?origin|walk.?forward|timeseriessplit|n.?folds", "seed-replicates": r"seed.?repro|repro|replicate",
        "bootstrap-ci": r"bootstrap ci|paired bootstrap|block bootstrap"}),
}
TAG = re.compile(r"\[axes:\s*([^\]]+)\]", re.I)


def explicit_tags(text):
    out = {}
    m = TAG.search(text or "")
    for part in (m.group(1).split(",") if m else []):
        if "=" in part:
            k, v = part.split("=", 1)
            out.setdefault(k.strip().lower(), set()).update(x.strip() for x in v.split("+") if x.strip())
    return out


def classify(x):
    """Non-default axis values for one experiment: {axis: {values}}."""
    text = x["hypothesis"].lower()
    found = explicit_tags(x["hypothesis"])
    if not found:  # untagged: keyword detection (tagged experiments are taken at their word)
        for axis, (_, vals) in AXES.items():
            for v, pat in vals.items():
                if re.search(pat, text):
                    found.setdefault(axis, set()).add(v)
    return {a: v for a, v in found.items() if a in AXES}


def report(window=None, s=None):
    s = s or core.state()
    ex = [x for x in s["experiments"].values() if not x["owner"].startswith("claude-")]
    if window:
        ex = ex[-window:]
    cls = [classify(x) for x in ex]
    rows = []
    for axis, (default, vals) in AXES.items():
        used = {}
        for c, x in zip(cls, ex):
            for v in c.get(axis, ()):
                used.setdefault(v, []).append(x["id"])
        non_default = sum(1 for c in cls if c.get(axis))
        share = non_default / len(ex) if ex else 0.0
        rows.append({"axis": axis, "default": default, "n_experiments": len(ex), "non_default_share": round(share, 3),
                     "values_used": {v: len(ids) for v, ids in sorted(used.items(), key=lambda kv: -len(kv[1]))},
                     "never_tried": sorted(set(vals) - set(used)),
                     "locked_in": share < 0.15})
    return {"window": window or "all", "n": len(ex), "axes": rows,
            "locked_axes": [r["axis"] for r in rows if r["locked_in"]]}

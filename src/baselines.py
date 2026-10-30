"""Simple baselines that DeepLOB has to beat.

- majority class: always predict the most common training label
- logistic regression on the window's latest snapshot (40 LOB values),
  plain and with class weights balanced against the label imbalance

Reported as accuracy and macro-F1 (each class weighted equally), since
"stationary" dominates the labels and accuracy alone rewards ignoring it.

Run ``python -m src.baselines`` from the project root.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import HORIZON_INDEX, LOBWindows, build_splits

EVAL_SPLITS = ("val", "test")


def latest_snapshot(ds: LOBWindows) -> np.ndarray:
    """(n_windows, 40) LOB values at each window's final snapshot."""
    return ds.x.numpy()[ds.ends]


def evaluate(splits: dict[str, LOBWindows]) -> dict[str, dict[str, tuple[float, float]]]:
    """{model: {split: (accuracy, macro-F1)}} for every baseline."""
    y = {name: ds.targets for name, ds in splits.items()}
    x = {name: latest_snapshot(ds) for name, ds in splits.items()}
    majority = np.bincount(y["train"]).argmax()

    preds = {"majority class": {s: np.full_like(y[s], majority) for s in EVAL_SPLITS}}
    for name, class_weight in (("logistic regression", None), ("logistic regression, balanced", "balanced")):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=class_weight))
        model.fit(x["train"], y["train"])
        preds[name] = {s: model.predict(x[s]) for s in EVAL_SPLITS}

    return {model: {s: (accuracy_score(y[s], p[s]), f1_score(y[s], p[s], average="macro"))
                    for s in EVAL_SPLITS}
            for model, p in preds.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score simple baselines on FI-2010.")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50], choices=sorted(HORIZON_INDEX))
    parser.add_argument("--split", default="paper", choices=["paper", "by_day"])
    parser.add_argument("--respect-boundaries", action="store_true")
    args = parser.parse_args()

    print(f"split={args.split} respect_boundaries={args.respect_boundaries}")
    print(f"{'k':>4}  {'model':<30} {'val acc':>8} {'val F1':>8} {'test acc':>9} {'test F1':>8}")
    for k in args.k:
        t = time.time()
        results = evaluate(build_splits(args.split, args.respect_boundaries, k))
        for model, r in results.items():
            (va, vf), (ta, tf) = r["val"], r["test"]
            print(f"{k:>4}  {model:<30} {va:>8.3f} {vf:>8.3f} {ta:>9.3f} {tf:>8.3f}")
        print(f"      ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

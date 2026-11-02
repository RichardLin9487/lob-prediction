"""Does DeepLOB's signal survive the bid-ask spread on the test days?

A simple rule trades each stock separately: on an "up" prediction buy, on a
"down" prediction short, hold for a fixed number of snapshots, then close. Only
one position is open at a time and every trade closes within its stock-day.
Returns are in basis points of the mid-price at entry:

    gross  enter and exit at the mid-price, so no trading cost
    net    cross the spread: buy at the best ask, sell at the best bid

Two controls use the same rule:

    random  random predictions with the model's share of up / stationary / down
    oracle  the true labels, the most any prediction of these labels can earn

This is optimistic: trades fill at the snapshot the prediction uses, at the best
quote, with no fees, latency or market impact.

Run ``python -m src.backtest`` from the project root after training the clean
setup with the paper's settings (see the README).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.dataset import CLASS_NAMES, HORIZON_INDEX, PAPER_TRAIN_FOLD, Series, build_splits, day_major
from src.load_data import N_DAYS, load_days
from src.model import DeepLOB
from src.train import RUNS_DIR, DeviceWindows, evaluate, pick_device

UP, STATIONARY, DOWN = (CLASS_NAMES.index(c) for c in ("up", "stationary", "down"))
RUN_NAME = "deeplob_k{k}_clean_zscore_papertrain_seed{seed}"
HOLDS = (1, 2, 5, 10, 20, 50, 100)
SEEDS = (0, 1, 2)
N_RANDOM_DRAWS = 10


def block_stops(series: Series, ends: np.ndarray) -> np.ndarray:
    """First index after the stock-day block holding each of ``ends``."""
    stops = np.r_[series.starts[1:], len(series)]
    return stops[np.searchsorted(series.starts, ends, side="right") - 1]


def entries(signal: np.ndarray, ends: np.ndarray, stops: np.ndarray, hold: int) -> np.ndarray:
    """Positions in ``ends`` where a trade opens, skipping signals while one is open."""
    active = np.flatnonzero(signal != STATIONARY)
    times, chosen, i = ends[active], [], 0
    while i < len(active):
        t = times[i]
        if t + hold < stops[active[i]]:
            chosen.append(active[i])
            i = int(np.searchsorted(times, t + hold))
        else:
            i += 1
    return np.array(chosen, dtype=np.int64)


def simulate(signal: np.ndarray, ends: np.ndarray, stops: np.ndarray, series: Series, hold: int) -> dict:
    """Trade ``signal`` and summarise the returns, overall and per stock-day."""
    ask, bid = series.lob[:, 0].astype(np.float64), series.lob[:, 2].astype(np.float64)
    mid = (ask + bid) / 2
    chosen = entries(signal, ends, stops, hold)
    t, x = ends[chosen], ends[chosen] + hold
    side = np.where(signal[chosen] == UP, 1.0, -1.0)
    gross = side * (mid[x] - mid[t]) / mid[t] * 1e4
    net = np.where(side > 0, bid[x] - ask[t], bid[t] - ask[x]) / mid[t] * 1e4
    blocks = np.searchsorted(series.starts, t, side="right") - 1
    block_gross = [gross[blocks == b].mean() for b in np.unique(blocks)]
    block_net = [net[blocks == b].mean() for b in np.unique(blocks)]
    return {"trades": len(chosen), "long_share": float((side > 0).mean()), "hit_rate": float((gross > 0).mean()),
            "gross_bps": float(gross.mean()), "net_bps": float(net.mean()),
            "stock_days": len(block_gross), "stock_days_gross_positive": int(np.sum(np.array(block_gross) > 0)),
            "stock_days_net_positive": int(np.sum(np.array(block_net) > 0))}


def predict(k: int, seed: int, data: DeviceWindows, device: torch.device) -> np.ndarray:
    """Test predictions of a trained run, checked against the accuracy it saved."""
    run_dir = RUNS_DIR / RUN_NAME.format(k=k, seed=seed)
    model = DeepLOB().to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    result = evaluate(model, data)
    saved = json.loads((run_dir / "results.json").read_text())["test"]["accuracy"]
    if abs(result["accuracy"] - saved) > 1e-3:
        raise ValueError(f"{run_dir.name}: accuracy {result['accuracy']:.4f} differs from saved {saved:.4f}")
    return result["pred"]


def mean_over(results: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in results]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest DeepLOB predictions against the bid-ask spread.")
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50], choices=sorted(HORIZON_INDEX))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    # DecPre keeps real price ratios; the models were trained on z-scored inputs.
    series = day_major(load_days(range(PAPER_TRAIN_FOLD + 1, N_DAYS + 1), normalization="decpre"))
    spread = (series.lob[:, 0] - series.lob[:, 2]) / ((series.lob[:, 0] + series.lob[:, 2]) / 2) * 1e4
    print(f"quoted spread: mean {spread.mean():.2f} bps, median {np.median(spread):.2f} bps")

    all_results = {}
    for k in args.k:
        ds = build_splits("by_day", True, k, normalization="zscore")["test"]
        if not np.array_equal(series.labels[:, HORIZON_INDEX[k]][ds.ends].astype(np.int64) - 1, ds.targets):
            raise ValueError("DecPre prices and z-scored windows are not aligned")
        ends, stops = ds.ends, block_stops(series, ds.ends)
        data = DeviceWindows(ds, device)
        preds = [predict(k, seed, data, device) for seed in SEEDS]
        rng = np.random.default_rng(0)

        print(f"\nk={k}  (model: mean of {len(SEEDS)} seeds, random: mean of {N_RANDOM_DRAWS} draws per seed)")
        print(f"{'hold':>4} {'trades':>7} {'long':>5} {'hit':>5} | {'gross bps':>9} {'net bps':>8} "
              f"{'stock-days gross>0':>18} {'net>0':>5} | {'random gross':>12} {'net':>7} | {'oracle gross':>12} {'net':>7}")
        all_results[k] = {}
        for hold in HOLDS:
            model = [simulate(p, ends, stops, series, hold) for p in preds]
            random = [simulate(rng.choice(3, size=len(p), p=np.bincount(p, minlength=3) / len(p)),
                               ends, stops, series, hold)
                      for p in preds for _ in range(N_RANDOM_DRAWS)]
            oracle = simulate(ds.targets, ends, stops, series, hold)
            all_results[k][hold] = {"model": model, "random": random, "oracle": oracle}
            print(f"{hold:>4} {mean_over(model, 'trades'):>7.0f} {mean_over(model, 'long_share'):>5.2f} "
                  f"{mean_over(model, 'hit_rate'):>5.2f} | {mean_over(model, 'gross_bps'):>9.2f} "
                  f"{mean_over(model, 'net_bps'):>8.2f} {mean_over(model, 'stock_days_gross_positive'):>15.1f}/15 "
                  f"{mean_over(model, 'stock_days_net_positive'):>5.1f} | {mean_over(random, 'gross_bps'):>12.2f} "
                  f"{mean_over(random, 'net_bps'):>7.2f} | {oracle['gross_bps']:>12.2f} {oracle['net_bps']:>7.2f}")

    (RUNS_DIR / "backtest_clean.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved {RUNS_DIR / 'backtest_clean.json'}")


if __name__ == "__main__":
    main()

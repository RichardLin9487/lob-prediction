"""Load the FI-2010 limit order book benchmark into per-day numpy arrays.

File format
-----------
Every ``.txt`` file is a whitespace-separated matrix of shape (149, N): one row
per feature, one column per LOB event. Rows are:

    0-39     raw LOB, 10 levels x [ask price, ask volume, bid price, bid volume]
             (row 4*l + 0/1/2/3 for level l = 0..9)
    40-143   104 hand-crafted features from Ntakaris et al. (2018)
    144-148  labels for 5 prediction horizons, 1 = up, 2 = stationary, 3 = down

There are 10 trading days, each containing 5 stocks concatenated one after the
other. The files follow an anchored cross-fold ("CF_k") setup:

    Test_..._CF_k   = day k+1 alone, stocks in order          (k = 1..9)
    Train_..._CF_k  = days 1..k, ordered stock-major:
                      [s1d1, s1d2, ..., s1dk, s2d1, ..., s5dk]

So ``Train_CF_k`` cannot be split into days by column counts alone. Instead,
day 1 is read from ``Train_CF_1`` and day d >= 2 from ``Test_CF_{d-1}``.

Normalisation caveat: ``DecPre`` is a fixed rescaling, so days from different
files are directly comparable. ``ZScore`` and ``MinMax`` are fitted on each
fold's training days, so day 1 here is normalised with day-1 statistics, day 2
with day-1 statistics, day 3 with days-1..2 statistics, and so on.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "BenchmarkDatasets"

N_DAYS = 10
N_STOCKS = 5
N_LEVELS = 10
N_ROWS = 149
LOB_ROWS = slice(0, 40)
FEATURE_ROWS = slice(40, 144)
LABEL_ROWS = slice(144, 149)
# Label horizons in dataset rows. Each row is a snapshot taken every 10 order
# book events, which is why DeepLOB and follow-ups call these k = 10, 20, 30, 50, 100.
HORIZONS = (1, 2, 3, 5, 10)

# normalisation key -> (directory prefix, directory name, name used in filenames)
_NORMALISATIONS = {
    "zscore": ("1", "Zscore", "ZScore"),
    "minmax": ("2", "MinMax", "MinMax"),
    "decpre": ("3", "DecPre", "DecPre"),
}


@dataclass
class Day:
    """One trading day, time-major (row t = event t)."""

    day: int                  # 1..10
    lob: np.ndarray           # (T, 40)  float, column layout as rows 0-39 above
    features: np.ndarray      # (T, 104) float
    labels: np.ndarray        # (T, 5)   int8 in {1, 2, 3}, one column per HORIZONS entry
    stock_starts: np.ndarray  # (5,)     index of each stock's first event; stock_starts[0] == 0

    def __len__(self) -> int:
        return self.lob.shape[0]

    def stock_slices(self) -> list[slice]:
        """Slices selecting each of the 5 stocks' events within this day."""
        ends = np.r_[self.stock_starts[1:], len(self)]
        return [slice(int(s), int(e)) for s, e in zip(self.stock_starts, ends)]


def file_path(split: str, k: int, normalization: str = "decpre", auction: bool = False,
              root: Path | str = DATA_ROOT) -> Path:
    """Path of ``{Train,Test}_Dst_{NoAuction,Auction}_{norm}_CF_{k}.txt``."""
    if split not in ("Train", "Test"):
        raise ValueError(f"split must be 'Train' or 'Test', got {split!r}")
    if normalization not in _NORMALISATIONS:
        raise ValueError(f"normalization must be one of {sorted(_NORMALISATIONS)}, got {normalization!r}")
    prefix, dir_name, file_name = _NORMALISATIONS[normalization]
    a = "Auction" if auction else "NoAuction"
    sub = "Training" if split == "Train" else "Testing"
    return (Path(root) / a / f"{prefix}.{a}_{dir_name}" / f"{a}_{dir_name}_{sub}"
            / f"{split}_Dst_{a}_{file_name}_CF_{k}.txt")


def read_raw(path: Path | str, dtype=np.float64) -> np.ndarray:
    """Read one FI-2010 file as its stored (149, N) matrix."""
    data = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c").to_numpy()
    if data.shape[0] != N_ROWS:
        raise ValueError(f"{path}: expected {N_ROWS} rows, got {data.shape[0]}")
    return data


def find_stock_starts(lob: np.ndarray) -> np.ndarray:
    """Locate the 4 stock changes in a day from the 4 largest mid-price jumps.

    The 5 stocks trade at clearly different price levels, so the boundaries
    are jumps orders of magnitude larger than any intraday move. This is
    invariant to the per-row affine normalisations used by the dataset.
    """
    mid = (lob[:, 0] + lob[:, 2]) / 2
    jumps = np.abs(np.diff(mid))
    order = np.argsort(jumps)[::-1]
    boundary, runner_up = jumps[order[N_STOCKS - 2]], jumps[order[N_STOCKS - 1]]
    # Across all 60 day-files the margin is >= 20x without auctions and >= 3.6x
    # with them (opening/closing auction prints are the next-largest jumps).
    if not boundary > 2 * runner_up:
        raise ValueError(f"stock boundaries are ambiguous: 4th-largest mid-price jump {boundary:.4g} "
                         f"vs 5th-largest {runner_up:.4g}")
    return np.r_[0, np.sort(order[:N_STOCKS - 1]) + 1]


def day_path(day: int, normalization: str = "decpre", auction: bool = False,
             root: Path | str = DATA_ROOT) -> Path:
    """File holding exactly one day: Train_CF_1 for day 1, Test_CF_{day-1} otherwise."""
    if not 1 <= day <= N_DAYS:
        raise ValueError(f"day must be in 1..{N_DAYS}, got {day}")
    if day == 1:
        return file_path("Train", 1, normalization, auction, root)
    return file_path("Test", day - 1, normalization, auction, root)


def load_day(day: int, normalization: str = "decpre", auction: bool = False,
             root: Path | str = DATA_ROOT, dtype=np.float64) -> Day:
    raw = read_raw(day_path(day, normalization, auction, root), dtype=dtype)
    lob = np.ascontiguousarray(raw[LOB_ROWS].T)
    return Day(
        day=day,
        lob=lob,
        features=np.ascontiguousarray(raw[FEATURE_ROWS].T),
        labels=raw[LABEL_ROWS].T.astype(np.int8),
        stock_starts=find_stock_starts(lob),
    )


def load_days(days=range(1, N_DAYS + 1), normalization: str = "decpre", auction: bool = False,
              root: Path | str = DATA_ROOT, dtype=np.float64) -> list[Day]:
    """Load the requested days (1-based) in order.

    ``normalization`` is one of "decpre" (default; consistent across days),
    "zscore" or "minmax" (fold-dependent, see module docstring).
    """
    return [load_day(d, normalization, auction, root, dtype) for d in days]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise FI-2010 per-day arrays.")
    parser.add_argument("--normalization", default="decpre", choices=sorted(_NORMALISATIONS))
    parser.add_argument("--auction", action="store_true")
    args = parser.parse_args()

    days = load_days(normalization=args.normalization, auction=args.auction)
    print(f"{'day':>3} {'events':>7}  {'stock lengths':<36}  label share up/stat/down at horizon {HORIZONS[-1]}")
    for d in days:
        lengths = [s.stop - s.start for s in d.stock_slices()]
        share = np.bincount(d.labels[:, -1], minlength=4)[1:] / len(d)
        print(f"{d.day:>3} {len(d):>7}  {str(lengths):<36}  {share.round(3)}")
    print(f"total events: {sum(len(d) for d in days)}")


if __name__ == "__main__":
    main()

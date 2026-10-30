"""Windowed FI-2010 samples for DeepLOB.

Each sample is the last ``window`` LOB snapshots, shaped (1, window, 40) as
DeepLOB's first conv layer expects, labelled with the class at the window's
final snapshot: 0 = up, 1 = stationary, 2 = down.

Two switches control how closely the setup follows the official DeepLOB code:

split="paper"   train/val = Train_CF_7 (days 1-7, stock-major) cut at 80%;
                test = days 8, 9, 10 concatenated (Test_CF_7, 8, 9).
split="by_day"  train = days 1-6, val = day 7, test = days 8-10.

respect_boundaries=False  windows may span two stocks or two days, as in the
                          official code.
respect_boundaries=True   every window lies inside one stock on one day.

With split="by_day" and normalization "zscore"/"minmax", days carry different
fold statistics (see load_data). "paper" reads Train_CF_7 itself, so it is
faithful for every normalization.

Run ``python -m src.dataset`` from the project root for a summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.load_data import (DATA_ROOT, LABEL_ROWS, LOB_ROWS, N_STOCKS, Day, file_path,
                           load_days, read_raw)

# DeepLOB's name for each horizon -> label column (see load_data.HORIZONS)
HORIZON_INDEX = {10: 0, 20: 1, 30: 2, 50: 3, 100: 4}
CLASS_NAMES = ("up", "stationary", "down")
PAPER_TRAIN_FOLD = 7
PAPER_TRAIN_FRACTION = 0.8


@dataclass
class Series:
    """Concatenated snapshots plus the first index of each stock-day block."""

    lob: np.ndarray     # (N, 40) float32
    labels: np.ndarray  # (N, 5)  int8 in {1, 2, 3}
    starts: np.ndarray  # (n_blocks,) block start indices, starts[0] == 0

    def __len__(self) -> int:
        return self.lob.shape[0]


def _block_starts(lengths: list[int]) -> np.ndarray:
    return np.cumsum([0] + lengths[:-1])


def day_major(days: list[Day]) -> Series:
    """Days in order, each day's stocks in order (the Test_CF_k layout)."""
    slices = [(d, sl) for d in days for sl in d.stock_slices()]
    return Series(
        lob=np.concatenate([d.lob[sl] for d, sl in slices]).astype(np.float32),
        labels=np.concatenate([d.labels[sl] for d, sl in slices]),
        starts=_block_starts([sl.stop - sl.start for _, sl in slices]),
    )


def stock_major_lengths(days: list[Day]) -> list[int]:
    """Block lengths in Train_CF_k order: stock 1 on each day, then stock 2, ..."""
    return [sl.stop - sl.start for s in range(N_STOCKS) for d in days for sl in [d.stock_slices()[s]]]


def paper_train_series(days: list[Day], normalization: str, auction: bool,
                       root: Path | str = DATA_ROOT) -> Series:
    """Train_CF_7 read as-is, with block starts recovered from the per-day stock lengths."""
    raw = read_raw(file_path("Train", PAPER_TRAIN_FOLD, normalization, auction, root))
    lengths = stock_major_lengths(days[:PAPER_TRAIN_FOLD])
    if sum(lengths) != raw.shape[1]:
        raise ValueError(f"Train_CF_{PAPER_TRAIN_FOLD} has {raw.shape[1]} events, "
                         f"days 1-{PAPER_TRAIN_FOLD} have {sum(lengths)}")
    return Series(
        lob=np.ascontiguousarray(raw[LOB_ROWS].T, dtype=np.float32),
        labels=raw[LABEL_ROWS].T.astype(np.int8),
        starts=_block_starts(lengths),
    )


class LOBWindows(Dataset):
    """Windows ending at every valid index of ``series[lo:hi]``.

    Arrays are shared with ``series``; windows are sliced on access, not stored.
    """

    def __init__(self, series: Series, lo: int, hi: int, k: int = 10, window: int = 100,
                 respect_boundaries: bool = False):
        if k not in HORIZON_INDEX:
            raise ValueError(f"k must be one of {sorted(HORIZON_INDEX)}, got {k}")
        cuts = [lo, hi]
        if respect_boundaries:
            cuts = [lo] + [int(s) for s in series.starts if lo < s < hi] + [hi]
        self.ends = np.concatenate([np.arange(a + window - 1, b) for a, b in zip(cuts[:-1], cuts[1:])])
        self.window = window
        self.x = torch.from_numpy(series.lob)
        self.y = torch.from_numpy(series.labels[:, HORIZON_INDEX[k]].astype(np.int64) - 1)

    def __len__(self) -> int:
        return len(self.ends)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = int(self.ends[i])
        return self.x[t - self.window + 1:t + 1].unsqueeze(0), self.y[t]

    @property
    def targets(self) -> np.ndarray:
        """Class of every window, in dataset order."""
        return self.y.numpy()[self.ends]


def build_splits(split: str = "paper", respect_boundaries: bool = False, k: int = 10,
                 window: int = 100, normalization: str = "decpre", auction: bool = False,
                 root: Path | str = DATA_ROOT) -> dict[str, LOBWindows]:
    """Return {"train", "val", "test"} datasets. Defaults reproduce the official setup."""
    days = load_days(normalization=normalization, auction=auction, root=root)
    test = day_major(days[PAPER_TRAIN_FOLD:])
    if split == "paper":
        train = paper_train_series(days, normalization, auction, root)
        cut = int(np.floor(len(train) * PAPER_TRAIN_FRACTION))
        parts = {"train": (train, 0, cut), "val": (train, cut, len(train))}
    elif split == "by_day":
        train, val = day_major(days[:PAPER_TRAIN_FOLD - 1]), day_major(days[PAPER_TRAIN_FOLD - 1:PAPER_TRAIN_FOLD])
        parts = {"train": (train, 0, len(train)), "val": (val, 0, len(val))}
    else:
        raise ValueError(f"split must be 'paper' or 'by_day', got {split!r}")
    parts["test"] = (test, 0, len(test))
    return {name: LOBWindows(s, lo, hi, k, window, respect_boundaries)
            for name, (s, lo, hi) in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise DeepLOB train/val/test windows.")
    parser.add_argument("--split", default="paper", choices=["paper", "by_day"])
    parser.add_argument("--respect-boundaries", action="store_true")
    parser.add_argument("--k", type=int, default=10, choices=sorted(HORIZON_INDEX))
    parser.add_argument("--normalization", default="decpre", choices=["decpre", "zscore", "minmax"])
    args = parser.parse_args()

    splits = build_splits(args.split, args.respect_boundaries, args.k, normalization=args.normalization)
    print(f"split={args.split} respect_boundaries={args.respect_boundaries} k={args.k} "
          f"normalization={args.normalization}")
    print(f"{'':>5} {'windows':>8}  share " + " / ".join(CLASS_NAMES))
    for name, ds in splits.items():
        share = np.bincount(ds.targets, minlength=3) / len(ds)
        print(f"{name:>5} {len(ds):>8}  {share.round(3)}")
    x, y = splits["train"][0]
    print(f"sample: x {tuple(x.shape)} {x.dtype}, y {y.item()} ({CLASS_NAMES[y]})")


if __name__ == "__main__":
    main()

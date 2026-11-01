# lob-prediction

Predicting mid-price movement (up / stationary / down) on the FI-2010 limit
order book benchmark, working towards a replication of
[DeepLOB](https://arxiv.org/abs/1808.03668) (Zhang, Zohren & Roberts, 2019).

## Status

- [x] Data loader that splits FI-2010 into per-day arrays
- [x] Windowed dataset matching the official DeepLOB setup
- [x] Simple baselines
- [x] DeepLOB model and training script
- [x] DeepLOB results at k = 10 compared with the paper
- [x] Training with the settings described in the paper
- [x] DeepLOB results at k = 20 and k = 50

## Setup

Requires Python 3 with `numpy`, `torch` and `scikit-learn`.

```bash
python -m venv .venv
.venv/bin/pip install numpy torch scikit-learn
```

FI-2010 isn't included (about 30 GB). Download it from Fairdata (search for
"FI-2010" or "Benchmark dataset for mid-price forecasting of limit order book
data") and unzip it so the folders sit at `data/BenchmarkDatasets/Auction/`
and `data/BenchmarkDatasets/NoAuction/`.

## Usage

Run from the project root:

```bash
.venv/bin/python src/load_data.py        # per-day summary of the raw data
.venv/bin/python -m src.dataset          # train/val/test window counts and class shares
.venv/bin/python -m src.baselines        # baseline scores for k = 10, 20, 50
.venv/bin/python -m src.train --k 10     # train DeepLOB (about 1 hour on an Apple-silicon GPU)
```

Training writes the best weights and a results file to `runs/<run name>/`.

## Code

| File | What it does |
| --- | --- |
| `src/load_data.py` | Reads the FI-2010 text files into per-day numpy arrays and finds where each of the 5 stocks starts within a day |
| `src/dataset.py` | Cuts the days into 100-snapshot windows, shaped `(1, 100, 40)`, with 0/1/2 labels |
| `src/baselines.py` | Majority-class and logistic regression baselines |
| `src/model.py` | The DeepLOB network |
| `src/train.py` | Training loop, validation and test scoring |

## About the data

Each file is a 149 × N matrix with one column per order book snapshot:

- rows 0–39: 10 price levels × ask price, ask volume, bid price, bid volume
- rows 40–143: 104 hand-crafted features (not used by DeepLOB)
- rows 144–148: labels for 5 horizons, 1 = up, 2 = stationary, 3 = down

The dataset covers 10 trading days of 5 Finnish stocks. `Test_CF_k` holds day
k+1 alone. `Train_CF_k` holds days 1..k, ordered by stock rather than by day,
so it can't be split into days by column counts.

## Evaluation setup

The default ("paper") setup copies the official DeepLOB code: train and
validate on `Train_CF_7` (days 1–7, split 80/20) and test on days 8–10.
`--split by_day --respect-boundaries` gives a cleaner alternative that holds
out day 7 for validation and keeps every window inside one stock on one day.

Most labels are "stationary" (about 70% of the test days at k = 10), so
models are scored with macro-F1 as well as accuracy. Weighted F1 is also
reported for comparison with the paper.

## Baseline results

Test days 8–10, paper setup:

| k | Model | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| 10 | Majority class | 0.707 | 0.276 |
| 10 | Logistic regression, balanced | 0.494 | 0.390 |
| 20 | Majority class | 0.621 | 0.255 |
| 20 | Logistic regression, balanced | 0.488 | 0.391 |
| 50 | Majority class | 0.275 | 0.144 |
| 50 | Logistic regression, balanced | 0.455 | 0.412 |

The logistic regression only sees the latest snapshot, so these numbers are
the floor that DeepLOB has to beat.

## DeepLOB results

Test days 8–10, paper setup. Each result is the mean ± standard deviation of
3 seeds, trained with the settings described in the paper: z-score data, Adam
at lr 0.01 with epsilon 1, batch size 32, and stopping after 20 epochs without
a better validation accuracy.

| k | Accuracy | Paper | Weighted F1 | Paper | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.827 ± 0.001 | 0.845 | 0.816 ± 0.001 | 0.834 | 0.711 ± 0.002 |
| 20 | 0.733 ± 0.001 | 0.749 | 0.715 ± 0.002 | 0.728 | 0.623 ± 0.006 |
| 50 | 0.780 ± 0.005 | 0.805 | 0.779 ± 0.005 | 0.804 | 0.762 ± 0.006 |

```bash
.venv/bin/python -m src.train --k 10 --normalization zscore --lr 0.01 --adam-eps 1 \
  --batch-size 32 --select-on val_accuracy --patience 20 --epochs 200
```

The paper doesn't report macro-F1. It also doesn't say how its F1 is averaged,
but its DeepLOB recall equals its accuracy at every horizon, which points to
weighted averaging, so weighted F1 is the closer comparison.

These settings were chosen after comparing test scores, so the test results may
be slightly optimistic.

### Earlier k = 10 runs

These used the official code's settings: Adam at lr 1e-4, batch size 64 and
early stopping after 5 epochs without a lower validation loss.

| Version | Accuracy | Weighted F1 | Macro-F1 |
| --- | --- | --- | --- |
| DecPre | 0.812 ± 0.012 | 0.793 ± 0.020 | 0.665 ± 0.042 |
| DecPre, class-weighted loss | 0.776 ± 0.004 | 0.785 ± 0.003 | 0.684 ± 0.005 |
| Z-score | 0.818 ± 0.002 | 0.803 ± 0.003 | 0.683 ± 0.005 |

Three seeds is a small sample, and most of the DecPre spread comes from one
weak seed. Macro-F1 doesn't clearly separate the versions. The class-weighted
loss trades precision for recall on the up and down classes.

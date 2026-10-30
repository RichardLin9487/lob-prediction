# lob-prediction

Predicting mid-price movement (up / stationary / down) on the FI-2010 limit
order book benchmark, working towards a replication of
[DeepLOB](https://arxiv.org/abs/1808.03668) (Zhang, Zohren & Roberts, 2019).

## Status

- [x] Data loader that splits FI-2010 into per-day arrays
- [x] Windowed dataset matching the official DeepLOB setup
- [x] Simple baselines
- [x] DeepLOB model and training script
- [ ] DeepLOB results compared with the paper

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
models are scored with macro-F1 as well as accuracy.

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

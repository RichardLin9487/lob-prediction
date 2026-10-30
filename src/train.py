"""Train DeepLOB on FI-2010 and score it on the test days.

Defaults follow the authors' PyTorch code: Adam at lr 1e-4, batch size 64,
50 epochs, keeping the weights with the lowest validation loss.

Each run writes to runs/<name>/: best.pt (weights) and results.json (per-epoch
history plus final test scores).

Run ``python -m src.train --k 10`` from the project root.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from src.dataset import CLASS_NAMES, HORIZON_INDEX, LOBWindows, build_splits
from src.model import DeepLOB

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class DeviceWindows:
    """A LOBWindows dataset moved to ``device``, batched by indexing on the device.

    Gathering whole batches with one index op avoids per-sample Python calls,
    which would otherwise dominate the time of an epoch.
    """

    def __init__(self, ds: LOBWindows, device: torch.device):
        self.x = ds.x.to(device)
        self.y = ds.y.to(device)
        self.ends = torch.from_numpy(ds.ends).to(device)
        self.offsets = torch.arange(-ds.window + 1, 1, device=device)

    def __len__(self) -> int:
        return len(self.ends)

    def batches(self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None,
                max_batches: int | None = None):
        order = (torch.randperm(len(self), generator=generator).to(self.ends.device) if shuffle
                 else torch.arange(len(self), device=self.ends.device))
        for n, i in enumerate(range(0, len(self), batch_size)):
            if max_batches is not None and n >= max_batches:
                return
            ends = self.ends[order[i:i + batch_size]]
            yield self.x[ends[:, None] + self.offsets].unsqueeze(1), self.y[ends]


@torch.no_grad()
def evaluate(model: nn.Module, data: DeviceWindows, max_batches: int | None = None) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, preds, targets = 0.0, [], []
    for xb, yb in data.batches(1024, max_batches=max_batches):
        logits = model(xb)
        total += loss_fn(logits, yb).item()
        preds.append(logits.argmax(1).cpu())
        targets.append(yb.cpu())
    y, p = torch.cat(targets).numpy(), torch.cat(preds).numpy()
    return {"loss": total / len(y), "accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"), "y": y, "pred": p}


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepLOB on FI-2010.")
    parser.add_argument("--k", type=int, default=10, choices=sorted(HORIZON_INDEX))
    parser.add_argument("--split", default="paper", choices=["paper", "by_day"])
    parser.add_argument("--respect-boundaries", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, help="cap batches per epoch and per evaluation (smoke tests)")
    parser.add_argument("--name", help="run directory name (default derived from the settings)")
    args = parser.parse_args()

    name = args.name or f"deeplob_k{args.k}_{args.split}" + ("_bounded" if args.respect_boundaries else "")
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    splits = build_splits(args.split, args.respect_boundaries, args.k)
    data = {split: DeviceWindows(ds, device) for split, ds in splits.items()}
    print(f"{name}: device={device}, windows " + ", ".join(f"{s}={len(d)}" for s, d in data.items()), flush=True)

    model = DeepLOB().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)

    history, best_val = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        train_loss, n = 0.0, 0
        for xb, yb in data["train"].batches(args.batch_size, shuffle=True, generator=generator,
                                            max_batches=args.max_batches):
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            n += len(yb)
        val = evaluate(model, data["val"], args.max_batches)
        improved = val["loss"] < best_val
        if improved:
            best_val = val["loss"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        history.append({"epoch": epoch, "train_loss": train_loss / n, "val_loss": val["loss"],
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                        "seconds": time.time() - t})
        print(f"epoch {epoch:>3}  train loss {train_loss / n:.4f}  val loss {val['loss']:.4f}  "
              f"val acc {val['accuracy']:.3f}  val F1 {val['macro_f1']:.3f}  "
              f"{time.time() - t:.0f}s{'  *best' if improved else ''}", flush=True)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test = evaluate(model, data["test"], args.max_batches)
    print(f"\ntest (best val-loss weights): accuracy {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}")
    print(classification_report(test["y"], test["pred"], labels=[0, 1, 2], target_names=CLASS_NAMES, digits=4))
    cm = confusion_matrix(test["y"], test["pred"], labels=[0, 1, 2])
    print("confusion matrix (rows = true, cols = predicted, order up/stationary/down):\n", cm)

    results = {"args": vars(args), "history": history,
               "test": {"accuracy": test["accuracy"], "macro_f1": test["macro_f1"],
                        "confusion_matrix": cm.tolist()}}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved {run_dir / 'best.pt'} and results.json")


if __name__ == "__main__":
    main()

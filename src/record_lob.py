"""Record live order book snapshots from Binance.US into compressed CSV files.

Binance.US's partial book depth stream pushes the top N levels of each symbol
several times a second. Each message becomes one row, with the levels in the
FI-2010 order so the rest of the pipeline can read them:

    recv_ms, update_id, then for level 1..N: ask_price, ask_size, bid_price, bid_size

recv_ms is this machine's clock when the message arrived (milliseconds since the
Unix epoch), so keep the clock synced. Prices and sizes are stored exactly as
the exchange sends them.

Files go to data/live/binanceus/<SYMBOL>/<YYYY-MM-DD>/<HHMM>.csv.gz, one per
period (UTC). Each is written as .partial and renamed when its period ends, so
finished files are always complete. The connection is reopened after any
error or disconnect (Binance.US closes streams after 24 hours), and every
reconnect is logged so gaps can be found later.

Run ``python -m src.record_lob --symbols btcusdt ethusdt`` from the project
root and stop it with Ctrl-C.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import websocket

LIVE_ROOT = Path(__file__).resolve().parent.parent / "data" / "live" / "binanceus"
STREAM_URL = "wss://stream.binance.us:9443/stream?streams={streams}"
RECV_TIMEOUT_S = 30
MAX_BACKOFF_S = 60
STATS_EVERY_S = 60


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {message}", flush=True)


def header(levels: int) -> list[str]:
    return ["recv_ms", "update_id"] + [f"{name}_{level}" for level in range(1, levels + 1)
                                       for name in ("ask_price", "ask_size", "bid_price", "bid_size")]


class PeriodWriter:
    """Writes one symbol's rows into a new file every ``file_minutes``."""

    def __init__(self, root: Path, symbol: str, levels: int, file_minutes: int):
        self.root, self.symbol, self.levels = root / symbol.upper(), symbol, levels
        self.period_ms = file_minutes * 60_000
        self.period, self.file, self.writer, self.partial, self.path = None, None, None, None, None

    def write(self, recv_ms: int, row: list) -> None:
        period = recv_ms // self.period_ms
        if period != self.period:
            self.close()
            self.open(period)
        self.writer.writerow(row)

    def open(self, period: int) -> None:
        start = datetime.fromtimestamp(period * self.period_ms / 1000, timezone.utc)
        folder = self.root / f"{start:%Y-%m-%d}"
        folder.mkdir(parents=True, exist_ok=True)
        # a restart within the same period gets a suffixed file instead of overwriting
        path, n = folder / f"{start:%H%M}.csv.gz", 1
        while path.exists() or path.with_name(path.name + ".partial").exists():
            path, n = folder / f"{start:%H%M}_{n}.csv.gz", n + 1
        self.period, self.path, self.partial = period, path, path.with_name(path.name + ".partial")
        self.file = gzip.open(self.partial, "wt", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header(self.levels))

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.partial.rename(self.path)
            self.file = None


def to_row(recv_ms: int, data: dict, levels: int) -> list:
    """One CSV row from a partial depth message; missing levels are left empty."""
    asks, bids = data["asks"], data["bids"]
    row = [recv_ms, data["lastUpdateId"]]
    for i in range(levels):
        ask = asks[i] if i < len(asks) else ["", ""]
        bid = bids[i] if i < len(bids) else ["", ""]
        row += [ask[0], ask[1], bid[0], bid[1]]
    return row


def record(symbols: list[str], levels: int, writers: dict[str, PeriodWriter], deadline: float | None) -> None:
    url = STREAM_URL.format(streams="/".join(f"{s}@depth{levels}@100ms" for s in symbols))
    counts = {s: {"rows": 0, "unchanged": 0, "crossed": 0} for s in symbols}
    last_id = {s: None for s in symbols}
    reconnects, backoff, next_stats = 0, 1, time.time() + STATS_EVERY_S

    while deadline is None or time.time() < deadline:
        ws = None
        try:
            ws = websocket.create_connection(url, timeout=RECV_TIMEOUT_S)
            log(f"connected: {', '.join(symbols)}")
            backoff = 1
            while deadline is None or time.time() < deadline:
                raw = ws.recv()
                recv_ms = time.time_ns() // 1_000_000
                msg = json.loads(raw)
                symbol, data = msg["stream"].split("@")[0], msg["data"]
                c = counts[symbol]
                if data["lastUpdateId"] == last_id[symbol]:
                    c["unchanged"] += 1
                    continue
                last_id[symbol] = data["lastUpdateId"]
                if data["asks"] and data["bids"] and float(data["asks"][0][0]) <= float(data["bids"][0][0]):
                    c["crossed"] += 1
                writers[symbol].write(recv_ms, to_row(recv_ms, data, levels))
                c["rows"] += 1
                if time.time() >= next_stats:
                    log(f"reconnects {reconnects}; " + "; ".join(f"{s} {v}" for s, v in counts.items()))
                    next_stats += STATS_EVERY_S
        except (websocket.WebSocketException, OSError, ValueError, KeyError) as e:
            reconnects += 1
            log(f"disconnected ({type(e).__name__}: {e}); retrying in {backoff} s")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
        finally:
            if ws is not None:
                ws.close()
    log(f"done; reconnects {reconnects}; " + "; ".join(f"{s} {v}" for s, v in counts.items()))


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="Record live Binance.US order book snapshots.")
    parser.add_argument("--symbols", nargs="+", default=["btcusdt"], help="lower-case pairs, e.g. btcusdt ethusdt")
    parser.add_argument("--levels", type=int, default=10, choices=[5, 10, 20])
    parser.add_argument("--out", type=Path, default=LIVE_ROOT)
    parser.add_argument("--file-minutes", type=int, default=60, help="start a new file this often")
    parser.add_argument("--minutes", type=float, help="stop after this long (default: run until stopped)")
    args = parser.parse_args()

    symbols = [s.lower() for s in args.symbols]
    writers = {s: PeriodWriter(args.out, s, args.levels, args.file_minutes) for s in symbols}
    deadline = None if args.minutes is None else time.time() + args.minutes * 60
    # SIGINT too: a shell starting this in the background makes Python ignore Ctrl-C
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    log(f"recording {', '.join(symbols)} at {args.levels} levels into {args.out}")
    try:
        record(symbols, args.levels, writers, deadline)
    except KeyboardInterrupt:
        log("stopped")
    finally:
        for w in writers.values():
            w.close()


if __name__ == "__main__":
    main()

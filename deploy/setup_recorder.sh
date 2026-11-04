#!/usr/bin/env bash
# Set up src/record_lob.py on a fresh Ubuntu or Debian VM and run it as a service.
#
# Copy the project's src/record_lob.py and this script to the VM first (see the
# README), then run it there as a normal user that has sudo:
#
#     bash ~/lob-prediction/deploy/setup_recorder.sh
#
# Set SYMBOLS to record other pairs, e.g. SYMBOLS="btcusdt ethusdt" bash ...
# Re-running it is safe: it reinstalls the package and restarts the service.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYMBOLS="${SYMBOLS:-btcusdt btcusd ethusdt}"
SERVICE=record-lob

[ -f "$APP_DIR/src/record_lob.py" ] || { echo "missing $APP_DIR/src/record_lob.py" >&2; exit 1; }

echo "== installing Python"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q websocket-client==1.9.2

echo "== checking the clock is synced (recv_ms uses it)"
timedatectl show -p NTPSynchronized --value | grep -qx yes \
  || echo "warning: clock is not NTP-synced yet; check 'timedatectl status'" >&2

echo "== test recording for 20 seconds"
test_dir="$(mktemp -d)"
(cd "$APP_DIR" && .venv/bin/python -u -m src.record_lob --symbols btcusdt --minutes 0.33 --out "$test_dir")
rows=$(find "$test_dir" -name '*.csv.gz' -exec gzip -dc {} + | tail -n +2 | wc -l)
rm -rf "$test_dir"
if [ "$rows" -lt 10 ]; then
  echo "test recorded only $rows rows; Binance.US may be blocked from this VM's location" >&2
  exit 1
fi
echo "test recorded $rows rows"

echo "== installing the $SERVICE service"
sudo tee /etc/systemd/system/$SERVICE.service > /dev/null <<EOF
[Unit]
Description=Record Binance.US order book snapshots
Wants=network-online.target
After=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/.venv/bin/python -u -m src.record_lob --symbols $SYMBOLS
Restart=always
RestartSec=10
StandardOutput=append:$APP_DIR/data/live/record_lob.log
StandardError=inherit

[Install]
WantedBy=multi-user.target
EOF
mkdir -p "$APP_DIR/data/live"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
sudo systemctl restart "$SERVICE"
sleep 5
systemctl --no-pager status "$SERVICE" | head -n 5
echo "== done. Log: $APP_DIR/data/live/record_lob.log"

# Palm Guard — Edge Hardware (Raspberry Pi)

Install + run guide for the on-device agent (`packages/edge`) on a **Raspberry Pi
Zero 2 W**. The agent captures trunk audio, classifies it on-device with TFLite,
and store-and-forwards detections to the API — surviving offline periods.

## Bill of materials

| Item | Notes |
|---|---|
| Raspberry Pi Zero 2 W | Quad-core; enough for TFLite inference at 8 kHz |
| microSD (16 GB+) | Raspberry Pi OS Lite (64-bit) |
| Contact / piezo sensor | Structure-borne pickup pressed to the trunk |
| USB sound card / ADC | If the sensor needs amplification/ADC (e.g. a USB audio dongle or an I²S ADC) |
| Power | Solar + battery for field deployment |

## OS + system packages

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libportaudio2 libatlas-base-dev
```

`libportaudio2` is needed by `sounddevice` (capture); `libatlas-base-dev` speeds
up NumPy.

## Python environment

```bash
git clone <repo> palm-guard && cd palm-guard/packages/edge
python3 -m venv .venv && source .venv/bin/activate

# Edge agent + the shared ML feature/inference code.
pip install -r requirements.txt
pip install -r ../ml/requirements.txt

# Pi-only runtime deps (commented out in requirements.txt):
pip install sounddevice
pip install tflite-runtime   # quantised inference without full TensorFlow
```

> The edge reuses `packages/ml` for the exact training feature path (added to
> `sys.path` automatically). Do not re-implement DSP on the device.

## Model artifact

Copy the quantised model produced by `make export` on a workstation:

```bash
scp packages/ml/artifacts/palmguard.tflite pi@<host>:~/palm-guard/packages/ml/artifacts/
```

Until a TFLite model is present the agent falls back to the RandomForest baseline
(`baseline.joblib`) so it is never unable to classify.

## Verify the audio input

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Confirm your contact-sensor input device, and that it captures at **8 kHz** mono
(the fixed `palmguard_ml.config.SAMPLE_RATE`). Set the default input device if
needed.

## Configure

Set environment (e.g. in `/etc/palmguard.env` or the systemd unit below):

```bash
EDGE_API_URL=https://your-api.example.com
EDGE_DEVICE_ID=pi-zero-grove-a-01
EDGE_TREE_ID=tree-001
EDGE_MODEL_PATH=/home/pi/palm-guard/packages/ml/artifacts/palmguard.tflite
EDGE_INTERVAL_S=3600        # hourly capture loop
EDGE_QUEUE_PATH=/var/lib/palmguard/queue.jsonl
```

## Run

```bash
# One-shot sanity check against a recorded clip (no hardware needed):
python run.py --sim ../../data/raw/infested/<...>.wav --once

# One live capture + flush:
python run.py --once

# Continuous hourly loop:
python run.py
```

## Run as a service (systemd)

`/etc/systemd/system/palmguard-edge.service`:

```ini
[Unit]
Description=Palm Guard edge agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
EnvironmentFile=/etc/palmguard.env
WorkingDirectory=/home/pi/palm-guard/packages/edge
ExecStart=/home/pi/palm-guard/packages/edge/.venv/bin/python run.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo mkdir -p /var/lib/palmguard
sudo systemctl enable --now palmguard-edge
journalctl -u palmguard-edge -f
```

## Offline behaviour

The agent appends every detection to `queue.jsonl` **before** any network
attempt, and only removes items after a confirmed POST. If the API is
unreachable, detections accumulate and flush automatically on the next reachable
cycle. A crash mid-flush at worst redelivers one detection (the API is
append-only, so duplicates are harmless). This is verified by
`packages/edge/tests/test_uploader.py`.

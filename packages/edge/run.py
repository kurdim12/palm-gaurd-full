"""Palm Guard edge agent entrypoint.

Hourly loop on a Raspberry Pi: capture a clip → classify on-device (TFLite) →
enqueue the detection → flush the queue to the API when reachable. Survives
offline periods: detections persist in ``queue.jsonl`` and flush once connectivity
returns.

Usage:
    python run.py                      # live capture loop (needs audio hardware)
    python run.py --once               # one live capture + flush, then exit
    python run.py --sim clip.wav --once  # classify a WAV instead of recording
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from palmguard_edge import capture, infer
from palmguard_edge.config import EdgeConfig, load_config
from palmguard_edge.uploader import UploadQueue, http_sender


def run_once(cfg: EdgeConfig, sim_path: str | None = None) -> dict:
    """Capture (or load) one clip, classify, enqueue, and try to flush.

    Returns the detection payload that was enqueued.
    """
    engine = infer.load_engine(cfg.model_path, threshold=cfg.threshold)
    signal = capture.load_clip(sim_path) if sim_path else capture.record()
    prediction = infer.classify(engine, signal)

    detection = {
        "device_id": cfg.device_id,
        "tree_id": cfg.tree_id,
        "label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model_version": type(engine).__name__,
    }

    queue = UploadQueue(cfg.queue_path)
    queue.enqueue(detection)  # durable first
    delivered = queue.flush(http_sender(cfg.api_url))
    print(
        f"[{detection['captured_at']}] {prediction.label} "
        f"(conf={prediction.confidence:.2f}) | flushed {delivered}, "
        f"{len(queue.pending())} pending"
    )
    return detection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Palm Guard edge agent")
    parser.add_argument("--sim", metavar="WAV", help="classify a WAV instead of recording")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.once or args.sim:
        run_once(cfg, sim_path=args.sim)
        return 0

    while True:  # pragma: no cover - long-running loop
        try:
            run_once(cfg)
        except Exception as exc:  # never let one bad cycle kill the agent
            print(f"cycle error: {exc}")
        time.sleep(cfg.capture_interval_s)


if __name__ == "__main__":
    raise SystemExit(main())

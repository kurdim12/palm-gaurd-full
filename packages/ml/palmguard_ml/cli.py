"""Command-line entrypoints used by the Makefile.

Usage:
    python -m palmguard_ml.cli {data,baseline,train,eval,export}

``data`` and ``baseline`` are light (NumPy/scikit-learn only). ``train``/``eval``/
``export`` pull in TensorFlow lazily.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import config


def _cmd_data(args: argparse.Namespace) -> int:
    any_real = bool(
        config.TREEVIBES_LOCAL
        or config.TREEVIBES_KAGGLE
        or config.TREEVIBES_URL
        or config.ESC50_URL
    )
    path = None
    if any_real:
        from .ingest import build_combined

        print("Building manifest from configured real sources (graceful fallback)…")
        path = build_combined()
    if path is None:
        from .ingest import synthetic

        if any_real:
            print("No real source produced data — falling back to synthetic.")
        else:
            print("No dataset URLs set — generating synthetic RPW-like dataset.")
        path = synthetic.build_manifest()

    from .manifest import read_manifest, summary, validate

    rows = read_manifest(path)
    validate(rows)
    print(f"Manifest written: {path}")
    print(json.dumps(summary(rows), indent=2))
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    from .baseline import train_and_eval
    from .evaluate import format_report

    print("Training RandomForest baseline on the site-split...")
    _, metrics = train_and_eval()
    print("\nBaseline metrics (held-out SITE split):")
    print(format_report(metrics))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .evaluate import format_report
    from .train import train

    _, metrics = train(epochs=args.epochs, backbone=args.backbone)
    print("\nCNN metrics (held-out SITE split):")
    print(format_report(metrics))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import format_report
    from .train import evaluate_saved

    metrics = evaluate_saved()
    print("CNN metrics (held-out SITE split):")
    print(format_report(metrics))
    if not metrics.meets_target():
        print(
            f"\n⚠ infested recall {metrics.infested_recall:.3f} < target "
            f"{config.TARGET_INFESTED_RECALL}. Consider minority-class augmentation, "
            "threshold tuning, or a different backbone — do NOT change config constants."
        )
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from .export import export_and_verify

    print("Exporting quantised TFLite + parity check...")
    report = export_and_verify()
    print(json.dumps(report, indent=2))
    print("TFLite parity OK." if report["passed"] else "TFLite parity FAILED.")
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="palmguard_ml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("data", help="generate/ingest dataset + manifest").set_defaults(func=_cmd_data)
    sub.add_parser("baseline", help="train+eval RandomForest baseline").set_defaults(
        func=_cmd_baseline
    )

    p_train = sub.add_parser("train", help="train the CNN")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--backbone", default=None)
    p_train.set_defaults(func=_cmd_train)

    sub.add_parser("eval", help="evaluate the saved CNN").set_defaults(func=_cmd_eval)
    sub.add_parser("export", help="export TFLite + parity check").set_defaults(func=_cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

"""Threshold-aware, recall-first metrics.

Per CLAUDE.md #2, the headline metrics are **infested recall** and **PR-AUC** on a
held-out *site* split — never raw accuracy alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from . import config


@dataclass
class Metrics:
    """Evaluation summary at a chosen decision threshold."""

    threshold: float
    infested_recall: float
    infested_precision: float
    infested_f1: float
    pr_auc: float
    accuracy: float
    confusion: list[list[int]]  # rows=true [clean, infested], cols=pred
    n: int

    def to_dict(self) -> dict:
        return asdict(self)

    def meets_target(self) -> bool:
        return self.infested_recall >= config.TARGET_INFESTED_RECALL


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
) -> Metrics:
    """Compute recall-first metrics for positive class = infested (1)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr_auc = float("nan")

    return Metrics(
        threshold=threshold,
        infested_recall=float(rec[1]),
        infested_precision=float(prec[1]),
        infested_f1=float(f1[1]),
        pr_auc=pr_auc,
        accuracy=float((y_pred == y_true).mean()),
        confusion=cm.tolist(),
        n=int(len(y_true)),
    )


def best_threshold_for_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_recall: float = config.TARGET_INFESTED_RECALL,
) -> float:
    """Highest threshold that still achieves ``target_recall`` on infested.

    Recall-first: among thresholds meeting the recall target we pick the one with
    best precision (the highest such threshold), falling back to the threshold
    that maximises recall if the target is unreachable.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    candidates = np.unique(np.concatenate([[0.0, 1.0], y_score]))
    best_t, best_recall = config.DEFAULT_THRESHOLD, -1.0
    chosen: float | None = None
    for t in sorted(candidates):
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if recall >= target_recall:
            chosen = t  # keep raising t while target holds → better precision
        if recall > best_recall:
            best_recall, best_t = recall, t
    return float(chosen if chosen is not None else best_t)


def format_report(metrics: Metrics) -> str:
    """Pretty multi-line report for the CLI."""
    cm = metrics.confusion
    lines = [
        f"  threshold          : {metrics.threshold:.3f}",
        f"  infested recall    : {metrics.infested_recall:.3f}"
        f"  {'✓' if metrics.meets_target() else '✗ (<%.2f)' % config.TARGET_INFESTED_RECALL}",
        f"  infested precision : {metrics.infested_precision:.3f}",
        f"  infested f1        : {metrics.infested_f1:.3f}",
        f"  PR-AUC             : {metrics.pr_auc:.3f}",
        f"  accuracy           : {metrics.accuracy:.3f}",
        f"  n (test clips)     : {metrics.n}",
        "  confusion (rows=true clean/infested, cols=pred):",
        f"      clean    -> [{cm[0][0]:>4}, {cm[0][1]:>4}]",
        f"      infested -> [{cm[1][0]:>4}, {cm[1][1]:>4}]",
    ]
    return "\n".join(lines)

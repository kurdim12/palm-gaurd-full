"""Tree status debounce engine.

A single noisy infested hit must NOT flip a tree to ``infested`` — a missed tree
is a dead tree, but a farmer who gets false alarms stops trusting the system.
A tree is confirmed ``infested`` only after ``streak`` *consecutive* infested
detections at/above ``confidence_threshold``. Below-threshold infested hits move a
clean tree to ``suspect`` (visible, but no alert). A clean detection resets the
streak.

The engine is pure: it takes the current tree + a detection and returns the next
tree state, so it's trivially testable and DB-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Detection, Label, Tree, TreeStatus


@dataclass(frozen=True)
class StatusDecision:
    """Result of applying one detection to a tree."""

    tree: Tree
    changed: bool
    became_infested: bool


class StatusEngine:
    """Debounced status transitions."""

    def __init__(self, streak: int = 3, confidence_threshold: float = 0.6) -> None:
        if streak < 1:
            raise ValueError("streak must be >= 1")
        self.streak = streak
        self.confidence_threshold = confidence_threshold

    def apply(self, tree: Tree, detection: Detection) -> StatusDecision:
        """Return the tree's next state after observing ``detection``."""
        prev_status = tree.status
        prev_streak = tree.infested_streak

        confident_infested = (
            detection.label == Label.INFESTED
            and detection.confidence >= self.confidence_threshold
        )

        if confident_infested:
            new_streak = prev_streak + 1
        elif detection.label == Label.CLEAN:
            new_streak = 0
        else:
            # Infested but low-confidence: don't advance, don't fully reset.
            new_streak = prev_streak

        # A treated tree stays treated until a confirmed re-infestation.
        if new_streak >= self.streak:
            new_status = TreeStatus.INFESTED
        elif prev_status == TreeStatus.INFESTED:
            # Confirmed infestation is sticky: only treatment clears it.
            new_status = TreeStatus.INFESTED
        elif prev_status == TreeStatus.TREATED:
            new_status = TreeStatus.TREATED
        elif detection.label == Label.CLEAN and new_streak == 0:
            new_status = TreeStatus.CLEAN
        elif detection.label == Label.INFESTED:
            # Some infested evidence, not yet confirmed.
            new_status = TreeStatus.SUSPECT
        else:
            new_status = prev_status

        updated = tree.model_copy(
            update={
                "status": new_status,
                "infested_streak": new_streak,
                "last_detection_at": detection.captured_at,
                "updated_at": detection.received_at,
            }
        )
        became_infested = (
            new_status == TreeStatus.INFESTED and prev_status != TreeStatus.INFESTED
        )
        return StatusDecision(
            tree=updated,
            changed=(new_status != prev_status or new_streak != prev_streak),
            became_infested=became_infested,
        )

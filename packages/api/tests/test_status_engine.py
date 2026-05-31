"""Status-engine debounce tests (CLAUDE.md definition of done for the backend).

A single noisy infested hit must NOT flip a tree to infested; N consecutive
confident infested hits must.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.models import Detection, Label, Tree, TreeStatus
from app.status_engine import StatusEngine


def _detection(label: Label, confidence: float, tree_id: str = "t1") -> Detection:
    now = datetime.now(timezone.utc)
    return Detection(
        device_id="dev",
        tree_id=tree_id,
        label=label,
        confidence=confidence,
        captured_at=now,
        received_at=now,
    )


def _tree() -> Tree:
    return Tree(id="t1", lat=0.0, lon=0.0)


def test_single_confident_infested_does_not_flip():
    engine = StatusEngine(streak=3, confidence_threshold=0.6)
    decision = engine.apply(_tree(), _detection(Label.INFESTED, 0.95))
    assert decision.tree.status == TreeStatus.SUSPECT
    assert not decision.became_infested
    assert decision.tree.infested_streak == 1


def test_three_consecutive_confident_infested_flips():
    engine = StatusEngine(streak=3, confidence_threshold=0.6)
    tree = _tree()
    became = []
    for _ in range(3):
        decision = engine.apply(tree, _detection(Label.INFESTED, 0.9))
        tree = decision.tree
        became.append(decision.became_infested)
    assert tree.status == TreeStatus.INFESTED
    assert became == [False, False, True]  # flips exactly on the third


def test_low_confidence_infested_never_advances_streak():
    engine = StatusEngine(streak=3, confidence_threshold=0.6)
    tree = _tree()
    for _ in range(10):
        decision = engine.apply(tree, _detection(Label.INFESTED, 0.4))
        tree = decision.tree
    assert tree.status == TreeStatus.SUSPECT
    assert tree.infested_streak == 0  # below threshold => no progress to infested


def test_clean_detection_resets_streak():
    engine = StatusEngine(streak=3, confidence_threshold=0.6)
    tree = _tree()
    tree = engine.apply(tree, _detection(Label.INFESTED, 0.9)).tree
    tree = engine.apply(tree, _detection(Label.INFESTED, 0.9)).tree
    assert tree.infested_streak == 2
    tree = engine.apply(tree, _detection(Label.CLEAN, 0.9)).tree
    assert tree.infested_streak == 0
    assert tree.status == TreeStatus.CLEAN


def test_intermittent_noise_does_not_confirm():
    # Alternating confident-infested and clean never reaches the streak.
    engine = StatusEngine(streak=3, confidence_threshold=0.6)
    tree = _tree()
    for label in (Label.INFESTED, Label.CLEAN, Label.INFESTED, Label.CLEAN, Label.INFESTED):
        tree = engine.apply(tree, _detection(label, 0.9)).tree
    assert tree.status != TreeStatus.INFESTED

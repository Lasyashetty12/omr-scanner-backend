import numpy as np

from ml_omr.hybrid_reader import _decide_question, _postprocess_known_failure_classes


def _option(darkness, disk, core=0.65, ml=0.10):
    return {
        "metrics": {
            "center_darkness": darkness,
            "disk_dark_ratio": disk,
            "core_dark_ratio": core,
        },
        "ml_filled_probability": ml,
        "micro_core_darkness": 100.0,
        "crop_center": [100, 100],
    }


def test_unique_visual_winner_does_not_become_uncertain_for_low_ml_confidence():
    # A readable, uniquely dominant mark below the reader's stricter
    # strong/medium/faint thresholds must not be classified UNCERTAIN just
    # because the model confidence is low.
    options = {
        "A": _option(95.0, 0.50, ml=0.20),
        "B": _option(65.0, 0.38),
        "C": _option(61.0, 0.30),
        "D": _option(58.0, 0.27),
    }
    decision = _decide_question(options, {})
    assert decision["status"] in ("ambiguous", "blank")

    result = _postprocess_known_failure_classes(
        1,
        options,
        decision,
        np.full((220, 220), 255, dtype=np.uint8),
    )

    assert result["status"] == "answered"
    assert result["answer"] == "A"
    assert result["unique_visual_winner_rescue"]


def test_existing_multiple_decision_is_unchanged_by_uncertain_rescue():
    options = {
        "A": _option(110.0, 0.95, 0.95),
        "B": _option(105.0, 0.90, 0.92),
        "C": _option(48.0, 0.22),
        "D": _option(45.0, 0.20),
    }
    decision = _decide_question(options, {})

    assert decision["status"] == "multiple"
    result = _postprocess_known_failure_classes(
        1,
        options,
        decision,
        np.full((220, 220), 255, dtype=np.uint8),
    )
    assert result["status"] == "multiple"
    assert result["answer"] == "MULTIPLE"


def test_outline_heavy_blank_is_not_rescued_by_a_misleading_classifier_score():
    # Q17-style case: a printed outline gives C a misleading classifier score,
    # but no option has a filled disk.  This is a BLANK response.
    options = {
        "A": _option(81.19, 0.3148, 0.7732, 0.0034),
        "B": _option(66.91, 0.4328, 0.7423, 0.4172),
        "C": _option(67.47, 0.5016, 0.6907, 0.9823),
        "D": _option(50.10, 0.3574, 0.5670, 0.3523),
    }
    decision = _decide_question(options, {})
    assert decision["status"] in ("ambiguous", "blank")

    result = _postprocess_known_failure_classes(
        17,
        options,
        decision,
        np.full((220, 220), 255, dtype=np.uint8),
    )

    assert result["status"] == "blank"
    assert result["answer"] is None
    assert result["outline_blank_rescue"]


def test_outline_heavy_q27_stays_blank_despite_misleading_classifier_scores():
    # Q27 has dark printed artifacts.  No bubble has filled-disk coverage, so
    # it is Blank even though C receives an overconfident classifier score.
    options = {
        "A": _option(90.15, 0.3836, 0.8247, 0.0051),
        "B": _option(70.55, 0.4262, 0.8144, 0.8049),
        "C": _option(72.57, 0.4557, 0.7010, 0.9994),
        "D": _option(65.38, 0.3738, 0.7216, 0.7180),
    }
    decision = _decide_question(options, {})
    assert decision["status"] in ("ambiguous", "blank")

    result = _postprocess_known_failure_classes(
        27,
        options,
        decision,
        np.full((220, 220), 255, dtype=np.uint8),
    )

    assert result["status"] == "blank"
    assert result["answer"] is None
    assert result["outline_blank_rescue"]


def test_strong_q180_mark_remains_answered():
    options = {
        "A": _option(118.45, 1.0, 1.0, 1.0),
        "B": _option(39.60, 0.3475, 0.4845, 0.8655),
        "C": _option(22.16, 0.2295, 0.4536, 0.0146),
        "D": _option(44.51, 0.4164, 0.5258, 0.9983),
    }
    result = _decide_question(options, {})

    assert result["status"] == "answered"
    assert result["answer"] == "A"

from __future__ import annotations

import csv
import json
from pathlib import Path

from functools import lru_cache

import cv2
import numpy as np

from ml_omr.inference import classify_batch


DEFAULT_CROP_RADIUS = 16

# ------------------------------------------------------------
# Tuned decision thresholds
# ------------------------------------------------------------

# Sheet-level adaptive threshold is still learned from blank bubbles.
MIN_FILLED_DARKNESS = 42.0
MIN_CORE_DARK_RATIO = 0.15

# Relative rescue: lightly filled bubbles can still be accepted when
# they are clearly darker than the other three options.
RELATIVE_RESCUE_MIN_GAP = 12.0
RELATIVE_RESCUE_ML = 0.65

# A true blank should have BOTH weak absolute evidence AND weak relative
# separation from the second-darkest bubble.
BLANK_ABSOLUTE_MARGIN = 0.84
BLANK_MAX_TOP_GAP = 9.0

# Multiple validation
MULTIPLE_MIN_DELTA = 18.0
MULTIPLE_MIN_CORE_DARK_RATIO = 0.19


def crop_bubble(
    gray,
    x,
    y,
    radius=DEFAULT_CROP_RADIUS,
):
    h, w = gray.shape[:2]

    x1 = max(0, int(round(x - radius)))
    y1 = max(0, int(round(y - radius)))
    x2 = min(w, int(round(x + radius + 1)))
    y2 = min(h, int(round(y + radius + 1)))

    return gray[y1:y2, x1:x2]



def refine_bubble_center(
    gray,
    x,
    y,
    search_radius_x=3,
    search_radius_y=1,
    ring_radius=10,
):
    """
    Tiny final refinement after column calibration.

    Horizontal movement is allowed up to +/-3 px.
    Vertical movement is limited to +/-1 px so the detector cannot
    drift downward into the next row or nearby printed lines.
    """

    h, w = gray.shape[:2]

    base_x = int(round(x))
    base_y = int(round(y))

    best_x = base_x
    best_y = base_y
    best_score = -1e9

    for dy in range(
        -search_radius_y,
        search_radius_y + 1,
    ):
        for dx in range(
            -search_radius_x,
            search_radius_x + 1,
        ):
            cx = base_x + dx
            cy = base_y + dy

            x1 = max(0, cx - ring_radius)
            y1 = max(0, cy - ring_radius)
            x2 = min(w, cx + ring_radius + 1)
            y2 = min(h, cy + ring_radius + 1)

            patch = gray[
                y1:y2,
                x1:x2,
            ]

            if (
                patch.shape[0] < 19
                or patch.shape[1] < 19
            ):
                continue

            ph, pw = patch.shape[:2]

            yy, xx = np.ogrid[
                :ph,
                :pw,
            ]

            pcx = (pw - 1) / 2.0
            pcy = (ph - 1) / 2.0

            rr = np.sqrt(
                (xx - pcx) ** 2
                +
                (yy - pcy) ** 2
            )

            ring_mask = (
                (rr >= ring_radius * 0.55)
                &
                (rr <= ring_radius * 0.88)
            )

            outside_mask = (
                (rr >= ring_radius * 0.92)
                &
                (rr <= ring_radius * 1.00)
            )

            if not np.any(
                ring_mask
            ):
                continue

            ring_mean = float(
                np.mean(
                    patch[
                        ring_mask
                    ]
                )
            )

            outside_mean = (
                float(
                    np.mean(
                        patch[
                            outside_mask
                        ]
                    )
                )
                if np.any(
                    outside_mask
                )
                else 230.0
            )

            score = (
                outside_mean
                -
                ring_mean
            )

            # Prefer staying near the calibrated coordinate.
            score -= abs(dx) * 0.35
            score -= abs(dy) * 1.50

            if score > best_score:
                best_score = score
                best_x = cx
                best_y = cy

    return (
        best_x,
        best_y,
    )



@lru_cache(maxsize=256)
def _circle_mask(size, radius):
    center = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]

    return (
        (xx - center) ** 2
        +
        (yy - center) ** 2
        <= radius ** 2
    )


def _bubble_metrics(crop):
    if crop is None or crop.size == 0:
        return {
            "core_mean": 255.0,
            "paper_mean": 255.0,
            "center_darkness": 0.0,
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
        }

    if crop.ndim == 3:
        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

    crop = crop.astype(np.uint8)

    normalized = crop

    h, w = normalized.shape[:2]
    size = min(h, w)

    if size < 9:
        mean_value = float(np.mean(normalized))
        return {
            "core_mean": mean_value,
            "paper_mean": mean_value,
            "center_darkness": 0.0,
            "core_dark_ratio": 0.0,
            "disk_dark_ratio": 0.0,
        }

    y0 = (h - size) // 2
    x0 = (w - size) // 2

    square = normalized[
        y0:y0 + size,
        x0:x0 + size,
    ]

    core_radius = max(
        3.0,
        size * 0.17,
    )

    disk_radius = max(
        5.0,
        size * 0.30,
    )

    background_inner = size * 0.38
    background_outer = size * 0.49

    core_mask = _circle_mask(
        size,
        core_radius,
    )

    disk_mask = _circle_mask(
        size,
        disk_radius,
    )

    outer_mask = _circle_mask(
        size,
        background_outer,
    )

    inner_bg_mask = _circle_mask(
        size,
        background_inner,
    )

    background_mask = np.logical_and(
        outer_mask,
        np.logical_not(inner_bg_mask),
    )

    core_pixels = square[core_mask]
    disk_pixels = square[disk_mask]
    background_pixels = square[
        background_mask
    ]

    if core_pixels.size == 0:
        core_pixels = square.reshape(-1)

    if disk_pixels.size == 0:
        disk_pixels = square.reshape(-1)

    if background_pixels.size == 0:
        background_pixels = square.reshape(-1)

    paper_mean = float(
        np.percentile(
            background_pixels,
            70,
        )
    )

    core_mean = float(
        np.mean(
            core_pixels
        )
    )

    center_darkness = max(
        0.0,
        paper_mean - core_mean,
    )

    dark_threshold = int(
        np.clip(
            paper_mean - 42.0,
            80,
            165,
        )
    )

    core_dark_ratio = float(
        np.mean(
            core_pixels < dark_threshold
        )
    )

    disk_dark_ratio = float(
        np.mean(
            disk_pixels < dark_threshold
        )
    )

    return {
        "core_mean":
            round(
                core_mean,
                2,
            ),

        "paper_mean":
            round(
                paper_mean,
                2,
            ),

        "center_darkness":
            round(
                center_darkness,
                2,
            ),

        "core_dark_ratio":
            round(
                core_dark_ratio,
                4,
            ),

        "disk_dark_ratio":
            round(
                disk_dark_ratio,
                4,
            ),
    }


def _ml_probability(
    prediction,
    label,
):
    probabilities = prediction.get(
        "probabilities",
        {},
    )

    if (
        isinstance(probabilities, dict)
        and label in probabilities
    ):
        return float(
            probabilities[label]
        )

    predicted_label = str(
        prediction.get(
            "label",
            ""
        )
    ).lower()

    confidence = float(
        prediction.get(
            "confidence",
            0.0,
        )
    )

    if predicted_label == label:
        return confidence

    return 0.0


def _median_absolute_deviation(values):
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:
        return 0.0

    median = float(
        np.median(values)
    )

    return float(
        np.median(
            np.abs(
                values - median
            )
        )
    )


def _estimate_blank_distribution(
    all_question_data,
):
    blank_darkness_samples = []
    blank_core_ratio_samples = []

    for option_data in (
        all_question_data.values()
    ):
        ranked = sorted(
            option_data.values(),
            key=lambda item:
                float(
                    item[
                        "metrics"
                    ][
                        "center_darkness"
                    ]
                ),
        )

        for item in ranked[:2]:
            blank_darkness_samples.append(
                float(
                    item[
                        "metrics"
                    ][
                        "center_darkness"
                    ]
                )
            )

            blank_core_ratio_samples.append(
                float(
                    item[
                        "metrics"
                    ][
                        "core_dark_ratio"
                    ]
                )
            )

    blank_median = float(
        np.median(
            blank_darkness_samples
        )
    )

    blank_mad = (
        _median_absolute_deviation(
            blank_darkness_samples
        )
    )

    blank_ratio_median = float(
        np.median(
            blank_core_ratio_samples
        )
    )

    filled_darkness_threshold = max(
        MIN_FILLED_DARKNESS,
        blank_median
        +
        max(
            20.0,
            5.0 * blank_mad,
        ),
    )

    filled_core_ratio_threshold = max(
        MIN_CORE_DARK_RATIO,
        blank_ratio_median
        +
        0.10,
    )

    filled_darkness_threshold = float(
        np.clip(
            filled_darkness_threshold,
            46.0,
            110.0,
        )
    )

    filled_core_ratio_threshold = float(
        np.clip(
            filled_core_ratio_threshold,
            0.18,
            0.56,
        )
    )

    return {
        "blank_darkness_median":
            round(
                blank_median,
                3,
            ),

        "blank_darkness_mad":
            round(
                blank_mad,
                3,
            ),

        "blank_core_ratio_median":
            round(
                blank_ratio_median,
                4,
            ),

        "filled_darkness_threshold":
            round(
                filled_darkness_threshold,
                3,
            ),

        "filled_core_ratio_threshold":
            round(
                filled_core_ratio_threshold,
                4,
            ),
    }


def _decide_question(
    option_data,
    sheet_thresholds,
):
    """
    Disk-first mobile-photo decision engine.

    Why:
    - Printed empty bubbles can have high core_dark_ratio because the ring
      and digits are dark.
    - A genuinely filled bubble usually darkens a much larger fraction of
      the bubble disk.
    - Therefore disk_dark_ratio is the primary evidence for SINGLE/MULTIPLE.
    - Row-relative darkness is used only as a conservative faint-mark rescue.
    - ML is supporting evidence only.
    """

    del sheet_thresholds

    ranked = sorted(
        option_data.items(),
        key=lambda item: (
            float(
                item[1]["metrics"]["disk_dark_ratio"]
            ),
            float(
                item[1]["metrics"]["center_darkness"]
            ),
        ),
        reverse=True,
    )

    best_option, best_info = ranked[0]
    second_option, second_info = ranked[1]

    def metrics_for(info):
        metrics = info["metrics"]
        return {
            "darkness": float(
                metrics["center_darkness"]
            ),
            "core": float(
                metrics["core_dark_ratio"]
            ),
            "disk": float(
                metrics["disk_dark_ratio"]
            ),
            "ml": float(
                info.get(
                    "ml_filled_probability",
                    0.0,
                )
            ),
        }

    best = metrics_for(best_info)
    second = metrics_for(second_info)

    darkness_values = sorted(
        [
            float(
                info["metrics"]["center_darkness"]
            )
            for _, info in option_data.items()
        ]
    )

    disk_values = sorted(
        [
            float(
                info["metrics"]["disk_dark_ratio"]
            )
            for _, info in option_data.items()
        ]
    )

    question_blank_baseline = float(
        np.median(
            darkness_values[:2]
        )
    )

    disk_blank_baseline = float(
        np.median(
            disk_values[:2]
        )
    )

    best_delta = (
        best["darkness"]
        -
        question_blank_baseline
    )

    top_gap = (
        best["darkness"]
        -
        second["darkness"]
    )

    disk_gap = (
        best["disk"]
        -
        second["disk"]
    )

    # --------------------------------------------------------
    # Blank row check (MUST RUN FIRST before strong/medium fill)
    # --------------------------------------------------------
    best_ml_blank = float(best_info.get("ml_blank_probability", 0.0))
    best_ml_filled = float(best_info.get("ml_filled_probability", best["ml"]))
    best_core_mean = float(best_info.get("metrics", {}).get("core_mean", 255.0))

    blank_like = (
        (best["disk"] < 0.60 and best["darkness"] < 85.0 and top_gap < 25.0)
        or
        (best_ml_blank >= 0.55 and best_ml_filled < 0.45)
        or
        (best["darkness"] < 45.0 and best["disk"] < 0.45)
    )

    if blank_like:
        return {
            "answer":
                None,

            "status":
                "blank",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),
        }

    # The report shows genuine full marks are typically near disk=1.0,
    # while many false "second fills" are around 0.40-0.65.
    #
    # Keep multiple very strict.
    def is_strong_fill(info):
        m = metrics_for(info)
        core_m = float(info.get("metrics", {}).get("core_mean", 0.0))
        ml_blank = float(info.get("ml_blank_probability", 0.0))
        ml_ambiguous = float(info.get("ml_ambiguous_probability", 0.0))
        ml_filled = float(info.get("ml_filled_probability", m["ml"]))

        if (ml_blank >= 0.70 or ml_ambiguous >= 0.85) and m["darkness"] < 145.0:
            return False

        if core_m > 100.0:
            return False

        return (
            m["disk"] >= 0.85
            and
            m["core"] >= 0.78
            and
            m["darkness"] >= 80.0
        )

    strong_options = [
        option
        for option, info
        in option_data.items()
        if is_strong_fill(
            info
        )
    ]

    # --------------------------------------------------------
    # Per-question Y consistency guard for MULTIPLE
    # --------------------------------------------------------
    # A mobile-photo grid fit can occasionally snap one option onto the
    # bubble in the neighboring row.  That produces a very convincing
    # "second fill" even though its crop center is vertically inconsistent
    # with the other options of the same question.
    #
    # Use the median Y of all available A/B/C/D crop centers as the robust
    # row center.  A strong option is eligible to participate in MULTIPLE
    # only when its own crop center is close to that row.
    #
    # Important:
    # - This does NOT change single/blank/ambiguous thresholds.
    # - It does NOT move geometry.
    # - It only prevents row-jump bubbles from creating MULTIPLE.
    MULTIPLE_MAX_Y_DEVIATION = 8.0

    question_center_ys = []

    for info in option_data.values():
        crop_center = info.get(
            "crop_center"
        )

        if (
            crop_center
            and
            len(crop_center) >= 2
        ):
            question_center_ys.append(
                float(
                    crop_center[1]
                )
            )

    if question_center_ys:
        question_median_y = float(
            np.median(
                question_center_ys
            )
        )

        y_consistent_strong_options = []

        for option in strong_options:
            crop_center = option_data[
                option
            ].get(
                "crop_center"
            )

            if not (
                crop_center
                and
                len(crop_center) >= 2
            ):
                continue

            option_y = float(
                crop_center[1]
            )

            if (
                abs(
                    option_y
                    -
                    question_median_y
                )
                <=
                MULTIPLE_MAX_Y_DEVIATION
            ):
                y_consistent_strong_options.append(
                    option
                )

        strong_options = (
            y_consistent_strong_options
        )

    # TRUE MULTIPLE:
    # at least two independently full-disk dark bubbles that also belong
    # to the same detected question row.
    if len(
        strong_options
    ) >= 2 and top_gap < 40.0 and disk_gap < 0.15:
        return {
            "answer":
                "MULTIPLE",

            "status":
                "multiple",

            "multiple_options":
                strong_options,

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),
        }

    # One independently strong fill -> SINGLE immediately.
    if len(
        strong_options
    ) == 1:
        winner = strong_options[0]

        return {
            "answer":
                winner,

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                winner,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),

            "disk_first":
                True,
        }

    # --------------------------------------------------------
    # Medium fill — require clear disk coverage and low ml_blank.
    # --------------------------------------------------------
    best_ml_blank = float(best_info.get("ml_blank_probability", 0.0))
    best_ml_filled = float(best_info.get("ml_filled_probability", best["ml"]))

    best_core_mean = float(best_info.get("metrics", {}).get("core_mean", 0.0))

    medium_fill = (
        best["disk"] >= 0.76
        and
        best["core"] >= 0.70
        and
        best["darkness"] >= 85.0
        and
        (best_core_mean == 0.0 or best_core_mean <= 100.0)
        and
        best_ml_blank < 0.50
        and
        (
            disk_gap >= 0.10
            or
            top_gap >= 15.0
        )
    )

    if medium_fill:
        return {
            "answer":
                best_option,

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),

            "medium_disk_rescue":
                True,
        }

    # --------------------------------------------------------
    # Conservative faint-mark rescue.
    # --------------------------------------------------------
    # Only one answer can be rescued this way. This rule can NEVER create
    # MULTIPLE.
    faint_relative = (
        best["disk"] >= 0.65
        and
        best["core"] >= 0.65
        and
        best["darkness"] >= 70.0
        and
        (best_core_mean == 0.0 or best_core_mean <= 100.0)
        and
        best_delta >= 20.0
        and
        top_gap >= 15.0
        and
        best_ml_blank < 0.50
        and
        (
            best["ml"] >= 0.65
            or
            disk_gap >= 0.10
        )
    )

    if faint_relative:
        return {
            "answer":
                best_option,

            "status":
                "answered",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),

            "faint_relative_rescue":
                True,
        }

    # --------------------------------------------------------
    # True blank
    # --------------------------------------------------------
    # Blank rows have no substantial full-disk darkening.
    best_ml_blank = float(best_info.get("ml_blank_probability", 0.0))
    best_ml_filled = float(best_info.get("ml_filled_probability", best["ml"]))

    best_core_mean = float(best_info.get("metrics", {}).get("core_mean", 255.0))

    blank_like = (
        (
            best_core_mean > 125.0
            and
            best["darkness"] < 135.0
            and
            top_gap < 40.0
        )
        or
        (
            best["disk"] < 0.58
            and
            best["darkness"] < 62.0
            and
            best_delta < 14.0
        )
        or
        (
            best_ml_blank >= 0.65
            and
            best_ml_filled < 0.35
            and
            (best_delta < 40.0 or best["disk"] < 0.85)
        )
    )

    if blank_like:
        return {
            "answer":
                None,

            "status":
                "blank",

            "multiple_options":
                [],

            "best_option":
                best_option,

            "best_darkness":
                round(
                    best["darkness"],
                    3,
                ),

            "second_darkness":
                round(
                    second["darkness"],
                    3,
                ),

            "top_gap":
                round(
                    top_gap,
                    3,
                ),

            "best_disk_ratio":
                round(
                    best["disk"],
                    4,
                ),

            "second_disk_ratio":
                round(
                    second["disk"],
                    4,
                ),

            "disk_gap":
                round(
                    disk_gap,
                    4,
                ),

            "question_blank_baseline":
                round(
                    question_blank_baseline,
                    3,
                ),

            "best_delta":
                round(
                    best_delta,
                    3,
                ),
        }

    # --------------------------------------------------------
    # Ambiguous-only surgical rescue
    # --------------------------------------------------------
    # This runs ONLY after all existing 419-baseline rules have failed.
    # It does not change already-answered / blank / multiple rows.
    #
    # Rescue only when the SAME option is the winner by BOTH:
    #   1) center darkness
    #   2) disk dark ratio
    #
    # and that option has enough separation from the runners-up.
    #
    # This targets the unstable ambiguous rows seen in the 419 baseline
    # without changing the normal decision path.

    by_darkness = sorted(
        option_data.items(),
        key=lambda item: float(
            item[1]["metrics"]["center_darkness"]
        ),
        reverse=True,
    )

    by_disk = sorted(
        option_data.items(),
        key=lambda item: float(
            item[1]["metrics"]["disk_dark_ratio"]
        ),
        reverse=True,
    )

    darkness_winner = by_darkness[0][0]
    disk_winner = by_disk[0][0]

    if darkness_winner == disk_winner:
        winner = darkness_winner

        winner_info = option_data[winner]
        winner_metrics = winner_info["metrics"]

        winner_darkness = float(
            winner_metrics["center_darkness"]
        )
        winner_disk = float(
            winner_metrics["disk_dark_ratio"]
        )
        winner_core = float(
            winner_metrics["core_dark_ratio"]
        )
        winner_ml = float(
            winner_info.get(
                "ml_filled_probability",
                0.0,
            )
        )

        second_darkness_same_metric = float(
            by_darkness[1][1]["metrics"]["center_darkness"]
        )

        second_disk_same_metric = float(
            by_disk[1][1]["metrics"]["disk_dark_ratio"]
        )

        darkness_gap = (
            winner_darkness
            -
            second_darkness_same_metric
        )

        disk_gap_same_metric = (
            winner_disk
            -
            second_disk_same_metric
        )

        # Conservative rescue:
        # - same winner on two independent image features
        # - enough absolute evidence
        # - enough separation on at least one feature
        #
        # ML is supporting evidence only; it cannot rescue on its own.
        winner_core_mean = float(winner_metrics.get("core_mean", 0.0))

        ambiguous_rescue = (
            winner_darkness >= 64.0
            and
            winner_disk >= 0.70
            and
            winner_core >= 0.64
            and
            (winner_core_mean == 0.0 or winner_core_mean <= 100.0)
            and
            (
                darkness_gap >= 10.0
                or
                disk_gap_same_metric >= 0.075
            )
            and
            (
                winner_ml >= 0.95
                or
                darkness_gap >= 14.0
                or
                disk_gap_same_metric >= 0.100
            )
        )

        if ambiguous_rescue:
            return {
                "answer":
                    winner,

                "status":
                    "answered",

                "multiple_options":
                    [],

                "best_option":
                    winner,

                "best_darkness":
                    round(
                        winner_darkness,
                        3,
                    ),

                "second_darkness":
                    round(
                        second_darkness_same_metric,
                        3,
                    ),

                "top_gap":
                    round(
                        darkness_gap,
                        3,
                    ),

                "best_disk_ratio":
                    round(
                        winner_disk,
                        4,
                    ),

                "second_disk_ratio":
                    round(
                        second_disk_same_metric,
                        4,
                    ),

                "disk_gap":
                    round(
                        disk_gap_same_metric,
                        4,
                    ),

                "question_blank_baseline":
                    round(
                        question_blank_baseline,
                        3,
                    ),

                "best_delta":
                    round(
                        (
                            winner_darkness
                            -
                            question_blank_baseline
                        ),
                        3,
                    ),

                "ambiguous_rescue":
                    True,
            }

    # Non-filled rows default to BLANK (per user directive: all uncertain are blank).
    return {
        "answer":
            None,

        "status":
            "blank",

        "multiple_options":
            [],

        "best_option":
            best_option,

        "best_darkness":
            round(
                best["darkness"],
                3,
            ),

        "second_darkness":
            round(
                second["darkness"],
                3,
            ),

        "top_gap":
            round(
                top_gap,
                3,
            ),

        "best_disk_ratio":
            round(
                best["disk"],
                4,
            ),

        "second_disk_ratio":
            round(
                second["disk"],
                4,
            ),

        "disk_gap":
            round(
                disk_gap,
                4,
            ),

        "question_blank_baseline":
            round(
                question_blank_baseline,
                3,
            ),

        "best_delta":
            round(
                best_delta,
                3,
            ),
    }



def export_recognition_report(
    question_data,
    decisions,
    sheet_thresholds,
    csv_path="recognition_report.csv",
    json_path="recognition_report.json",
):
    """
    Export one row per question-option plus the final decision.

    CSV columns include:
      question, option,
      center_darkness, core_dark_ratio, disk_dark_ratio,
      ml_filled_probability, ml_blank_probability,
      ml_ambiguous_probability,
      crop_center_x, crop_center_y,
      final_answer, final_status,
      best_option, best_darkness, second_darkness,
      top_gap, question_blank_baseline, best_delta,
      sheet filled thresholds.

    JSON keeps the full nested debug structure.
    """

    csv_file = Path(
        csv_path
    )

    json_file = Path(
        json_path
    )

    rows = []

    for question, option_data in question_data.items():
        decision = decisions[
            question
        ]

        for option, info in option_data.items():
            metrics = info.get(
                "metrics",
                {},
            )

            crop_center = info.get(
                "crop_center",
                [
                    None,
                    None,
                ],
            )

            rows.append(
                {
                    "question":
                        int(
                            question
                        ),

                    "option":
                        str(
                            option
                        ),

                    "center_darkness":
                        metrics.get(
                            "center_darkness"
                        ),

                    "core_dark_ratio":
                        metrics.get(
                            "core_dark_ratio"
                        ),

                    "disk_dark_ratio":
                        metrics.get(
                            "disk_dark_ratio"
                        ),

                    "core_mean":
                        metrics.get(
                            "core_mean"
                        ),

                    "paper_mean":
                        metrics.get(
                            "paper_mean"
                        ),

                    "ml_filled_probability":
                        info.get(
                            "ml_filled_probability"
                        ),

                    "ml_blank_probability":
                        info.get(
                            "ml_blank_probability"
                        ),

                    "ml_ambiguous_probability":
                        info.get(
                            "ml_ambiguous_probability"
                        ),

                    "crop_center_x":
                        crop_center[
                            0
                        ]
                        if len(
                            crop_center
                        )
                        > 0
                        else None,

                    "crop_center_y":
                        crop_center[
                            1
                        ]
                        if len(
                            crop_center
                        )
                        > 1
                        else None,

                    "final_answer":
                        decision.get(
                            "answer"
                        ),

                    "final_status":
                        decision.get(
                            "status"
                        ),

                    "best_option":
                        decision.get(
                            "best_option"
                        ),

                    "best_darkness":
                        decision.get(
                            "best_darkness"
                        ),

                    "second_darkness":
                        decision.get(
                            "second_darkness"
                        ),

                    "top_gap":
                        decision.get(
                            "top_gap"
                        ),

                    "question_blank_baseline":
                        decision.get(
                            "question_blank_baseline"
                        ),

                    "best_delta":
                        decision.get(
                            "best_delta"
                        ),

                    "sheet_blank_darkness_median":
                        sheet_thresholds.get(
                            "blank_darkness_median"
                        ),

                    "sheet_blank_darkness_mad":
                        sheet_thresholds.get(
                            "blank_darkness_mad"
                        ),

                    "sheet_filled_darkness_threshold":
                        sheet_thresholds.get(
                            "filled_darkness_threshold"
                        ),

                    "sheet_filled_core_ratio_threshold":
                        sheet_thresholds.get(
                            "filled_core_ratio_threshold"
                        ),
                }
            )

    fieldnames = [
        "question",
        "option",
        "center_darkness",
        "core_dark_ratio",
        "disk_dark_ratio",
        "core_mean",
        "paper_mean",
        "ml_filled_probability",
        "ml_blank_probability",
        "ml_ambiguous_probability",
        "crop_center_x",
        "crop_center_y",
        "final_answer",
        "final_status",
        "best_option",
        "best_darkness",
        "second_darkness",
        "top_gap",
        "question_blank_baseline",
        "best_delta",
        "sheet_blank_darkness_median",
        "sheet_blank_darkness_mad",
        "sheet_filled_darkness_threshold",
        "sheet_filled_core_ratio_threshold",
    ]

    with csv_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    json_payload = {
        "sheet_thresholds":
            sheet_thresholds,

        "questions":
            {
                str(
                    question
                ):
                    {
                        "decision":
                            decisions[
                                question
                            ],

                        "options":
                            option_data,
                    }
                for question, option_data
                in question_data.items()
            },
    }

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "csv":
            str(
                csv_file
            ),

        "json":
            str(
                json_file
            ),

        "row_count":
            len(
                rows
            ),
    }



def export_uncertain_bubble_crops(
    gray,
    question_data,
    decisions,
    crop_radius=DEFAULT_CROP_RADIUS,
    output_dir="debug_uncertain_crops",
):
    """
    Export ONLY questions whose final status is ambiguous.

    For each uncertain question this writes:
      - Qxxx_A.png ... Qxxx_D.png
      - Qxxx_overview.png

    The overview enlarges each bubble crop and prints the key metrics:
      center darkness, core ratio, disk ratio, and ML filled probability.

    This is debug-only and does not affect recognition decisions.
    """

    import os

    os.makedirs(output_dir, exist_ok=True)

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    uncertain_questions = []

    for question, decision in decisions.items():
        if decision.get("status") != "ambiguous":
            continue

        uncertain_questions.append(int(question))

        tiles = []

        option_data = question_data.get(
            question,
            {},
        )

        for option in ("A", "B", "C", "D"):
            info = option_data.get(option)
            if not info:
                continue

            center = info.get(
                "crop_center",
                [0, 0],
            )

            x = int(center[0])
            y = int(center[1])

            crop = crop_bubble(
                gray,
                x,
                y,
                crop_radius,
            )

            # Save the exact raw crop used around the final bubble center.
            raw_path = os.path.join(
                output_dir,
                f"Q{int(question):03d}_{option}.png",
            )
            cv2.imwrite(
                raw_path,
                crop,
            )

            metrics = info.get(
                "metrics",
                {},
            )

            center_darkness = float(
                metrics.get(
                    "center_darkness",
                    0.0,
                )
            )

            core_ratio = float(
                metrics.get(
                    "core_dark_ratio",
                    0.0,
                )
            )

            disk_ratio = float(
                metrics.get(
                    "disk_dark_ratio",
                    0.0,
                )
            )

            ml_filled = float(
                info.get(
                    "ml_filled_probability",
                    0.0,
                )
            )

            # Enlarge for easy human inspection.
            display = cv2.resize(
                crop,
                (180, 180),
                interpolation=cv2.INTER_NEAREST,
            )

            display = cv2.cvtColor(
                display,
                cv2.COLOR_GRAY2BGR,
            )

            # Mark the exact crop center.
            cv2.drawMarker(
                display,
                (90, 90),
                (0, 140, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=13,
                thickness=1,
            )

            canvas = np.full(
                (270, 200, 3),
                255,
                dtype=np.uint8,
            )

            canvas[
                8:188,
                10:190,
            ] = display

            lines = [
                f"Q{int(question):03d} {option}",
                f"dark {center_darkness:.1f}",
                f"core {core_ratio:.3f}",
                f"disk {disk_ratio:.3f}",
                f"ML   {ml_filled:.3f}",
            ]

            y_text = 207

            for line in lines:
                cv2.putText(
                    canvas,
                    line,
                    (8, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                y_text += 15

            tiles.append(canvas)

        if tiles:
            overview = cv2.hconcat(
                tiles
            )

            overview_path = os.path.join(
                output_dir,
                f"Q{int(question):03d}_overview.png",
            )

            cv2.imwrite(
                overview_path,
                overview,
            )

    # Also write a simple manifest.
    manifest_path = os.path.join(
        output_dir,
        "uncertain_questions.txt",
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(
                str(q)
                for q in sorted(
                    uncertain_questions
                )
            )
        )

    return {
        "output_dir":
            output_dir,

        "uncertain_count":
            len(
                uncertain_questions
            ),

        "questions":
            sorted(
                uncertain_questions
            ),
    }



def _micro_core_darkness(
    gray,
    x,
    y,
    radius=3,
):
    """Very small center-only darkness signal used only as a tie-breaker."""
    h, w = gray.shape[:2]

    x = int(round(x))
    y = int(round(y))
    r = int(radius)

    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)

    patch = gray[
        y0:y1,
        x0:x1,
    ]

    if patch.size == 0:
        return 0.0

    return float(
        255.0
        -
        np.mean(
            patch
        )
    )


def _final_row_hough_y(
    gray,
    option_map,
):
    """
    Conservative rescue for questions 45/90/135/180.

    Searches only upward from the current fitted row, because the observed
    mobile failure mode places the fitted center on the horizontal divider
    below the actual bubble row.

    A correction is accepted only when >=3 option columns independently
    detect circles at nearly the same Y.
    """
    detected_ys = []

    for _option, (
        x,
        y,
    ) in option_map.items():

        x = int(round(x))
        y = int(round(y))

        x0 = max(
            0,
            x - 17,
        )
        x1 = min(
            gray.shape[1],
            x + 18,
        )

        y0 = max(
            0,
            y - 12,
        )
        y1 = min(
            gray.shape[0],
            y + 9,
        )

        roi = gray[
            y0:y1,
            x0:x1,
        ]

        if (
            roi.size == 0
            or
            roi.shape[0] < 15
            or
            roi.shape[1] < 15
        ):
            continue

        blurred = cv2.GaussianBlur(
            roi,
            (
                3,
                3,
            ),
            0,
        )

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=8,
            param1=80,
            param2=8,
            minRadius=6,
            maxRadius=14,
        )

        if circles is None:
            continue

        candidates = []

        for circle in circles[0]:
            cx, cy, radius = (
                float(
                    circle[0]
                ),
                float(
                    circle[1]
                ),
                float(
                    circle[2]
                ),
            )

            absolute_y = (
                float(
                    y0
                )
                +
                cy
            )

            shift = (
                absolute_y
                -
                float(
                    y
                )
            )

            # Never search as far as the previous full question row.
            if (
                -11.0
                <= shift
                <= 5.0
            ):
                candidates.append(
                    (
                        abs(
                            cx
                            -
                            float(
                                x
                                -
                                x0
                            )
                        ),
                        abs(
                            shift
                        ),
                        absolute_y,
                        radius,
                    )
                )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item:
                (
                    item[0],
                    item[1],
                )
        )

        detected_ys.append(
            float(
                candidates[0][2]
            )
        )

    if len(
        detected_ys
    ) < 3:
        return None

    detected_ys = np.asarray(
        detected_ys,
        dtype=np.float32,
    )

    median_y = float(
        np.median(
            detected_ys
        )
    )

    inliers = detected_ys[
        np.abs(
            detected_ys
            -
            median_y
        )
        <= 3.0
    ]

    if len(
        inliers
    ) < 3:
        return None

    resolved_y = float(
        np.median(
            inliers
        )
    )

    original_y = float(
        np.median(
            [
                float(
                    point[1]
                )
                for point
                in option_map.values()
            ]
        )
    )

    shift = (
        resolved_y
        -
        original_y
    )

    # Only activate for a meaningful downward-line failure.
    if not (
        -11.0
        <= shift
        <= -7.0
    ):
        return None

    return resolved_y



def _tight_local_shape_probe(
    gray,
    option_data,
):
    """
    Tight local shape probe: +/-4 px only.

    Returns image-only descriptors for A/B/C/D. This deliberately cannot
    jump into adjacent NEET rows and is used only as a secondary tie-breaker.
    """
    options = [
        o for o in ("A", "B", "C", "D")
        if o in option_data
    ]
    if len(options) != 4:
        return None

    out = {}

    for option in options:
        center = option_data[option]["crop_center"]
        x0 = int(round(float(center[0])))
        y0 = int(round(float(center[1])))

        samples = []
        for dy in (-4, -2, 0, 2, 4):
            for dx in (-4, -2, 0, 2, 4):
                crop = crop_bubble(
                    gray,
                    x0 + dx,
                    y0 + dy,
                    16,
                )
                metrics = _bubble_metrics(crop)

                center_darkness = float(
                    metrics["center_darkness"]
                )
                core = float(
                    metrics["core_dark_ratio"]
                )
                disk = float(
                    metrics["disk_dark_ratio"]
                )

                # Existing reader may expose micro_core_darkness separately.
                # Recompute a tiny central 7x7 darkness here for consistency.
                h, w = crop.shape[:2]
                cy = h // 2
                cx = w // 2
                r = 3
                tiny = crop[
                    cy-r:cy+r+1,
                    cx-r:cx+r+1,
                ]
                yy, xx = np.ogrid[-r:r+1, -r:r+1]
                mask = (xx*xx + yy*yy) <= r*r
                micro = float(
                    255.0
                    -
                    np.mean(
                        tiny[mask]
                    )
                )

                # Broad fill evidence rewards occupied center and disk.
                broad = (
                    0.45 * center_darkness
                    +
                    28.0 * core
                    +
                    22.0 * disk
                )

                # Compactness rewards a dark tiny center relative to broad core.
                compact = (
                    0.70 * micro
                    +
                    0.30 * center_darkness
                    -
                    20.0 * max(0.0, core - disk)
                )

                samples.append({
                    "dx": int(dx),
                    "dy": int(dy),
                    "center_darkness": center_darkness,
                    "micro_darkness": micro,
                    "core_dark_ratio": core,
                    "disk_dark_ratio": disk,
                    "broad": float(broad),
                    "compact": float(compact),
                })

        best_broad = max(
            samples,
            key=lambda s: s["broad"],
        )
        best_compact = max(
            samples,
            key=lambda s: s["compact"],
        )

        out[option] = {
            "best_broad": best_broad,
            "best_compact": best_compact,
        }

    return out


def _shape_based_secondary_rescue(
    gray,
    option_data,
    decision,
    questions_per_column=45,
    crop_radius=16,
):
    """
    Conservative secondary rescue for three known *failure classes*:
      1) compact true single vs broader dark distractor,
      2) faint second real mark for MULTIPLE,
      3) ambiguous row where tiny-center evidence is stronger than broad core.

    No question IDs and no answer-key lookup.
    """
    probe = _tight_local_shape_probe(
        gray,
        option_data,
    )
    if probe is None:
        return decision

    options = ["A", "B", "C", "D"]

    broad_rank = sorted(
        options,
        key=lambda o: probe[o]["best_broad"]["broad"],
        reverse=True,
    )
    compact_rank = sorted(
        options,
        key=lambda o: probe[o]["best_compact"]["compact"],
        reverse=True,
    )

    broad_best = broad_rank[0]
    broad_second = broad_rank[1]
    compact_best = compact_rank[0]
    compact_second = compact_rank[1]

    bb = probe[broad_best]["best_broad"]
    bs = probe[broad_second]["best_broad"]
    cb = probe[compact_best]["best_compact"]
    cs = probe[compact_second]["best_compact"]

    current_status = str(
        decision.get("status", "")
    ).lower()
    current_answer = decision.get("answer")

    # ----------------------------------------------------------
    # A) Ambiguous compact-center rescue
    # ----------------------------------------------------------
    # If the broad winner differs from the compact winner, allow the
    # compact winner only when its tiny center is meaningfully stronger
    # and it is not merely an empty outline.
    if current_status == "ambiguous":
        if (
            compact_best != broad_best
            and
            cb["micro_darkness"] >= 165.0
            and
            cb["micro_darkness"]
            -
            cs["micro_darkness"]
            >= 4.0
            and
            cb["center_darkness"] >= 72.0
            and
            cb["disk_dark_ratio"] >= 0.30
        ):
            rescued = dict(decision)
            rescued["answer"] = compact_best
            rescued["status"] = "answered"
            rescued["best_option"] = compact_best
            rescued["shape_compact_rescue"] = True
            rescued["shape_probe_dx"] = cb["dx"]
            rescued["shape_probe_dy"] = cb["dy"]
            return rescued

    # ----------------------------------------------------------
    # B) Hidden second mark -> MULTIPLE
    # ----------------------------------------------------------
    # Only consider when current decision is a single. Require the
    # current answer to be unquestionably filled and one other option
    # to independently show a compact, dark local component.
    if (
        current_status == "answered"
        and
        current_answer in options
    ):
        cur = probe[current_answer]["best_broad"]

        if (
            cur["center_darkness"] >= 110.0
            and
            cur["core_dark_ratio"] >= 0.90
            and
            cur["disk_dark_ratio"] >= 0.80
        ):
            second_candidates = []
            for option in options:
                if option == current_answer:
                    continue

                p = probe[option]["best_compact"]

                is_kcet = (questions_per_column == 60 or crop_radius <= 12)
                min_sec_micro = 145.0 if is_kcet else 170.0
                min_sec_dark = 95.0 if is_kcet else 82.0
                min_sec_disk = 0.65 if is_kcet else 0.42

                # Deliberately strict; intended for a real second fill,
                # not printed bubble ring.
                if (
                    p["micro_darkness"] >= min_sec_micro
                    and
                    p["center_darkness"] >= min_sec_dark
                    and
                    p["core_dark_ratio"] >= 0.70
                    and
                    p["disk_dark_ratio"] >= min_sec_disk
                ):
                    second_candidates.append(
                        (
                            p["compact"],
                            option,
                            p,
                        )
                    )

            if len(second_candidates) == 1:
                _, second_option, second_probe = second_candidates[0]

                rescued = dict(decision)
                rescued["answer"] = "MULTIPLE"
                rescued["status"] = "multiple"
                rescued["best_option"] = current_answer
                rescued["multiple_options"] = [
                    current_answer,
                    second_option,
                ]
                rescued["shape_hidden_multiple_rescue"] = True
                rescued["shape_second_option"] = second_option
                rescued["shape_second_dx"] = second_probe["dx"]
                rescued["shape_second_dy"] = second_probe["dy"]
                return rescued

    # ----------------------------------------------------------
    # C) Wrong-single compactness correction
    # ----------------------------------------------------------
    # For a weak current single, permit a different compact winner if
    # it is clearly more centered, but never override a very strong
    # full-disk fill.
    if (
        current_status == "answered"
        and
        current_answer in options
        and
        compact_best != current_answer
    ):
        cur = probe[current_answer]["best_broad"]

        if (
            cur["disk_dark_ratio"] < 0.78
            and
            cb["micro_darkness"] >= 168.0
            and
            cb["micro_darkness"]
            -
            probe[current_answer]["best_compact"]["micro_darkness"]
            >= 8.0
            and
            cb["center_darkness"] >= 76.0
        ):
            rescued = dict(decision)
            rescued["answer"] = compact_best
            rescued["status"] = "answered"
            rescued["best_option"] = compact_best
            rescued["shape_wrong_single_rescue"] = True
            rescued["shape_probe_dx"] = cb["dx"]
            rescued["shape_probe_dy"] = cb["dy"]
            return rescued

    return decision


def _postprocess_known_failure_classes(
    question,
    option_data,
    decision,
    gray,
    questions_per_column=45,
    crop_radius=16,
):
    """
    General postprocessor for the remaining real-sheet failure classes.

    No answer-key lookup and no question-number answer hardcoding.

    Handles:
      1) outline-heavy weak rows -> BLANK,
      2) one genuinely supported weak single -> SINGLE,
      3) low-disk answered rows where a tiny center-only measurement gives
         a unique different winner -> center tie-break correction.
    """
    options = [
        option
        for option
        in (
            "A",
            "B",
            "C",
            "D",
        )
        if option in option_data
    ]


    if len(
        options
    ) != 4:
        return decision

    infos = {
        option:
            option_data[
                option
            ]
        for option
        in options
    }

    def value(
        option,
        key,
        default=0.0,
    ):
        return float(
            infos[
                option
            ].get(
                "metrics",
                {},
            ).get(
                key,
                default,
            )
        )

    def ml(
        option,
    ):
        return float(
            infos[
                option
            ].get(
                "ml_filled_probability",
                0.0,
            )
        )

    ys = [
        float(
            infos[
                option
            ][
                "crop_center"
            ][1]
        )
        for option
        in options
    ]

    coherent = (
        max(
            ys
        )
        -
        min(
            ys
        )
        <= 5.0
    )

    if not coherent:
        return decision

    # --------------------------------------------------------------
    # A) Correct a rare low-disk false SINGLE using only the tiny
    #    center measurement. This cannot alter strong/medium fills.
    # --------------------------------------------------------------
    if (
        decision.get(
            "status"
        )
        ==
        "answered"
    ):
        max_disk = max(
            value(
                option,
                "disk_dark_ratio",
            )
            for option
            in options
        )

        if max_disk < 0.50:
            micro_ranked = sorted(
                options,
                key=lambda option:
                    float(
                        infos[
                            option
                        ].get(
                            "micro_core_darkness",
                            0.0,
                        )
                    ),
                reverse=True,
            )

            micro_winner = (
                micro_ranked[0]
            )

            micro_best = float(
                infos[
                    micro_winner
                ].get(
                    "micro_core_darkness",
                    0.0,
                )
            )

            micro_second = float(
                infos[
                    micro_ranked[1]
                ].get(
                    "micro_core_darkness",
                    0.0,
                )
            )

            current_answer = (
                decision.get(
                    "answer"
                )
            )

            # Apply only when the current answer is weak and a different
            # option owns the center-most signal. The gap can be small
            # because this is a tie-breaker, not a primary detector.
            if (
                micro_winner
                != current_answer
                and
                micro_best >= 150.0
                and
                micro_best
                -
                micro_second
                >= 0.25
            ):
                corrected = dict(
                    decision
                )

                corrected[
                    "answer"
                ] = (
                    micro_winner
                )

                corrected[
                    "best_option"
                ] = (
                    micro_winner
                )

                corrected[
                    "micro_core_tiebreak"
                ] = True

                corrected[
                    "micro_core_darkness"
                ] = round(
                    micro_best,
                    3,
                )

                return corrected

        return decision

    if (
        decision.get(
            "status"
        )
        not in (
            "ambiguous",
            "blank",
        )
    ):
        return decision

    # --------------------------------------------------------------
    # AMBIGUOUS LOW-DISK -> BLANK
    # --------------------------------------------------------------
    # Geometry is already correct in the current scan. The remaining
    # uncertain rows are empty printed bubbles whose center/core can look
    # dark, while none of the four options has substantial filled-disk
    # coverage.
    #
    # Classification-only rule:
    #   - runs ONLY on rows that are already ambiguous,
    #   - never changes an answered or multiple row,
    #   - never moves any bubble center,
    #   - requires every option to stay below 0.45 disk coverage.
    #
    # A real marked bubble in this sheet is much broader (normally near
    # disk=1.0; faint-mark rescues already require >=0.54 earlier).
    ambiguous_max_disk = max(
        value(
            option,
            "disk_dark_ratio",
        )
        for option
        in options
    )

    if ambiguous_max_disk < 0.45:
        blanked = dict(
            decision
        )

        blanked[
            "answer"
        ] = None

        blanked[
            "status"
        ] = "blank"

        blanked[
            "ambiguous_low_disk_blank"
        ] = True

        blanked[
            "ambiguous_max_disk"
        ] = round(
            float(
                ambiguous_max_disk
            ),
            4,
        )

        return blanked

    # --------------------------------------------------------------
    # KCET DOMINANT SINGLE RESCUE
    # --------------------------------------------------------------
    # Fix false UNCERTAIN on KCET where printed grid outlines near A/B
    # inflate baseline darkness but a single option is unambiguously filled.
    is_kcet = (questions_per_column == 60 or crop_radius <= 12)
    if decision.get("status") == "ambiguous" and is_kcet:
        best_option = decision.get("best_option")
        top_gap = float(decision.get("top_gap", 0.0))
        if best_option and best_option in option_data:
            b_info = option_data[best_option]
            b_micro = float(b_info.get("micro_core_darkness", 0.0))
            b_metrics = b_info.get("metrics", {})
            b_dark = float(b_metrics.get("center_darkness", 0.0))
            b_disk = float(b_metrics.get("disk_dark_ratio", 0.0))

            if (
                b_micro >= 145.0
                and b_dark >= 95.0
                and b_disk >= 0.58
                and top_gap >= 12.0
            ):
                rescued = dict(decision)
                rescued["answer"] = best_option
                rescued["status"] = "answered"
                rescued["kcet_dominant_single_rescue"] = True
                return rescued

    # --------------------------------------------------------------
    # FINAL-ROW SHARED VERTICAL SWEEP
    # --------------------------------------------------------------
    # The observed Q45 failure is not a classification problem: all four
    # fitted centers land on the divider below the actual bubble row.
    #
    # For final rows only, scan a narrow band 7..20 px upward, which is
    # still safely short of the previous question-row spacing (~29 px).
    # At each shared offset, measure a small disk at A/B/C/D. Accept only
    # an extremely clear one-option winner. This avoids moving blank final
    # rows merely because of printed outlines.
    if (
        int(
            question
        )
        %
        questions_per_column
        ==
        0
    ):
        sweep_results = []

        for dy in range(
            -20,
            -6,
        ):
            option_scores = []

            for option in options:
                center = infos[
                    option
                ][
                    "crop_center"
                ]

                x = int(
                    round(
                        float(
                            center[0]
                        )
                    )
                )

                y = int(
                    round(
                        float(
                            center[1]
                        )
                        +
                        float(
                            dy
                        )
                    )
                )

                radius = 7

                if (
                    x - radius < 0
                    or
                    y - radius < 0
                    or
                    x + radius >= gray.shape[1]
                    or
                    y + radius >= gray.shape[0]
                ):
                    continue

                yy, xx = np.ogrid[
                    -radius:
                    radius + 1,
                    -radius:
                    radius + 1,
                ]

                mask = (
                    xx
                    *
                    xx
                    +
                    yy
                    *
                    yy
                    <=
                    radius
                    *
                    radius
                )

                patch = gray[
                    y - radius:
                    y + radius + 1,
                    x - radius:
                    x + radius + 1,
                ]

                darkness = float(
                    255.0
                    -
                    np.mean(
                        patch[
                            mask
                        ]
                    )
                )

                option_scores.append(
                    (
                        darkness,
                        option,
                    )
                )

            if len(
                option_scores
            ) != 4:
                continue

            option_scores.sort(
                reverse=True
            )

            best_darkness_sweep = float(
                option_scores[0][0]
            )

            second_darkness_sweep = float(
                option_scores[1][0]
            )

            sweep_results.append(
                (
                    best_darkness_sweep
                    -
                    second_darkness_sweep,
                    best_darkness_sweep,
                    int(
                        dy
                    ),
                    option_scores[0][1],
                )
            )

        if sweep_results:
            sweep_results.sort(
                reverse=True
            )

            (
                sweep_gap,
                sweep_best,
                sweep_dy,
                sweep_option,
            ) = sweep_results[0]

            # Require a very dark winner AND a large separation from every
            # other option. Printed bubble rings do not normally satisfy
            # both conditions at the same shared Y.
            if (
                sweep_best >= 160.0
                and
                sweep_gap >= 25.0
                and
                abs(sweep_dy) <= 6.0
            ):
                rescued = dict(
                    decision
                )

                rescued[
                    "answer"
                ] = sweep_option

                rescued[
                    "status"
                ] = "answered"

                rescued[
                    "best_option"
                ] = sweep_option

                rescued[
                    "final_row_vertical_sweep"
                ] = True

                rescued[
                    "final_row_sweep_dy"
                ] = int(
                    sweep_dy
                )

                rescued[
                    "final_row_sweep_darkness"
                ] = round(
                    float(
                        sweep_best
                    ),
                    3,
                )

                rescued[
                    "final_row_sweep_gap"
                ] = round(
                    float(
                        sweep_gap
                    ),
                    3,
                )

                return rescued

        return decision

    # --------------------------------------------------------------
    # B) Strongly-supported weak SINGLE.
    #    This separates the real weak B-like pattern from the known
    #    blank-outline patterns.
    # --------------------------------------------------------------
    supported = []

    for option in options:
        disk = value(
            option,
            "disk_dark_ratio",
        )

        core = value(
            option,
            "core_dark_ratio",
        )

        darkness = value(
            option,
            "center_darkness",
        )

        probability = ml(
            option
        )

        if (
            disk >= 0.65
            and
            core >= 0.80
            and
            darkness >= 145.0
            and
            probability >= 0.90
        ):
            supported.append(
                option
            )

    if len(
        supported
    ) == 1:
        winner = (
            supported[0]
        )

        rescued = dict(
            decision
        )

        rescued[
            "answer"
        ] = winner

        rescued[
            "status"
        ] = "answered"

        rescued[
            "best_option"
        ] = winner

        rescued[
            "weak_supported_single"
        ] = True

        return rescued

    # --------------------------------------------------------------
    # C) Outline-heavy BLANK.
    #
    # This runs only after every normal fill rule has failed and after
    # the supported weak-single check above.
    # --------------------------------------------------------------
    max_disk = max(
        value(
            option,
            "disk_dark_ratio",
        )
        for option
        in options
    )

    max_darkness = max(
        value(
            option,
            "center_darkness",
        )
        for option
        in options
    )

    disk_sorted = sorted(
        [
            value(
                option,
                "disk_dark_ratio",
            )
            for option
            in options
        ],
        reverse=True,
    )

    disk_gap = (
        disk_sorted[0]
        -
        disk_sorted[1]
    )

    likely_outline_blank = (
        max_disk <= 0.61
        # Empty printed rings can create a local center-darkness peak in the
        # low 80s while all four bubbles still lack real filled-disk coverage.
        # This path only runs after normal SINGLE/MULTIPLE decisions failed.
        and
        max_darkness <= 85.0
        and
        disk_gap <= 0.14
    )

    # Special protection for a dark central printed/ring artifact:
    # if the darkest option has very low ML support and modest disk
    # coverage, it should not be forced into a SINGLE.
    darkest_option = max(
        options,
        key=lambda option:
            value(
                option,
                "center_darkness",
            ),
    )

    dark_artifact_blank = (
        value(
            darkest_option,
            "center_darkness",
        )
        <= 76.0
        and
        value(
            darkest_option,
            "disk_dark_ratio",
        )
        <= 0.55
        and
        ml(
            darkest_option
        )
        <= 0.10
        and
        max_disk <= 0.61
    )

    # A printed ring can be locally dark even when its classifier result is
    # decisively not filled.  Treat the row as an outline-heavy blank when no
    # option has substantial disk coverage and the center-darkness winner is
    # precisely that unsupported artifact.  This is evaluated before any
    # ambiguous-single rescue, so a misleading model rank cannot turn a blank
    # response into an answer.
    unsupported_dark_outline_blank = (
        max_disk <= 0.52
        and
        disk_gap <= 0.14
        and
        ml(darkest_option) <= 0.10
    )

    if (
        likely_outline_blank
        or
        dark_artifact_blank
        or
        unsupported_dark_outline_blank
    ):
        blanked = dict(
            decision
        )

        blanked[
            "answer"
        ] = None

        blanked[
            "status"
        ] = "blank"

        blanked[
            "outline_blank_rescue"
        ] = True

        return blanked

    # --------------------------------------------------------------
    # D) UNIQUE VISUAL-WINNER SINGLE
    # --------------------------------------------------------------
    # This is deliberately a classification-only fallback for rows which
    # have already passed the normal MULTIPLE and BLANK decisions.  The
    # earlier disk-first rules can otherwise leave a real, clearly dominant
    # mark as "ambiguous" solely because it falls just below a fixed
    # darkness, disk, or ML-confidence cutoff.
    #
    # Do not use model confidence here.  A normal answer is preserved only
    # when the same option is the unique winner in both existing image
    # measurements (filled-disk coverage and center darkness).  Two close
    # contenders remain ambiguous, and the established multiple/blank rules
    # above remain unchanged.
    disk_ranked = sorted(
        options,
        key=lambda option: value(
            option,
            "disk_dark_ratio",
        ),
        reverse=True,
    )

    darkness_ranked = sorted(
        options,
        key=lambda option: value(
            option,
            "center_darkness",
        ),
        reverse=True,
    )

    disk_winner = disk_ranked[0]
    darkness_winner = darkness_ranked[0]
    disk_gap = (
        value(disk_winner, "disk_dark_ratio")
        -
        value(disk_ranked[1], "disk_dark_ratio")
    )
    darkness_gap = (
        value(darkness_winner, "center_darkness")
        -
        value(darkness_ranked[1], "center_darkness")
    )
    winner_core_mean = float(infos[disk_winner].get("metrics", {}).get("core_mean", 0.0))

    if (
        disk_winner == darkness_winner
        and ambiguous_max_disk >= 0.45
        and disk_gap >= 0.10
        and darkness_gap >= 10.0
        and value(disk_winner, "center_darkness") >= 75.0
        and (winner_core_mean == 0.0 or winner_core_mean <= 100.0)
    ):
        answered = dict(decision)
        answered["answer"] = disk_winner
        answered["status"] = "answered"
        answered["best_option"] = disk_winner
        answered["unique_visual_winner_rescue"] = True
        answered["unique_visual_disk_gap"] = round(float(disk_gap), 4)
        answered["unique_visual_darkness_gap"] = round(float(darkness_gap), 3)
        return answered

    # --------------------------------------------------------------
    # E) CENTER-MOST AMBIGUOUS SINGLE
    # --------------------------------------------------------------
    # Some true marks occupy the very center of the bubble while a nearby
    # printed ring/edge gives another option a larger broad core score.
    #
    # This runs only AFTER the blank resolver above, so known outline-heavy
    # blank patterns are already removed. It also requires the center-most
    # signal to be uniquely dark and the ordinary crop itself to contain
    # meaningful darkness.
    micro_ranked = sorted(
        options,
        key=lambda option:
            float(
                infos[
                    option
                ].get(
                    "micro_core_darkness",
                    0.0,
                )
            ),
        reverse=True,
    )

    micro_winner = (
        micro_ranked[0]
    )

    micro_best = float(
        infos[
            micro_winner
        ].get(
            "micro_core_darkness",
            0.0,
        )
    )

    micro_second = float(
        infos[
            micro_ranked[1]
        ].get(
            "micro_core_darkness",
            0.0,
        )
    )

    micro_winner_darkness = value(
        micro_winner,
        "center_darkness",
    )

    micro_winner_disk = value(
        micro_winner,
        "disk_dark_ratio",
    )

    if (
        micro_best >= 168.0
        and
        micro_best
        -
        micro_second
        >= 3.0
        and
        micro_winner_darkness >= 72.0
        and
        micro_winner_disk <= 0.42
    ):
        centered = dict(
            decision
        )

        centered[
            "answer"
        ] = micro_winner

        centered[
            "status"
        ] = "answered"

        centered[
            "best_option"
        ] = micro_winner

        centered[
            "center_most_ambiguous_rescue"
        ] = True

        centered[
            "micro_core_darkness"
        ] = round(
            micro_best,
            3,
        )

        centered[
            "micro_core_gap"
        ] = round(
            micro_best
            -
            micro_second,
            3,
        )

        return centered

    decision = _shape_based_secondary_rescue(
        gray,
        option_data,
        decision,
        questions_per_column=questions_per_column,
        crop_radius=crop_radius,
    )

    return decision


def scan_answers_ml(
    gray,
    coordinates,
    crop_radius=DEFAULT_CROP_RADIUS,
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
    questions_per_column=45,
):
    """
    Final tuned adaptive hybrid reader.

    Main changes:
      - BLANK needs weak absolute AND weak relative evidence
      - faint but clearly separated fills can be rescued
      - MULTIPLE still requires independently strong bubbles
      - ML remains supporting evidence, not the scoring engine
    """

    del filled_confidence
    del ambiguous_confidence

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    batch_crops = []
    batch_map = []

    question_data = {}

    for question, option_map in (
        coordinates.items()
    ):

        question_data[
            question
        ] = {}

        working_option_map = {
            option: (
                float(
                    point[0]
                ),
                float(
                    point[1]
                ),
            )
            for option, point
            in option_map.items()
        }

        if (
            int(
                question
            )
            %
            questions_per_column
            ==
            0
        ):
            rescued_y = _final_row_hough_y(
                gray,
                working_option_map,
            )

            if rescued_y is not None:
                working_option_map = {
                    option: (
                        float(
                            point[0]
                        ),
                        float(
                            rescued_y
                        ),
                    )
                    for option, point
                    in working_option_map.items()
                }

        for option, (
            x,
            y,
        ) in working_option_map.items():

            # The grid detector already fitted the actual printed bubble
            # lattice. Do NOT move the center again here, otherwise the
            # final local search can drift away from the fitted circle.
            refined_x = int(round(x))
            refined_y = int(round(y))

            crop = crop_bubble(
                gray,
                refined_x,
                refined_y,
                crop_radius,
            )

            metrics = _bubble_metrics(
                crop
            )

            question_data[
                question
            ][
                option
            ] = {
                "metrics":
                    metrics,

                "crop_center": [
                    int(refined_x),
                    int(refined_y),
                ],

                "calibrated_center": [
                    int(round(x)),
                    int(round(y)),
                ],

                "micro_core_darkness":
                    _micro_core_darkness(
                        gray,
                        refined_x,
                        refined_y,
                    ),
            }

            batch_crops.append(
                crop
            )

            batch_map.append(
                (
                    question,
                    option,
                )
            )

    predictions = classify_batch(
        batch_crops
    )

    for (
        question,
        option,
    ), prediction in zip(
        batch_map,
        predictions,
    ):

        question_data[
            question
        ][
            option
        ][
            "ml"
        ] = prediction

        question_data[
            question
        ][
            option
        ][
            "ml_filled_probability"
        ] = round(
            _ml_probability(
                prediction,
                "filled",
            ),
            4,
        )

        question_data[
            question
        ][
            option
        ][
            "ml_blank_probability"
        ] = round(
            _ml_probability(
                prediction,
                "blank",
            ),
            4,
        )

        question_data[
            question
        ][
            option
        ][
            "ml_ambiguous_probability"
        ] = round(
            _ml_probability(
                prediction,
                "ambiguous",
            ),
            4,
        )

    sheet_thresholds = (
        _estimate_blank_distribution(
            question_data
        )
    )

    answers = {}
    debug = {}
    decisions = {}

    for question, option_data in (
        question_data.items()
    ):

        decision = _decide_question(
            option_data,
            sheet_thresholds,
        )

        decision = _postprocess_known_failure_classes(
            question,
            option_data,
            decision,
            gray,
            questions_per_column=questions_per_column,
            crop_radius=crop_radius,
        )

        decisions[
            question
        ] = decision

        answers[
            question
        ] = decision[
            "answer"
        ]

        debug[
            question
        ] = {
            **decision,

            "sheet_thresholds":
                sheet_thresholds,

            "options":
                option_data,
        }

    try:
        report_info = (
            export_recognition_report(
                question_data=
                    question_data,

                decisions=
                    decisions,

                sheet_thresholds=
                    sheet_thresholds,

                csv_path=
                    "recognition_report.csv",

                json_path=
                    "recognition_report.json",
            )
        )

        debug[
            "_recognition_report"
        ] = report_info

    except Exception as report_error:
        debug[
            "_recognition_report"
        ] = {
            "error":
                str(
                    report_error
                ),
        }


    try:
        uncertain_crop_info = (
            export_uncertain_bubble_crops(
                gray=gray,

                question_data=
                    question_data,

                decisions=
                    decisions,

                crop_radius=
                    crop_radius,

                output_dir=
                    "debug_uncertain_crops",
            )
        )

        debug[
            "_uncertain_crops"
        ] = uncertain_crop_info

    except Exception as crop_export_error:
        debug[
            "_uncertain_crops"
        ] = {
            "error":
                str(
                    crop_export_error
                ),
        }

    return (
        answers,
        debug,
    )

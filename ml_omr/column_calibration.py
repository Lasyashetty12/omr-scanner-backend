from __future__ import annotations

import cv2
import numpy as np


from functools import lru_cache


# ============================================================
# MOBILE-PHOTO CALIBRATION SETTINGS
# ============================================================

DEFAULT_SEARCH_RADIUS = 6
DEFAULT_PATCH_RADIUS = 11

# Dense anchor spacing. For 45 rows this gives about 10 anchors/column.
ANCHOR_STEP = 9

# Hard safety limits. Ordinary handheld perspective should remain well
# inside these values after the main registration-block homography.
HARD_ABSOLUTE_OFFSET = 20.0
HARD_LOCAL_JUMP = 16.0


# ============================================================
# LOW-LEVEL LOCAL CENTER SEARCH
# ============================================================

@lru_cache(maxsize=256)
def _circle_mask(
    height,
    width,
    cx,
    cy,
    inner_radius,
    outer_radius,
):
    yy, xx = np.ogrid[
        :height,
        :width,
    ]

    rr = np.sqrt(
        (xx - cx) ** 2
        +
        (yy - cy) ** 2
    )

    return (
        (rr >= inner_radius)
        &
        (rr <= outer_radius)
    )


def _score_bubble_center(
    patch,
):
    """
    Score how well the patch center matches a printed OMR bubble.

    We primarily look for the printed circular ring. This works for both
    empty and filled bubbles, so calibration is not biased toward answers.
    """

    if patch is None or patch.size == 0:
        return -1e9

    if patch.ndim == 3:
        patch = cv2.cvtColor(
            patch,
            cv2.COLOR_BGR2GRAY,
        )

    h, w = patch.shape[:2]

    if h < 15 or w < 15:
        return -1e9

    cx = (
        w - 1
    ) / 2.0

    cy = (
        h - 1
    ) / 2.0

    radius = min(
        h,
        w,
    ) / 2.0

    ring_mask = _circle_mask(
        h,
        w,
        cx,
        cy,
        radius * 0.46,
        radius * 0.82,
    )

    center_mask = _circle_mask(
        h,
        w,
        cx,
        cy,
        0.0,
        radius * 0.25,
    )

    outside_mask = _circle_mask(
        h,
        w,
        cx,
        cy,
        radius * 0.85,
        radius * 0.98,
    )

    if (
        not np.any(
            ring_mask
        )
        or
        not np.any(
            center_mask
        )
    ):
        return -1e9

    ring_mean = float(
        np.mean(
            patch[
                ring_mask
            ]
        )
    )

    center_mean = float(
        np.mean(
            patch[
                center_mask
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
        else 220.0
    )

    ring_darkness = (
        outside_mean
        -
        ring_mean
    )

    # Printed ring should be darker than nearby background.
    # The center may be dark or light depending on whether the bubble is filled,
    # so it only contributes a tiny stabilizing term.
    return float(
        ring_darkness
        +
        0.02
        *
        (
            255.0
            -
            center_mean
        )
    )


def _local_search_best_center(
    gray,
    x,
    y,
    search_radius=DEFAULT_SEARCH_RADIUS,
    patch_radius=DEFAULT_PATCH_RADIUS,
):
    """
    Search around one expected JSON bubble coordinate.

    Returns:
        ((best_x, best_y), score)
    """

    h, w = gray.shape[:2]

    best_score = -1e9
    best_xy = (
        int(
            round(
                x
            )
        ),
        int(
            round(
                y
            )
        ),
    )

    # 2 px search step provides spatial noise filtering for mobile photo sensors.
    for dy in range(
        -search_radius,
        search_radius + 1,
        2,
    ):
        for dx in range(
            -search_radius,
            search_radius + 1,
            2,
        ):
            cx = int(
                round(
                    x + dx
                )
            )

            cy = int(
                round(
                    y + dy
                )
            )

            x1 = max(
                0,
                cx - patch_radius,
            )

            y1 = max(
                0,
                cy - patch_radius,
            )

            x2 = min(
                w,
                cx + patch_radius + 1,
            )

            y2 = min(
                h,
                cy + patch_radius + 1,
            )

            patch = gray[
                y1:y2,
                x1:x2,
            ]

            score = _score_bubble_center(
                patch
            )

            if score > best_score:
                best_score = score
                best_xy = (
                    cx,
                    cy,
                )

    return (
        best_xy,
        float(
            best_score
        ),
    )


# ============================================================
# ROBUST STATISTICS / SMOOTHING
# ============================================================

def _robust_median(
    values,
):
    if not values:
        return 0.0

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    median = float(
        np.median(
            values
        )
    )

    deviation = np.abs(
        values
        -
        median
    )

    mad = float(
        np.median(
            deviation
        )
    )

    if mad <= 1e-6:
        return median

    keep = (
        deviation
        <=
        max(
            2.5,
            3.5 * mad,
        )
    )

    filtered = values[
        keep
    ]

    if filtered.size == 0:
        return median

    return float(
        np.median(
            filtered
        )
    )


def _moving_median(
    values,
    radius=1,
):
    values = list(
        values
    )

    output = []

    for index in range(
        len(
            values
        )
    ):
        start = max(
            0,
            index - radius,
        )

        end = min(
            len(
                values
            ),
            index + radius + 1,
        )

        output.append(
            float(
                np.median(
                    values[
                        start:end
                    ]
                )
            )
        )

    return output


def _smooth_anchors(
    anchors,
):
    if len(
        anchors
    ) <= 2:
        return anchors

    dx_values = [
        float(
            anchor[
                "dx"
            ]
        )
        for anchor
        in anchors
    ]

    dy_values = [
        float(
            anchor[
                "dy"
            ]
        )
        for anchor
        in anchors
    ]

    dx_values = _moving_median(
        dx_values,
        radius=1,
    )

    dy_values = _moving_median(
        dy_values,
        radius=1,
    )

    smoothed = []

    for anchor, dx, dy in zip(
        anchors,
        dx_values,
        dy_values,
    ):
        smoothed.append(
            {
                "row":
                    int(
                        anchor[
                            "row"
                        ]
                    ),

                "dx":
                    round(
                        float(
                            dx
                        ),
                        2,
                    ),

                "dy":
                    round(
                        float(
                            dy
                        ),
                        2,
                    ),

                "raw_dx":
                    anchor.get(
                        "raw_dx",
                        anchor[
                            "dx"
                        ],
                    ),

                "raw_dy":
                    anchor.get(
                        "raw_dy",
                        anchor[
                            "dy"
                        ],
                    ),
            }
        )

    return smoothed


def _interpolate_offset(
    row_index,
    anchors,
):
    anchors = sorted(
        anchors,
        key=lambda item:
            item[
                "row"
            ],
    )

    if not anchors:
        return (
            0.0,
            0.0,
        )

    if row_index <= anchors[0]["row"]:
        return (
            float(
                anchors[0][
                    "dx"
                ]
            ),
            float(
                anchors[0][
                    "dy"
                ]
            ),
        )

    if row_index >= anchors[-1]["row"]:
        return (
            float(
                anchors[-1][
                    "dx"
                ]
            ),
            float(
                anchors[-1][
                    "dy"
                ]
            ),
        )

    for left, right in zip(
        anchors[:-1],
        anchors[1:],
    ):
        if (
            left[
                "row"
            ]
            <= row_index
            <= right[
                "row"
            ]
        ):
            span = max(
                1,
                right[
                    "row"
                ]
                -
                left[
                    "row"
                ],
            )

            t = (
                row_index
                -
                left[
                    "row"
                ]
            ) / float(
                span
            )

            dx = (
                float(
                    left[
                        "dx"
                    ]
                )
                +
                t
                *
                (
                    float(
                        right[
                            "dx"
                        ]
                    )
                    -
                    float(
                        left[
                            "dx"
                        ]
                    )
                )
            )

            dy = (
                float(
                    left[
                        "dy"
                    ]
                )
                +
                t
                *
                (
                    float(
                        right[
                            "dy"
                        ]
                    )
                    -
                    float(
                        left[
                            "dy"
                        ]
                    )
                )
            )

            return (
                dx,
                dy,
            )

    return (
        0.0,
        0.0,
    )


# ============================================================
# DENSE MOBILE-PHOTO AUTO CALIBRATION
# ============================================================

def auto_calibrate_neet_columns(
    gray,
    template,
    search_radius=DEFAULT_SEARCH_RADIUS,
):
    """
    Dense local calibration designed for mobile photos.

    Instead of one dx/dy per column, this creates many anchors down each
    response column. That allows the runtime coordinates to follow gradual
    perspective / paper curvature.

    JSON stays unchanged.
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    columns = template[
        "columns"
    ]

    y_positions = template[
        "question_y_positions"
    ]

    options = template[
        "options"
    ]

    row_count = len(
        y_positions
    )

    row_spacing = (
        float(y_positions[-1] - y_positions[0]) / float(len(y_positions) - 1)
        if len(y_positions) > 1
        else 28.0
    )

    calib_patch_radius = min(
        DEFAULT_PATCH_RADIUS,
        max(8, int(round(row_spacing * 0.42))),
    )

    # Include first and last rows explicitly.
    anchor_rows = list(
        range(
            0,
            row_count,
            ANCHOR_STEP,
        )
    )

    if (
        row_count - 1
        not in anchor_rows
    ):
        anchor_rows.append(
            row_count - 1
        )

    calibration = {}

    for column_index, column in enumerate(
        columns
    ):
        anchors = []

        for anchor_row in anchor_rows:

            # Use a local band around each anchor.
            band_start = max(
                0,
                anchor_row - 2,
            )

            band_end = min(
                row_count,
                anchor_row + 3,
            )

            dx_values = []
            dy_values = []
            detection_scores = []

            for row_index in range(
                band_start,
                band_end,
            ):
                y = y_positions[
                    row_index
                ]

                for option in options:
                    x = column[
                        option
                    ]

                    (
                        detected_xy,
                        score,
                    ) = _local_search_best_center(
                        gray,
                        x,
                        y,
                        search_radius=
                            search_radius,
                        patch_radius=
                            calib_patch_radius,
                    )

                    dx_values.append(
                        float(
                            detected_xy[
                                0
                            ]
                            -
                            x
                        )
                    )

                    dy_values.append(
                        float(
                            detected_xy[
                                1
                            ]
                            -
                            y
                        )
                    )

                    detection_scores.append(
                        float(
                            score
                        )
                    )

            raw_dx = _robust_median(
                dx_values
            )

            raw_dy = _robust_median(
                dy_values
            )

            raw_dx = float(
                np.clip(
                    raw_dx,
                    -HARD_ABSOLUTE_OFFSET,
                    HARD_ABSOLUTE_OFFSET,
                )
            )

            raw_dy = float(
                np.clip(
                    raw_dy,
                    -HARD_ABSOLUTE_OFFSET,
                    HARD_ABSOLUTE_OFFSET,
                )
            )

            anchors.append(
                {
                    "row":
                        int(
                            anchor_row
                        ),

                    "dx":
                        round(
                            raw_dx,
                            2,
                        ),

                    "dy":
                        round(
                            raw_dy,
                            2,
                        ),

                    "raw_dx":
                        round(
                            raw_dx,
                            2,
                        ),

                    "raw_dy":
                        round(
                            raw_dy,
                            2,
                        ),

                    "mean_detection_score":
                        round(
                            float(
                                np.mean(
                                    detection_scores
                                )
                            )
                            if detection_scores
                            else 0.0,
                            3,
                        ),
                }
            )

        anchors = _smooth_anchors(
            anchors
        )

        calibration[
            column_index
        ] = {
            "anchors":
                anchors,

            "dx":
                round(
                    float(
                        np.median(
                            [
                                anchor[
                                    "dx"
                                ]
                                for anchor
                                in anchors
                            ]
                        )
                    ),
                    2,
                ),

            "dy":
                round(
                    float(
                        np.median(
                            [
                                anchor[
                                    "dy"
                                ]
                                for anchor
                                in anchors
                            ]
                        )
                    ),
                    2,
                ),
        }

    return calibration


# ============================================================
# MOBILE-FRIENDLY VALIDATION
# ============================================================

def validate_column_alignment(
    column_offsets,
    hard_limit=HARD_ABSOLUTE_OFFSET,
    max_local_jump=HARD_LOCAL_JUMP,
):
    """
    Mobile-photo policy:

    Normal gradual unevenness is ACCEPTED and corrected automatically.

    Reject only genuinely unsafe geometry:
      - an anchor hits ~20 px correction
      - neighboring calibration anchors suddenly jump ~16 px

    This is intentionally tolerant for handheld captures.
    """

    warnings = []

    for column_index, data in (
        column_offsets.items()
    ):
        anchors = data.get(
            "anchors",
            [],
        )

        if not anchors:
            continue

        for anchor in anchors:

            dx = abs(
                float(
                    anchor.get(
                        "dx",
                        0.0,
                    )
                )
            )

            dy = abs(
                float(
                    anchor.get(
                        "dy",
                        0.0,
                    )
                )
            )

            if (
                dx >= hard_limit
                or dy >= hard_limit
            ):
                raise ValueError(
                    "OMR photo is too distorted for reliable evaluation. "
                    "Keep the complete sheet visible and take the photo again."
                )

        for first, second in zip(
            anchors[:-1],
            anchors[1:],
        ):
            dx_jump = abs(
                float(
                    second[
                        "dx"
                    ]
                )
                -
                float(
                    first[
                        "dx"
                    ]
                )
            )

            dy_jump = abs(
                float(
                    second[
                        "dy"
                    ]
                )
                -
                float(
                    first[
                        "dy"
                    ]
                )
            )

            if (
                dx_jump > max_local_jump
                or dy_jump > max_local_jump
            ):
                # Do not immediately reject a single jump; record it.
                warnings.append(
                    {
                        "column":
                            int(
                                column_index
                                +
                                1
                            ),

                        "from_row":
                            int(
                                first[
                                    "row"
                                ]
                            ),

                        "to_row":
                            int(
                                second[
                                    "row"
                                ]
                            ),

                        "dx_jump":
                            round(
                                dx_jump,
                                2,
                            ),

                        "dy_jump":
                            round(
                                dy_jump,
                                2,
                            ),
                    }
                )

    # Only reject if there are several independent severe jumps.
    if len(
        warnings
    ) >= 4:
        raise ValueError(
            "OMR photo has several severe local distortions. "
            "Please flatten the sheet slightly and take the photo again."
        )

    return {
        "status":
            "auto_corrected",

        "warnings":
            warnings,
    }


# ============================================================
# RUNTIME COORDINATE GENERATION
# ============================================================

def generate_calibrated_bubble_coordinates(
    template,
    column_offsets=None,
):
    """
    Generate per-row runtime coordinates.

    Each question row receives an interpolated dx/dy correction from the
    dense mobile-photo calibration anchors.
    """

    columns = template[
        "columns"
    ]

    options = template[
        "options"
    ]

    y_positions = template[
        "question_y_positions"
    ]

    questions_per_column = int(
        template[
            "questions_per_column"
        ]
    )

    if column_offsets is None:
        column_offsets = {}

    coordinates = {}

    for column_index, column in enumerate(
        columns
    ):
        data = column_offsets.get(
            column_index,
            {},
        )

        anchors = data.get(
            "anchors",
            [],
        )

        if not anchors:
            dx = float(
                data.get(
                    "dx",
                    0.0,
                )
            )

            dy = float(
                data.get(
                    "dy",
                    0.0,
                )
            )

            anchors = [
                {
                    "row":
                        0,

                    "dx":
                        dx,

                    "dy":
                        dy,
                },

                {
                    "row":
                        len(
                            y_positions
                        )
                        -
                        1,

                    "dx":
                        dx,

                    "dy":
                        dy,
                },
            ]

        for row_index, y in enumerate(
            y_positions
        ):
            dx, dy = _interpolate_offset(
                row_index,
                anchors,
            )

            question = (
                column_index
                *
                questions_per_column
                +
                row_index
                +
                1
            )

            coordinates[
                question
            ] = {}

            for option in options:
                coordinates[
                    question
                ][
                    option
                ] = (
                    int(
                        round(
                            float(
                                column[
                                    option
                                ]
                            )
                            +
                            dx
                        )
                    ),

                    int(
                        round(
                            float(
                                y
                            )
                            +
                            dy
                        )
                    ),
                )

    return coordinates


# ============================================================
# DEBUG IMAGE
# ============================================================

def draw_calibration_debug(
    corrected_image,
    template,
    column_offsets,
):
    """
    Green circles:
        exact runtime coordinates after dense mobile-photo correction.

    Red lines/text:
        local calibration anchors.
    """

    debug = corrected_image.copy()

    coordinates = (
        generate_calibrated_bubble_coordinates(
            template,
            column_offsets,
        )
    )

    # Draw all calibrated bubble centers.
    for question, option_map in (
        coordinates.items()
    ):
        for option, (
            x,
            y,
        ) in option_map.items():
            cv2.circle(
                debug,
                (
                    int(
                        x
                    ),
                    int(
                        y
                    ),
                ),
                7,
                (
                    0,
                    255,
                    0,
                ),
                2,
            )

    # Draw anchor paths per response column.
    for column_index, data in (
        column_offsets.items()
    ):
        anchors = data.get(
            "anchors",
            [],
        )

        if not anchors:
            continue

        column = template[
            "columns"
        ][
            column_index
        ]

        options = template[
            "options"
        ]

        # Midpoint of A..D x positions as anchor-path visualization.
        base_x = float(
            np.mean(
                [
                    float(
                        column[
                            option
                        ]
                    )
                    for option
                    in options
                ]
            )
        )

        points = []

        for anchor in anchors:

            row = int(
                anchor[
                    "row"
                ]
            )

            y = float(
                template[
                    "question_y_positions"
                ][
                    row
                ]
            )

            x = (
                base_x
                +
                float(
                    anchor[
                        "dx"
                    ]
                )
            )

            y = (
                y
                +
                float(
                    anchor[
                        "dy"
                    ]
                )
            )

            points.append(
                (
                    int(
                        round(
                            x
                        )
                    ),
                    int(
                        round(
                            y
                        )
                    ),
                )
            )

        if len(
            points
        ) >= 2:
            cv2.polylines(
                debug,
                [
                    np.array(
                        points,
                        dtype=np.int32,
                    ).reshape(
                        -1,
                        1,
                        2,
                    )
                ],
                False,
                (
                    0,
                    0,
                    255,
                ),
                2,
            )

    # Compact text summary.
    y_text = 28

    for column_index in sorted(
        column_offsets
    ):
        data = column_offsets[
            column_index
        ]

        text = (
            f"C{column_index + 1} "
            f"median dx={data.get('dx', 0)} "
            f"dy={data.get('dy', 0)} "
            f"anchors={len(data.get('anchors', []))}"
        )

        cv2.putText(
            debug,
            text,
            (
                18,
                y_text,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        y_text += 26

    return debug

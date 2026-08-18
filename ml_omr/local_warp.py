from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# MOBILE LOCAL-WARP SETTINGS
# ============================================================

# Search only around already-calibrated coordinates.
LOCAL_SEARCH_X = 5
LOCAL_SEARCH_Y = 5
PATCH_RADIUS = 11

# Use every Nth question row as a local warp anchor.
ANCHOR_STEP = 5

# Reject only extreme outliers; normal handheld warping is expected.
MAX_ANCHOR_OFFSET = 14.0

# Final smoothed local warp is clamped to this range.
MAX_APPLIED_DX = 10.0
MAX_APPLIED_DY = 10.0


# ============================================================
# RING SCORE
# ============================================================

def _ring_score(
    gray,
    cx,
    cy,
    radius=PATCH_RADIUS,
):
    h, w = gray.shape[:2]

    x1 = max(
        0,
        int(cx - radius),
    )
    y1 = max(
        0,
        int(cy - radius),
    )
    x2 = min(
        w,
        int(cx + radius + 1),
    )
    y2 = min(
        h,
        int(cy + radius + 1),
    )

    patch = gray[
        y1:y2,
        x1:x2,
    ]

    if (
        patch.shape[0] < 17
        or patch.shape[1] < 17
    ):
        return -1e9

    ph, pw = patch.shape[:2]
    yy, xx = np.ogrid[
        :ph,
        :pw,
    ]

    pcx = (
        pw - 1
    ) / 2.0
    pcy = (
        ph - 1
    ) / 2.0

    rr = np.sqrt(
        (xx - pcx) ** 2
        +
        (yy - pcy) ** 2
    )

    ring = (
        (rr >= radius * 0.48)
        &
        (rr <= radius * 0.82)
    )

    outer = (
        (rr >= radius * 0.88)
        &
        (rr <= radius * 1.00)
    )

    if not np.any(
        ring
    ):
        return -1e9

    ring_mean = float(
        np.mean(
            patch[
                ring
            ]
        )
    )

    outer_mean = (
        float(
            np.mean(
                patch[
                    outer
                ]
            )
        )
        if np.any(
            outer
        )
        else 230.0
    )

    # Printed ring should be darker than surrounding paper.
    return float(
        outer_mean
        -
        ring_mean
    )


def _find_local_center(
    gray,
    x,
    y,
    search_x=LOCAL_SEARCH_X,
    search_y=LOCAL_SEARCH_Y,
):
    base_x = int(
        round(
            x
        )
    )
    base_y = int(
        round(
            y
        )
    )

    best_x = base_x
    best_y = base_y
    best_score = -1e9

    for dy in range(
        -search_y,
        search_y + 1,
    ):
        for dx in range(
            -search_x,
            search_x + 1,
    ):
            cx = base_x + dx
            cy = base_y + dy

            score = _ring_score(
                gray,
                cx,
                cy,
            )

            # Small movement penalty keeps search stable.
            score -= abs(
                dx
            ) * 0.20

            score -= abs(
                dy
            ) * 0.35

            if score > best_score:
                best_score = score
                best_x = cx
                best_y = cy

    return (
        best_x,
        best_y,
        float(
            best_score
        ),
    )


# ============================================================
# ROBUST FITTING HELPERS
# ============================================================

def _robust_median(
    values,
):
    if not values:
        return 0.0

    arr = np.asarray(
        values,
        dtype=np.float32,
    )

    median = float(
        np.median(
            arr
        )
    )

    deviation = np.abs(
        arr
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
            2.0,
            3.5 * mad,
        )
    )

    filtered = arr[
        keep
    ]

    if filtered.size == 0:
        return median

    return float(
        np.median(
            filtered
        )
    )


def _smooth_series(
    values,
    radius=1,
):
    values = list(
        values
    )

    if not values:
        return []

    out = []

    for i in range(
        len(
            values
        )
    ):
        start = max(
            0,
            i - radius,
        )

        end = min(
            len(
                values
            ),
            i + radius + 1,
        )

        out.append(
            float(
                np.median(
                    values[
                        start:end
                    ]
                )
            )
        )

    return out


def _interp(
    row,
    anchors,
    key,
):
    if not anchors:
        return 0.0

    anchors = sorted(
        anchors,
        key=lambda item:
            item[
                "row"
            ],
    )

    if row <= anchors[0]["row"]:
        return float(
            anchors[0][
                key
            ]
        )

    if row >= anchors[-1]["row"]:
        return float(
            anchors[-1][
                key
            ]
        )

    for left, right in zip(
        anchors[:-1],
        anchors[1:],
    ):
        if (
            left["row"]
            <= row
            <= right["row"]
        ):
            span = max(
                1,
                right["row"]
                -
                left["row"],
            )

            t = (
                row
                -
                left["row"]
            ) / float(
                span
            )

            return (
                float(
                    left[
                        key
                    ]
                )
                +
                t
                *
                (
                    float(
                        right[
                            key
                        ]
                    )
                    -
                    float(
                        left[
                            key
                        ]
                    )
                )
            )

    return 0.0


# ============================================================
# LOCAL WARP MODEL
# ============================================================

def build_local_warp_model(
    gray,
    coordinates,
    template,
):
    """
    Build a non-linear row-wise correction model.

    Important:
    - coordinates are already after global/canonical + column calibration.
    - this stage measures actual printed bubble-ring centers.
    - every response column gets many anchors down the page.
    - dx and dy are smoothed and interpolated row-by-row.

    The template JSON is never changed.
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(
            gray,
            cv2.COLOR_BGR2GRAY,
        )

    options = template[
        "options"
    ]

    questions_per_column = int(
        template[
            "questions_per_column"
        ]
    )

    rows_per_column = len(
        template[
            "question_y_positions"
        ]
    )

    models = {}

    for column_index in range(
        len(
            template[
                "columns"
            ]
        )
    ):
        anchor_rows = list(
            range(
                0,
                rows_per_column,
                ANCHOR_STEP,
            )
        )

        if (
            rows_per_column - 1
            not in anchor_rows
        ):
            anchor_rows.append(
                rows_per_column - 1
            )

        anchors = []

        for anchor_row in anchor_rows:
            dx_values = []
            dy_values = []
            scores = []

            # Use a small vertical band around each anchor row.
            for row_index in range(
                max(
                    0,
                    anchor_row - 1,
                ),
                min(
                    rows_per_column,
                    anchor_row + 2,
                ),
            ):
                question = (
                    column_index
                    *
                    questions_per_column
                    +
                    row_index
                    +
                    1
                )

                option_map = coordinates.get(
                    question,
                    {},
                )

                for option in options:
                    if option not in option_map:
                        continue

                    x, y = option_map[
                        option
                    ]

                    (
                        rx,
                        ry,
                        score,
                    ) = _find_local_center(
                        gray,
                        x,
                        y,
                    )

                    dx = float(
                        rx
                        -
                        float(
                            x
                        )
                    )
                    dy = float(
                        ry
                        -
                        float(
                            y
                        )
                    )

                    if (
                        abs(
                            dx
                        )
                        <=
                        MAX_ANCHOR_OFFSET
                        and
                        abs(
                            dy
                        )
                        <=
                        MAX_ANCHOR_OFFSET
                    ):
                        dx_values.append(
                            dx
                        )
                        dy_values.append(
                            dy
                        )
                        scores.append(
                            score
                        )

            dx = _robust_median(
                dx_values
            )

            dy = _robust_median(
                dy_values
            )

            anchors.append(
                {
                    "row":
                        int(
                            anchor_row
                        ),

                    "dx":
                        round(
                            dx,
                            3,
                        ),

                    "dy":
                        round(
                            dy,
                            3,
                        ),

                    "mean_score":
                        round(
                            float(
                                np.mean(
                                    scores
                                )
                            )
                            if scores
                            else 0.0,
                            3,
                        ),

                    "sample_count":
                        int(
                            len(
                                dx_values
                            )
                        ),
                }
            )

        # Smooth independently in x and y so one noisy anchor cannot bend
        # the entire column.
        dx_smoothed = _smooth_series(
            [
                a[
                    "dx"
                ]
                for a
                in anchors
            ],
            radius=1,
        )

        dy_smoothed = _smooth_series(
            [
                a[
                    "dy"
                ]
                for a
                in anchors
            ],
            radius=1,
        )

        for anchor, dx, dy in zip(
            anchors,
            dx_smoothed,
            dy_smoothed,
        ):
            anchor[
                "raw_dx"
            ] = anchor[
                "dx"
            ]

            anchor[
                "raw_dy"
            ] = anchor[
                "dy"
            ]

            anchor[
                "dx"
            ] = round(
                float(
                    np.clip(
                        dx,
                        -MAX_APPLIED_DX,
                        MAX_APPLIED_DX,
                    )
                ),
                3,
            )

            anchor[
                "dy"
            ] = round(
                float(
                    np.clip(
                        dy,
                        -MAX_APPLIED_DY,
                        MAX_APPLIED_DY,
                    )
                ),
                3,
            )

        models[
            column_index
        ] = {
            "anchors":
                anchors,

            "median_dx":
                round(
                    float(
                        np.median(
                            [
                                a[
                                    "dx"
                                ]
                                for a
                                in anchors
                            ]
                        )
                    ),
                    3,
                ),

            "median_dy":
                round(
                    float(
                        np.median(
                            [
                                a[
                                    "dy"
                                ]
                                for a
                                in anchors
                            ]
                        )
                    ),
                    3,
                ),
        }

    return models


def apply_local_warp(
    coordinates,
    template,
    warp_model,
):
    """
    Apply row-wise non-linear correction to every A/B/C/D bubble.
    """

    questions_per_column = int(
        template[
            "questions_per_column"
        ]
    )

    warped = {}

    for question, option_map in coordinates.items():
        q = int(
            question
        )

        column_index = (
            q - 1
        ) // questions_per_column

        row_index = (
            q - 1
        ) % questions_per_column

        model = warp_model.get(
            column_index,
            {},
        )

        anchors = model.get(
            "anchors",
            [],
        )

        dx = _interp(
            row_index,
            anchors,
            "dx",
        )

        dy = _interp(
            row_index,
            anchors,
            "dy",
        )

        warped[
            question
        ] = {}

        for option, (
            x,
            y,
        ) in option_map.items():
            warped[
                question
            ][
                option
            ] = (
                float(
                    x
                )
                +
                dx,

                float(
                    y
                )
                +
                dy,
            )

    return warped


# ============================================================
# DEBUG
# ============================================================

def draw_local_warp_debug(
    image,
    template,
    original_coordinates,
    warped_coordinates,
    warp_model,
):
    """
    Blue  = pre-local-warp calibrated coordinate
    Green = final local-warp coordinate
    Red path = row-wise local warp model through each column
    """

    debug = image.copy()

    for question, option_map in warped_coordinates.items():
        original_map = original_coordinates[
            question
        ]

        for option, (
            wx,
            wy,
        ) in option_map.items():
            ox, oy = original_map[
                option
            ]

            cv2.circle(
                debug,
                (
                    int(
                        round(
                            ox
                        )
                    ),
                    int(
                        round(
                            oy
                        )
                    ),
                ),
                4,
                (
                    255,
                    0,
                    0,
                ),
                1,
            )

            cv2.circle(
                debug,
                (
                    int(
                        round(
                            wx
                        )
                    ),
                    int(
                        round(
                            wy
                        )
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

    questions_per_column = int(
        template[
            "questions_per_column"
        ]
    )

    options = template[
        "options"
    ]

    for column_index, model in warp_model.items():
        points = []

        start_question = (
            column_index
            *
            questions_per_column
            +
            1
        )

        for anchor in model.get(
            "anchors",
            [],
        ):
            row = int(
                anchor[
                    "row"
                ]
            )

            question = (
                start_question
                +
                row
            )

            option_map = warped_coordinates.get(
                question
            )

            if not option_map:
                continue

            xs = [
                option_map[
                    option
                ][0]
                for option
                in options
            ]

            ys = [
                option_map[
                    option
                ][1]
                for option
                in options
            ]

            points.append(
                (
                    int(
                        round(
                            np.mean(
                                xs
                            )
                        )
                    ),
                    int(
                        round(
                            np.mean(
                                ys
                            )
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
                    np.asarray(
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

    y_text = 25

    for column_index in sorted(
        warp_model
    ):
        model = warp_model[
            column_index
        ]

        text = (
            f"LocalWarp C{column_index + 1} "
            f"dx={model.get('median_dx', 0)} "
            f"dy={model.get('median_dy', 0)} "
            f"anchors={len(model.get('anchors', []))}"
        )

        cv2.putText(
            debug,
            text,
            (
                15,
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

        y_text += 25

    return debug

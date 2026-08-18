from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

ROI_MARGIN_X = 24
ROI_MARGIN_Y = 18

MIN_SIZE = 10
MAX_SIZE = 28
MIN_AREA = 35
MAX_AREA = 700
MIN_CIRCULARITY = 0.18

MAX_INITIAL_MATCH_DISTANCE = 17.0

# Normal mobile-photo rows may need a larger vertical correction because
# the paper can curve locally even after the global homography.
MAX_DIRECT_PIN_DX = 7.0
MAX_DIRECT_PIN_DY = 10.0
MAX_FINAL_DX_FROM_INPUT = 8.0
MAX_FINAL_DY_FROM_INPUT = 10.0

# Keep the final row of each 45-question block tightly protected so it
# cannot jump onto the bottom border / signature-area line.
LAST_ROW_DIRECT_PIN_DY = 6.0
LAST_ROW_FINAL_DY_FROM_INPUT = 6.0

# Conservative rescue for the final row of a 45-question block.
# It activates only when >=3 detected bubble candidates agree on the same
# vertical offset. Otherwise the original ±6 px protection remains.
LAST_ROW_RESCUE_DY = 10.0
LAST_ROW_RESCUE_MIN_PINS = 3
LAST_ROW_RESCUE_MAX_SPREAD = 4.0
MIN_MATCHES_FOR_GRID = 12

# Residual correction is limited so a false contour cannot bend the grid badly.
MAX_ROW_RESIDUAL_X = 7.0
MAX_ROW_RESIDUAL_Y = 7.0


# ============================================================
# CANDIDATE DETECTION
# ============================================================

def _prepare_binary(gray_roi):
    if gray_roi.ndim == 3:
        gray_roi = cv2.cvtColor(
            gray_roi,
            cv2.COLOR_BGR2GRAY,
        )

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(6, 6),
    )

    norm = clahe.apply(
        gray_roi
    )

    blur = cv2.GaussianBlur(
        norm,
        (3, 3),
        0,
    )

    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    return binary


def _candidate_score(
    contour,
    x,
    y,
    w,
    h,
):
    area = float(
        cv2.contourArea(
            contour
        )
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    circularity = (
        4.0
        *
        np.pi
        *
        area
        /
        (
            perimeter
            *
            perimeter
            +
            1e-6
        )
    )

    size = (
        w
        +
        h
    ) / 2.0

    # Bubbles are close to ~20-22 px diameter on canonical sheet.
    size_score = max(
        0.0,
        1.0
        -
        abs(
            size
            -
            20.0
        )
        /
        12.0,
    )

    return float(
        circularity
        *
        2.0
        +
        size_score
    )


def _deduplicate_candidates(
    candidates,
    min_distance=5.0,
):
    candidates = sorted(
        candidates,
        key=lambda item:
            item[
                "score"
            ],
        reverse=True,
    )

    kept = []

    for candidate in candidates:
        cx = candidate[
            "x"
        ]
        cy = candidate[
            "y"
        ]

        duplicate = False

        for previous in kept:
            distance = float(
                np.hypot(
                    cx
                    -
                    previous[
                        "x"
                    ],
                    cy
                    -
                    previous[
                        "y"
                    ],
                )
            )

            if distance < min_distance:
                duplicate = True
                break

        if not duplicate:
            kept.append(
                candidate
            )

    return kept


def detect_circle_candidates(
    gray,
    roi,
):
    """
    Detect actual printed bubble-ring / filled-bubble candidates in a response block.

    roi = (x1, y1, x2, y2)
    Returned coordinates are full-image coordinates.
    """

    x1, y1, x2, y2 = [
        int(
            round(
                value
            )
        )
        for value
        in roi
    ]

    h, w = gray.shape[:2]

    x1 = max(
        0,
        x1,
    )
    y1 = max(
        0,
        y1,
    )
    x2 = min(
        w,
        x2,
    )
    y2 = min(
        h,
        y2,
    )

    crop = gray[
        y1:y2,
        x1:x2,
    ]

    binary = _prepare_binary(
        crop
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(
            contour
        )

        if (
            bw < MIN_SIZE
            or bh < MIN_SIZE
            or bw > MAX_SIZE
            or bh > MAX_SIZE
        ):
            continue

        aspect = (
            float(
                bw
            )
            /
            max(
                1.0,
                float(
                    bh
                ),
            )
        )

        if (
            aspect < 0.55
            or aspect > 1.45
        ):
            continue

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if (
            area < MIN_AREA
            or area > MAX_AREA
        ):
            continue

        perimeter = float(
            cv2.arcLength(
                contour,
                True,
            )
        )

        if perimeter <= 0:
            continue

        circularity = (
            4.0
            *
            np.pi
            *
            area
            /
            (
                perimeter
                *
                perimeter
                +
                1e-6
            )
        )

        if circularity < MIN_CIRCULARITY:
            continue

        center_x = (
            x1
            +
            bx
            +
            bw / 2.0
        )

        center_y = (
            y1
            +
            by
            +
            bh / 2.0
        )

        score = _candidate_score(
            contour,
            bx,
            by,
            bw,
            bh,
        )

        candidates.append(
            {
                "x":
                    float(
                        center_x
                    ),

                "y":
                    float(
                        center_y
                    ),

                "w":
                    int(
                        bw
                    ),

                "h":
                    int(
                        bh
                    ),

                "score":
                    float(
                        score
                    ),
            }
        )

    return _deduplicate_candidates(
        candidates
    )


# ============================================================
# GRID FITTING
# ============================================================

def _block_question_range(
    column_index,
    template,
):
    qpc = int(
        template[
            "questions_per_column"
        ]
    )

    start = (
        column_index
        *
        qpc
        +
        1
    )

    end = (
        start
        +
        qpc
    )

    return (
        start,
        end,
    )


def _block_roi(
    coordinates,
    column_index,
    template,
):
    start, end = _block_question_range(
        column_index,
        template,
    )

    xs = []
    ys = []

    for question in range(
        start,
        end,
    ):
        option_map = coordinates[
            question
        ]

        for x, y in option_map.values():
            xs.append(
                float(
                    x
                )
            )
            ys.append(
                float(
                    y
                )
            )

    return (
        min(
            xs
        )
        -
        ROI_MARGIN_X,

        min(
            ys
        )
        -
        ROI_MARGIN_Y,

        max(
            xs
        )
        +
        ROI_MARGIN_X,

        max(
            ys
        )
        +
        ROI_MARGIN_Y,
    )


def _expected_points(
    coordinates,
    column_index,
    template,
):
    start, end = _block_question_range(
        column_index,
        template,
    )

    options = template[
        "options"
    ]

    expected = []

    for question in range(
        start,
        end,
    ):
        row_index = (
            question
            -
            start
        )

        for option_index, option in enumerate(
            options
        ):
            x, y = coordinates[
                question
            ][
                option
            ]

            expected.append(
                {
                    "question":
                        int(
                            question
                        ),

                    "row":
                        int(
                            row_index
                        ),

                    "option":
                        option,

                    "option_index":
                        int(
                            option_index
                        ),

                    "x":
                        float(
                            x
                        ),

                    "y":
                        float(
                            y
                        ),
                }
            )

    return expected


def _initial_correspondences(
    expected,
    candidates,
    max_match_distance=MAX_INITIAL_MATCH_DISTANCE,
):
    """
    Greedy nearest-neighbour assignment from expected lattice to detected circles.
    """

    if not candidates:
        return []

    candidate_xy = np.asarray(
        [
            [
                item[
                    "x"
                ],
                item[
                    "y"
                ],
            ]
            for item
            in candidates
        ],
        dtype=np.float32,
    )

    possible = []

    for expected_index, point in enumerate(
        expected
    ):
        diff = (
            candidate_xy
            -
            np.asarray(
                [
                    point[
                        "x"
                    ],
                    point[
                        "y"
                    ],
                ],
                dtype=np.float32,
            )
        )

        distances = np.sqrt(
            np.sum(
                diff
                *
                diff,
                axis=1,
            )
        )

        candidate_index = int(
            np.argmin(
                distances
            )
        )

        distance = float(
            distances[
                candidate_index
            ]
        )

        if distance <= max_match_distance:
            possible.append(
                (
                    distance,
                    expected_index,
                    candidate_index,
                )
            )

    # Ensure one candidate is not assigned to multiple expected bubbles.
    possible.sort(
        key=lambda item:
            item[
                0
            ]
    )

    used_expected = set()
    used_candidates = set()

    correspondences = []

    for (
        distance,
        expected_index,
        candidate_index,
    ) in possible:
        if expected_index in used_expected:
            continue

        if candidate_index in used_candidates:
            continue

        used_expected.add(
            expected_index
        )

        used_candidates.add(
            candidate_index
        )

        point = expected[
            expected_index
        ]

        candidate = candidates[
            candidate_index
        ]

        correspondences.append(
            {
                **point,

                "detected_x":
                    float(
                        candidate[
                            "x"
                        ]
                    ),

                "detected_y":
                    float(
                        candidate[
                            "y"
                        ]
                    ),

                "distance":
                    float(
                        distance
                    ),
            }
        )

    return correspondences


def _fit_affine(
    correspondences,
):
    if len(
        correspondences
    ) < 6:
        return None, None

    src = np.asarray(
        [
            [
                item[
                    "x"
                ],
                item[
                    "y"
                ],
            ]
            for item
            in correspondences
        ],
        dtype=np.float32,
    )

    dst = np.asarray(
        [
            [
                item[
                    "detected_x"
                ],
                item[
                    "detected_y"
                ],
            ]
            for item
            in correspondences
        ],
        dtype=np.float32,
    )

    matrix, inliers = cv2.estimateAffine2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=2500,
        confidence=0.995,
        refineIters=25,
    )

    return (
        matrix,
        inliers,
    )


def _apply_affine_point(
    matrix,
    x,
    y,
):
    if matrix is None:
        return (
            float(
                x
            ),
            float(
                y
            ),
        )

    point = np.asarray(
        [
            x,
            y,
            1.0,
        ],
        dtype=np.float64,
    )

    out = matrix @ point

    return (
        float(
            out[
                0
            ]
        ),
        float(
            out[
                1
            ]
        ),
    )


def _robust_row_residuals(
    correspondences,
    matrix,
    rows_per_column,
):
    """
    After affine fit, measure remaining local residual per question row.
    Missing rows are interpolated.
    """

    row_dx = {
        row: []
        for row
        in range(
            rows_per_column
        )
    }

    row_dy = {
        row: []
        for row
        in range(
            rows_per_column
        )
    }

    for item in correspondences:
        predicted_x, predicted_y = _apply_affine_point(
            matrix,
            item[
                "x"
            ],
            item[
                "y"
            ],
        )

        dx = (
            item[
                "detected_x"
            ]
            -
            predicted_x
        )

        dy = (
            item[
                "detected_y"
            ]
            -
            predicted_y
        )

        if (
            abs(
                dx
            )
            <=
            MAX_ROW_RESIDUAL_X
            and
            abs(
                dy
            )
            <=
            MAX_ROW_RESIDUAL_Y
        ):
            row_dx[
                item[
                    "row"
                ]
            ].append(
                dx
            )

            row_dy[
                item[
                    "row"
                ]
            ].append(
                dy
            )

    known_rows = []
    known_dx = []
    known_dy = []

    for row in range(
        rows_per_column
    ):
        if (
            row_dx[
                row
            ]
            and
            row_dy[
                row
            ]
        ):
            known_rows.append(
                row
            )

            known_dx.append(
                float(
                    np.median(
                        row_dx[
                            row
                        ]
                    )
                )
            )

            known_dy.append(
                float(
                    np.median(
                        row_dy[
                            row
                        ]
                    )
                )
            )

    if not known_rows:
        return {
            row:
                (
                    0.0,
                    0.0,
                )
            for row
            in range(
                rows_per_column
            )
        }

    # Smooth known residuals before interpolation.
    if len(
        known_dx
    ) >= 3:
        smoothed_dx = []

        smoothed_dy = []

        for i in range(
            len(
                known_rows
            )
        ):
            start = max(
                0,
                i - 1,
            )

            end = min(
                len(
                    known_rows
                ),
                i + 2,
            )

            smoothed_dx.append(
                float(
                    np.median(
                        known_dx[
                            start:end
                        ]
                    )
                )
            )

            smoothed_dy.append(
                float(
                    np.median(
                        known_dy[
                            start:end
                        ]
                    )
                )
            )

        known_dx = smoothed_dx
        known_dy = smoothed_dy

    all_rows = np.arange(
        rows_per_column,
        dtype=np.float32,
    )

    interp_dx = np.interp(
        all_rows,
        np.asarray(
            known_rows,
            dtype=np.float32,
        ),
        np.asarray(
            known_dx,
            dtype=np.float32,
        ),
    )

    interp_dy = np.interp(
        all_rows,
        np.asarray(
            known_rows,
            dtype=np.float32,
        ),
        np.asarray(
            known_dy,
            dtype=np.float32,
        ),
    )

    return {
        int(
            row
        ):
            (
                float(
                    np.clip(
                        interp_dx[
                            row
                        ],
                        -MAX_ROW_RESIDUAL_X,
                        MAX_ROW_RESIDUAL_X,
                    )
                ),
                float(
                    np.clip(
                        interp_dy[
                            row
                        ],
                        -MAX_ROW_RESIDUAL_Y,
                        MAX_ROW_RESIDUAL_Y,
                    )
                ),
            )
        for row
        in range(
            rows_per_column
        )
    }


def fit_response_grid(
    gray,
    coordinates,
    template,
):
    """
    Detect actual response-bubble circles and fit the known 4 x 45 lattice.

    IMPORTANT:
    If an expected bubble has a reliable detected candidate ("pin dot"),
    that detected candidate center becomes the FINAL center directly.

    The affine / row-residual model is used ONLY for bubbles whose actual
    circle could not be matched reliably.

    Returns:
        fitted_coordinates, debug_info
    """

    rows_per_column = len(
        template[
            "question_y_positions"
        ]
    )

    # KCET's 60-row response blocks have only ~21 px between rows.  Its
    # pin/match window must therefore be narrower than NEET's 45-row window
    # or a ring from the adjacent row can become a valid match.  Templates
    # without these settings retain the established NEET limits.
    max_initial_match_distance = float(
        template.get(
            "grid_max_initial_match_distance",
            MAX_INITIAL_MATCH_DISTANCE,
        )
    )
    max_direct_pin_dy = float(
        template.get(
            "grid_max_direct_pin_dy",
            MAX_DIRECT_PIN_DY,
        )
    )
    max_final_dy_from_input = float(
        template.get(
            "grid_max_final_dy_from_input",
            MAX_FINAL_DY_FROM_INPUT,
        )
    )
    last_row_rescue_dy = min(
        float(LAST_ROW_RESCUE_DY),
        max_direct_pin_dy,
        max_final_dy_from_input,
    )

    y_positions = template.get("question_y_positions", [])
    row_spacing = (
        float(y_positions[-1] - y_positions[0]) / float(len(y_positions) - 1)
        if len(y_positions) > 1
        else 28.0
    )
    outlier_min_dist = min(9.0, max(5.5, row_spacing * 0.30))
    min_pair_sep = min(10.0, max(6.0, row_spacing * 0.32))
    max_pair_sep = min(22.0, max(14.0, row_spacing * 0.75))

    fitted = {}
    debug_info = {}

    for column_index in range(
        len(
            template[
                "columns"
            ]
        )
    ):
        roi = _block_roi(
            coordinates,
            column_index,
            template,
        )

        candidates = detect_circle_candidates(
            gray,
            roi,
        )

        expected = _expected_points(
            coordinates,
            column_index,
            template,
        )

        correspondences = _initial_correspondences(
            expected,
            candidates,
            max_match_distance=max_initial_match_distance,
        )

        matrix = None
        inliers = None

        if len(
            correspondences
        ) >= MIN_MATCHES_FOR_GRID:
            matrix, inliers = _fit_affine(
                correspondences
            )

        start_question, end_question = (
            _block_question_range(
                column_index,
                template,
            )
        )

        # Direct lookup of real detected pin centers.
        # Keep broad correspondences for RANSAC fitting, but only "pin lock"
        # when the detected center is close in BOTH x and y.
        #
        # FINAL-ROW CONSENSUS RESCUE:
        # Normally the last row stays protected at ±6 px.  If at least three
        # detected circles from that final row independently agree on nearly
        # the same vertical offset, temporarily allow ±10 px for that row.
        # This is designed for curved/mobile captures where the whole last
        # response row shifts together, while still rejecting isolated border
        # or signature-line candidates.
        last_row_rescue_enabled = False

        last_row_dy_candidates = []

        for item in correspondences:
            item_row = int(
                item.get(
                    "row",
                    -1,
                )
            )

            if item_row != rows_per_column - 1:
                continue

            dx_pin = (
                float(
                    item[
                        "detected_x"
                    ]
                )
                -
                float(
                    item[
                        "x"
                    ]
                )
            )

            dy_pin = (
                float(
                    item[
                        "detected_y"
                    ]
                )
                -
                float(
                    item[
                        "y"
                    ]
                )
            )

            if (
                abs(
                    dx_pin
                )
                <= MAX_DIRECT_PIN_DX
                and
                abs(
                    dy_pin
                )
                <= last_row_rescue_dy
            ):
                last_row_dy_candidates.append(
                    float(
                        dy_pin
                    )
                )

        if (
            len(
                last_row_dy_candidates
            )
            >= LAST_ROW_RESCUE_MIN_PINS
        ):
            last_row_dy_array = np.asarray(
                last_row_dy_candidates,
                dtype=np.float32,
            )

            if (
                float(
                    np.max(
                        last_row_dy_array
                    )
                    -
                    np.min(
                        last_row_dy_array
                    )
                )
                <= LAST_ROW_RESCUE_MAX_SPREAD
            ):
                last_row_rescue_enabled = True

        direct_match = {}

        for item in correspondences:
            dx_pin = (
                float(
                    item[
                        "detected_x"
                    ]
                )
                -
                float(
                    item[
                        "x"
                    ]
                )
            )

            dy_pin = (
                float(
                    item[
                        "detected_y"
                    ]
                )
                -
                float(
                    item[
                        "y"
                    ]
                )
            )

            item_row = int(
                item.get(
                    "row",
                    -1,
                )
            )

            direct_pin_dy_limit = (
                (
                    last_row_rescue_dy
                    if last_row_rescue_enabled
                    else LAST_ROW_DIRECT_PIN_DY
                )
                if item_row == rows_per_column - 1
                else max_direct_pin_dy
            )

            if (
                abs(
                    dx_pin
                )
                <=
                MAX_DIRECT_PIN_DX
                and
                abs(
                    dy_pin
                )
                <=
                direct_pin_dy_limit
            ):
                direct_match[
                    (
                        int(
                            item[
                                "question"
                            ]
                        ),
                        str(
                            item[
                                "option"
                            ]
                        ),
                    )
                ] = (
                    float(
                        item[
                            "detected_x"
                        ]
                    ),
                    float(
                        item[
                            "detected_y"
                        ]
                    ),
                    float(
                        item[
                            "distance"
                        ]
                    ),
                )

        # If affine fitting failed, still use direct matches wherever possible.
        if matrix is None:
            direct_used = 0

            for question in range(
                start_question,
                end_question,
            ):
                fitted[
                    question
                ] = {}

                for option, (
                    x,
                    y,
                ) in coordinates[
                    question
                ].items():
                    key = (
                        int(
                            question
                        ),
                        str(
                            option
                        ),
                    )

                    if key in direct_match:
                        dx, dy, _distance = direct_match[
                            key
                        ]

                        fitted[
                            question
                        ][
                            option
                        ] = (
                            dx,
                            dy,
                        )

                        direct_used += 1

                    else:
                        fitted[
                            question
                        ][
                            option
                        ] = (
                            float(
                                x
                            ),
                            float(
                                y
                            ),
                        )

            debug_info[
                column_index
            ] = {
                "status":
                    "direct_pin_fallback",

                "roi":
                    [
                        round(
                            float(
                                value
                            ),
                            2,
                        )
                        for value
                        in roi
                    ],

                "candidate_count":
                    len(
                        candidates
                    ),

                "match_count":
                    len(
                        correspondences
                    ),

                "direct_pin_count":
                    int(
                        direct_used
                    ),

                "inlier_count":
                    0,

                "affine":
                    None,
            }

            continue

        residuals = _robust_row_residuals(
            correspondences,
            matrix,
            rows_per_column,
        )

        direct_used = 0
        model_used = 0

        for question in range(
            start_question,
            end_question,
        ):
            row = (
                question
                -
                start_question
            )

            dx_residual, dy_residual = residuals[
                row
            ]

            final_dy_limit = (
                (
                    last_row_rescue_dy
                    if last_row_rescue_enabled
                    else LAST_ROW_FINAL_DY_FROM_INPUT
                )
                if row == rows_per_column - 1
                else max_final_dy_from_input
            )

            fitted[
                question
            ] = {}

            for option, (
                x,
                y,
            ) in coordinates[
                question
            ].items():
                key = (
                    int(
                        question
                    ),
                    str(
                        option
                    ),
                )

                # ------------------------------------------------
                # BEST CASE:
                # use the real detected printed-circle center.
                # ------------------------------------------------
                if key in direct_match:
                    pin_x, pin_y, distance = direct_match[
                        key
                    ]

                    fitted[
                        question
                    ][
                        option
                    ] = (
                        float(
                            np.clip(
                                pin_x,
                                float(x)
                                -
                                MAX_FINAL_DX_FROM_INPUT,
                                float(x)
                                +
                                MAX_FINAL_DX_FROM_INPUT,
                            )
                        ),
                        float(
                            np.clip(
                                pin_y,
                                float(y)
                                -
                                final_dy_limit,
                                float(y)
                                +
                                final_dy_limit,
                            )
                        ),
                    )

                    direct_used += 1

                    continue

                # ------------------------------------------------
                # FALLBACK:
                # no trustworthy pin for this bubble.
                # Use fitted lattice + local row residual.
                # ------------------------------------------------
                fx, fy = _apply_affine_point(
                    matrix,
                    float(
                        x
                    ),
                    float(
                        y
                    ),
                )

                final_x = (
                    fx
                    +
                    dx_residual
                )

                final_y = (
                    fy
                    +
                    dy_residual
                )

                fitted[
                    question
                ][
                    option
                ] = (
                    float(
                        np.clip(
                            final_x,
                            float(x)
                            -
                            MAX_FINAL_DX_FROM_INPUT,
                            float(x)
                            +
                            MAX_FINAL_DX_FROM_INPUT,
                        )
                    ),
                    float(
                        np.clip(
                            final_y,
                            float(y)
                            -
                            final_dy_limit,
                            float(y)
                            +
                            final_dy_limit,
                        )
                    ),
                )

                model_used += 1

            # ------------------------------------------------
            # PER-QUESTION ROW CONSISTENCY REPAIR
            # ------------------------------------------------
            # Mobile perspective / local pin matching can occasionally
            # snap one option onto the bubble from the neighboring row.
            #
            # Repair only a clear 3-vs-1 pattern:
            #   - at least 3 option centers form a tight vertical cluster
            #   - an outlier is >= 9 px away from that cluster
            #
            # Do NOT touch 2-vs-2 splits or already-consistent rows.
            # This keeps genuine local geometry while fixing obvious
            # single-option row jumps such as Q31/Q35/Q137-style cases.
            question_points = fitted[
                question
            ]

            option_y_values = {
                option: float(
                    point[1]
                )
                for option, point
                in question_points.items()
            }

            if len(
                option_y_values
            ) >= 4:
                y_values = np.array(
                    list(
                        option_y_values.values()
                    ),
                    dtype=np.float32,
                )

                median_y = float(
                    np.median(
                        y_values
                    )
                )

                cluster_options = [
                    option
                    for option, option_y
                    in option_y_values.items()
                    if abs(
                        option_y
                        -
                        median_y
                    ) <= 4.0
                ]

                if len(
                    cluster_options
                ) >= 3:
                    cluster_y = float(
                        np.median(
                            [
                                option_y_values[
                                    option
                                ]
                                for option
                                in cluster_options
                            ]
                        )
                    )

                    for option, option_y in list(
                        option_y_values.items()
                    ):
                        if (
                            option
                            not in cluster_options
                            and
                            abs(
                                option_y
                                -
                                cluster_y
                            ) >= outlier_min_dist
                        ):
                            original_x, original_y = coordinates[
                                question
                            ][
                                option
                            ]

                            current_x, _ = fitted[
                                question
                            ][
                                option
                            ]

                            repaired_y = float(
                                np.clip(
                                    cluster_y,
                                    float(
                                        original_y
                                    )
                                    -
                                    final_dy_limit,
                                    float(
                                        original_y
                                    )
                                    +
                                    final_dy_limit,
                                )
                            )

                            fitted[
                                question
                            ][
                                option
                            ] = (
                                float(
                                    current_x
                                ),
                                repaired_y,
                            )

                # --------------------------------------------
                # CONSERVATIVE 2-vs-2 ROW-SPLIT REPAIR
                # --------------------------------------------
                # Some mobile captures create a clean 2-vs-2 split where
                # two options snap slightly above the intended row and two
                # slightly below it. The earlier 3-vs-1 repair intentionally
                # leaves this untouched.
                #
                # Only repair when:
                #   * both pairs are internally tight (<= 3.5 px),
                #   * the pair centers are separated by 10..22 px, and
                #   * all four options are present.
                #
                # In that very specific case, use the calibrated/template
                # question-row Y as the common row center. X coordinates are
                # preserved. This targets cases like Q41/Q98 without changing
                # ordinary rows.
                repaired_y_values = {
                    option: float(
                        point[1]
                    )
                    for option, point
                    in fitted[
                        question
                    ].items()
                }

                if len(
                    repaired_y_values
                ) == 4:
                    sorted_items = sorted(
                        repaired_y_values.items(),
                        key=lambda item:
                            item[
                                1
                            ],
                    )

                    low_pair = sorted_items[
                        :2
                    ]

                    high_pair = sorted_items[
                        2:
                    ]

                    low_spread = abs(
                        low_pair[
                            1
                        ][
                            1
                        ]
                        -
                        low_pair[
                            0
                        ][
                            1
                        ]
                    )

                    high_spread = abs(
                        high_pair[
                            1
                        ][
                            1
                        ]
                        -
                        high_pair[
                            0
                        ][
                            1
                        ]
                    )

                    low_center = float(
                        np.mean(
                            [
                                item[
                                    1
                                ]
                                for item
                                in low_pair
                            ]
                        )
                    )

                    high_center = float(
                        np.mean(
                            [
                                item[
                                    1
                                ]
                                for item
                                in high_pair
                            ]
                        )
                    )

                    pair_separation = (
                        high_center
                        -
                        low_center
                    )

                    if (
                        low_spread <= 3.5
                        and
                        high_spread <= 3.5
                        and
                        min_pair_sep
                        <= pair_separation
                        <= max_pair_sep
                    ):
                        original_row_y = float(
                            np.median(
                                [
                                    float(
                                        coordinates[
                                            question
                                        ][
                                            option
                                        ][
                                            1
                                        ]
                                    )
                                    for option
                                    in fitted[
                                        question
                                    ].keys()
                                ]
                            )
                        )

                        for option, (
                            current_x,
                            _current_y,
                        ) in list(
                            fitted[
                                question
                            ].items()
                        ):
                            _original_x, original_y = coordinates[
                                question
                            ][
                                option
                            ]

                            split_repaired_y = float(
                                np.clip(
                                    original_row_y,
                                    float(
                                        original_y
                                    )
                                    -
                                    final_dy_limit,
                                    float(
                                        original_y
                                    )
                                    +
                                    final_dy_limit,
                                )
                            )

                            fitted[
                                question
                            ][
                                option
                            ] = (
                                float(
                                    current_x
                                ),
                                split_repaired_y,
                            )

        inlier_count = (
            int(
                inliers.sum()
            )
            if inliers is not None
            else 0
        )

        debug_info[
            column_index
        ] = {
            "status":
                "pin_locked_grid",

            "roi":
                [
                    round(
                        float(
                            value
                        ),
                        2,
                    )
                    for value
                    in roi
                ],

            "candidate_count":
                len(
                    candidates
                ),

            "match_count":
                len(
                    correspondences
                ),

            "direct_pin_count":
                int(
                    direct_used
                ),

            "model_fallback_count":
                int(
                    model_used
                ),

            "inlier_count":
                inlier_count,

            "affine":
                matrix.tolist(),

            "correspondences":
                correspondences,
        }

    return (
        fitted,
        debug_info,
    )


# ============================================================
# DEBUG DRAWING
# ============================================================

def draw_grid_detection_debug(
    image,
    template,
    input_coordinates,
    fitted_coordinates,
    debug_info,
):
    """
    Blue  = input calibrated coordinates
    Green = detected/fitted grid coordinates
    Yellow dots = detected candidate circles
    """

    debug = image.copy()

    # Draw input/fitted coordinates.
    for question, option_map in fitted_coordinates.items():
        before_map = input_coordinates[
            question
        ]

        for option, (
            fx,
            fy,
        ) in option_map.items():
            bx, by = before_map[
                option
            ]

            cv2.circle(
                debug,
                (
                    int(
                        round(
                            bx
                        )
                    ),
                    int(
                        round(
                            by
                        )
                    ),
                ),
                3,
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
                            fx
                        )
                    ),
                    int(
                        round(
                            fy
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

    # Re-detect candidates for visualization.
    gray = (
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
        if image.ndim == 3
        else image
    )

    for column_index in range(
        len(
            template[
                "columns"
            ]
        )
    ):
        roi = _block_roi(
            input_coordinates,
            column_index,
            template,
        )

        candidates = detect_circle_candidates(
            gray,
            roi,
        )

        for candidate in candidates:
            cv2.circle(
                debug,
                (
                    int(
                        round(
                            candidate[
                                "x"
                            ]
                        )
                    ),
                    int(
                        round(
                            candidate[
                                "y"
                            ]
                        )
                    ),
                ),
                2,
                (
                    0,
                    255,
                    255,
                ),
                -1,
            )

    y_text = 25

    for column_index in sorted(
        debug_info
    ):
        info = debug_info[
            column_index
        ]

        text = (
            f"Grid C{column_index + 1} "
            f"{info.get('status')} "
            f"candidates={info.get('candidate_count', 0)} "
            f"matches={info.get('match_count', 0)} "
            f"pins={info.get('direct_pin_count', 0)} "
            f"inliers={info.get('inlier_count', 0)}"
        )

        cv2.putText(
            debug,
            text,
            (
                15,
                y_text,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        y_text += 24

    return debug

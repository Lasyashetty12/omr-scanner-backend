"""
REFERENCE USAGE POLICY

The clean NEET reference image is used ONLY for geometric preprocessing:
- canonical orientation selection
- ORB fine registration
- optional high-confidence ECC fine registration

It is NEVER used to:
- decide whether a bubble is filled
- read answers
- read the paper code
- score marks

All answer/paper-code reading happens later from the corrected scan using
the JSON coordinates + runtime column calibration + ML/classical reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np



DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 2200

# Registration-block centres are the template's geometry authority.  Feature
# refinement may only correct very small residual error after that mapping;
# a larger warp would move valid bubble coordinates away from their bubbles.
MAX_FINE_ALIGNMENT_CORNER_ERROR = 24.0

# Canonical registration-mark centres in the user's clean NEET reference.
CANONICAL_MARKERS_1600_2200 = np.array(
    [
        [81.2, 78.3],       # TL
        [1522.0, 78.3],     # TR
        [1523.3, 2124.2],   # BR
        [79.9, 2120.4],     # BL
    ],
    dtype=np.float32,
)


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    ordered[0] = points[np.argmin(sums)]      # TL
    ordered[2] = points[np.argmax(sums)]      # BR
    ordered[1] = points[np.argmin(diffs)]     # TR
    ordered[3] = points[np.argmax(diffs)]     # BL

    return ordered


def _resize_for_detection(
    image: np.ndarray,
    max_side: int = 1500,
) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))

    if scale < 1.0:
        resized = cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    return image.copy(), 1.0


def _binary_dark(gray: np.ndarray) -> np.ndarray:
    """
    Produce a dark-object mask robust to uneven mobile lighting.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Global Otsu + adaptive threshold, then combine.
    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        9,
    )

    mask = cv2.bitwise_and(otsu, adaptive)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        ),
        iterations=1,
    )

    return mask


def _candidate_black_blocks(
    image: np.ndarray,
) -> list[Dict[str, Any]]:
    """
    Find compact dark filled rectangles/squares that could be registration marks.
    """
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mask = _binary_dark(gray)

    contours, _ = cv2.findContours(
        mask,
        # A dark desk/background can surround the bright page and otherwise
        # hide the inset black registration blocks from EXTERNAL retrieval.
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    h, w = gray.shape[:2]
    image_area = float(h * w)

    candidates: list[Dict[str, Any]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        # Registration blocks are visually large but still small relative to frame.
        if area < image_area * 0.00015:
            continue

        if area > image_area * 0.035:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        if bw < 10 or bh < 10:
            continue

        aspect = bw / float(bh)

        # Bottom marks can merge slightly with page rules, so allow some elongation.
        if not 0.45 <= aspect <= 2.2:
            continue

        rect_area = float(bw * bh)
        fill = area / max(rect_area, 1.0)

        if fill < 0.52:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True,
        )

        compactness = (
            4.0 * np.pi * area /
            max(perimeter * perimeter, 1.0)
        )

        cx = x + bw / 2.0
        cy = y + bh / 2.0

        candidates.append(
            {
                "center": np.array(
                    [cx, cy],
                    dtype=np.float32,
                ),
                "bbox": (x, y, bw, bh),
                "area": area,
                "fill": fill,
                "aspect": aspect,
                "vertices": len(approx),
                "compactness": float(compactness),
            }
        )

    return candidates


def _corner_region_score(
    candidate: Dict[str, Any],
    corner: str,
    width: int,
    height: int,
) -> float:
    cx, cy = candidate["center"]

    nx = cx / max(float(width), 1.0)
    ny = cy / max(float(height), 1.0)

    target = {
        "TL": (0.12, 0.10),
        "TR": (0.88, 0.10),
        "BR": (0.88, 0.90),
        "BL": (0.12, 0.90),
    }[corner]

    distance = np.hypot(
        nx - target[0],
        ny - target[1],
    )

    area_score = min(
        candidate["area"] / max(width * height * 0.004, 1.0),
        2.0,
    )

    square_score = max(
        0.0,
        1.0 - abs(
            np.log(
                max(candidate["aspect"], 1e-6)
            )
        ),
    )

    fill_score = candidate["fill"]

    vertex_score = (
        1.0
        if 4 <= candidate["vertices"] <= 8
        else 0.5
    )

    return (
        -distance * 8.0
        + area_score * 1.3
        + square_score * 1.5
        + fill_score * 1.6
        + vertex_score
    )


def _pick_corner_candidate(
    candidates: list[Dict[str, Any]],
    corner: str,
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    """
    Search a generous corner quadrant but reject centre-page content.
    """
    chosen = []

    for candidate in candidates:
        cx, cy = candidate["center"]
        nx = cx / float(width)
        ny = cy / float(height)

        if corner == "TL":
            inside = nx < 0.48 and ny < 0.50
        elif corner == "TR":
            inside = nx > 0.52 and ny < 0.50
        elif corner == "BR":
            inside = nx > 0.52 and ny > 0.50
        else:
            inside = nx < 0.48 and ny > 0.50

        if not inside:
            continue

        score = _corner_region_score(
            candidate,
            corner,
            width,
            height,
        )

        chosen.append(
            (
                score,
                candidate,
            )
        )

    if not chosen:
        return None

    chosen.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return chosen[0][1]


def _validate_marker_geometry(
    markers: np.ndarray,
    width: int,
    height: int,
) -> None:
    tl, tr, br, bl = order_points(markers)

    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)

    if min(top, bottom, left, right) < min(width, height) * 0.32:
        raise ValueError(
            "Registration markers are too close together. "
            "A wrong black object was probably selected."
        )

    polygon = np.array(
        [tl, tr, br, bl],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    area = abs(float(cv2.contourArea(polygon)))
    coverage = area / float(width * height)

    if coverage < 0.35:
        raise ValueError(
            "Registration-marker quadrilateral is too small."
        )

    # Opposite sides should not differ absurdly.
    if max(top, bottom) / max(min(top, bottom), 1.0) > 1.8:
        raise ValueError(
            "Top/bottom registration geometry is inconsistent."
        )

    if max(left, right) / max(min(left, right), 1.0) > 1.8:
        raise ValueError(
            "Left/right registration geometry is inconsistent."
        )


def _detect_registration_blocks_internal(
    small: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], list[Dict[str, Any]]]:
    h, w = small.shape[:2]
    candidates = _candidate_black_blocks(small)

    picked_dict = {}
    for corner in ("TL", "TR", "BR", "BL"):
        cand = _pick_corner_candidate(candidates, corner, w, h)
        if cand is not None:
            picked_dict[corner] = cand["center"]

    # Fallback recovery for missing corner blocks
    if "TL" in picked_dict and "TR" in picked_dict:
        tl = picked_dict["TL"]
        tr = picked_dict["TR"]
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        perp_x = -dy * 1.375
        perp_y = dx * 1.375

        if "BL" not in picked_dict:
            est_bl = np.array([tl[0] + perp_x, tl[1] + perp_y], dtype=np.float32)
            sub_cands = [c for c in candidates if c["center"][0] < w * 0.48 and c["center"][1] > h * 0.60]
            if sub_cands:
                best_cand = min(sub_cands, key=lambda c: np.linalg.norm(c["center"] - est_bl))
                picked_dict["BL"] = best_cand["center"]
            else:
                picked_dict["BL"] = est_bl

        if "BR" not in picked_dict:
            est_br = np.array([tr[0] + perp_x, tr[1] + perp_y], dtype=np.float32)
            sub_cands = [c for c in candidates if c["center"][0] > w * 0.52 and c["center"][1] > h * 0.60]
            if sub_cands:
                best_cand = min(sub_cands, key=lambda c: np.linalg.norm(c["center"] - est_br))
                picked_dict["BR"] = best_cand["center"]
            else:
                picked_dict["BR"] = est_br

    return picked_dict, candidates


def detect_registration_blocks(
    image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect the four large black registration blocks visible on the OMR sheet.
    Handles both portrait and landscape input photos automatically.

    Output order: TL, TR, BR, BL.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")

    small, scale = _resize_for_detection(
        image,
        max_side=1500,
    )

    h, w = small.shape[:2]

    # Primary attempt in natural orientation (0°)
    picked_dict, candidates = _detect_registration_blocks_internal(small)

    # If not all 4 blocks found in natural view, try remaining cardinal rotations
    if len(picked_dict) < 4:
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, "90_cw"),
            (cv2.ROTATE_180, "180"),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, "270_ccw"),
        ]

        for rot_code, rot_name in rotations:
            small_rot = cv2.rotate(small, rot_code)
            rot_picked, rot_cands = _detect_registration_blocks_internal(small_rot)

            if len(rot_picked) > len(picked_dict):
                mapped_dict = {}
                for k, pt in rot_picked.items():
                    x_rot, y_rot = pt
                    if rot_name == "90_cw":
                        mapped_dict[k] = np.array([y_rot, h - x_rot], dtype=np.float32)
                    elif rot_name == "180":
                        mapped_dict[k] = np.array([w - x_rot, h - y_rot], dtype=np.float32)
                    elif rot_name == "270_ccw":
                        mapped_dict[k] = np.array([w - y_rot, x_rot], dtype=np.float32)
                picked_dict = mapped_dict
                if len(picked_dict) >= 4:
                    break

    picked = []
    for corner in ("TL", "TR", "BR", "BL"):
        if corner not in picked_dict:
            raise ValueError(
                f"Could not detect {corner} registration block. "
                "Keep the whole OMR sheet visible, reduce glare, "
                "and avoid covering the corner marks."
            )
        picked.append(picked_dict[corner] / scale)

    markers = order_points(
        np.array(
            picked,
            dtype=np.float32,
        )
    )

    full_h, full_w = image.shape[:2]

    _validate_marker_geometry(
        markers,
        full_w,
        full_h,
    )

    debug = {
        "candidate_count":
            len(candidates),

        "scale":
            float(scale),

        "markers": [
            [
                round(
                    float(point[0]),
                    2,
                ),
                round(
                    float(point[1]),
                    2,
                ),
            ]
            for point
            in markers
        ],
    }

    return markers, debug


def _canonical_marker_positions(
    width: int,
    height: int,
) -> np.ndarray:
    markers = (
        CANONICAL_MARKERS_1600_2200
        .copy()
    )

    markers[:, 0] *= (
        width / 1600.0
    )

    markers[:, 1] *= (
        height / 2200.0
    )

    return markers.astype(
        np.float32
    )


def _validate_canonical_marker_positions(
    markers: np.ndarray,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Reject an A4 warp whose internal registration blocks are implausible.

    Page corners establish the coordinate system.  These blocks are a
    validation signal only; they are deliberately not used for another crop
    or page-boundary warp.
    """
    expected = _canonical_marker_positions(width, height)
    distances = np.linalg.norm(markers - expected, axis=1)
    mean_error = float(np.mean(distances))
    max_error = float(np.max(distances))
    valid = mean_error <= 85.0 and max_error <= 130.0
    debug = {
        "detected": True,
        "expected_markers": [[round(float(x), 2), round(float(y), 2)] for x, y in expected],
        "mean_position_error": round(mean_error, 2),
        "max_position_error": round(max_error, 2),
        "valid": valid,
    }
    if not valid:
        raise ValueError(
            "Unable to align the complete OMR sheet. Please place the entire "
            "A4 OMR inside the camera frame with all four corners visible and capture again."
        )
    return debug


def warp_from_registration_blocks(
    image: np.ndarray,
    source_markers: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    source = order_points(
        source_markers
    ).astype(np.float32)

    destination = (
        _canonical_marker_positions(
            width,
            height,
        )
    )

    matrix = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    corrected = cv2.warpPerspective(
        image,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    return corrected, matrix


def _prepare_feature_image(
    image: np.ndarray,
) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8,
        ),
    )

    return cv2.GaussianBlur(
        clahe.apply(gray),
        (
            3,
            3,
        ),
        0,
    )


def _alignment_feature_mask(
    width: int,
    height: int,
) -> np.ndarray:
    """
    Use stable printed structure and de-emphasize changing filled bubbles.
    """
    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.uint8,
    )

    # Header / identity / paper-code region.
    mask[
        :
        int(
            height * 0.36
        ),
        :
    ] = 255

    # Side timing/registration bars.
    mask[
        :,
        :
        int(
            width * 0.12
        )
    ] = 255

    mask[
        :,
        int(
            width * 0.88
        )
        :
    ] = 255

    # Vertical response-column separators.
    for fraction in (
        0.25,
        0.50,
        0.75,
    ):
        x = int(
            width * fraction
        )

        half = int(
            width * 0.018
        )

        mask[
            int(
                height * 0.32
            )
            :,
            max(
                0,
                x - half,
            )
            :
            min(
                width,
                x + half,
            ),
        ] = 255

    # Bottom rules / signature strip.
    mask[
        int(
            height * 0.90
        )
        :,
        :
    ] = 255

    return mask


def _orb_refine(
    moving: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = reference.shape[:2]

    moving_gray = _prepare_feature_image(
        moving
    )

    reference_gray = _prepare_feature_image(
        reference
    )

    mask = _alignment_feature_mask(
        w,
        h,
    )

    orb = cv2.ORB_create(
        nfeatures=6000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=20,
        patchSize=31,
        fastThreshold=10,
    )

    kp_m, des_m = orb.detectAndCompute(
        moving_gray,
        mask,
    )

    kp_r, des_r = orb.detectAndCompute(
        reference_gray,
        mask,
    )

    debug = {
        "orb_keypoints_moving":
            len(
                kp_m or []
            ),

        "orb_keypoints_reference":
            len(
                kp_r or []
            ),

        "orb_good_matches":
            0,

        "orb_inliers":
            0,

        "orb_applied":
            False,
    }

    if (
        des_m is None
        or des_r is None
    ):
        return moving, debug

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        des_m,
        des_r,
        k=2,
    )

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if (
            first.distance
            < 0.72
            * second.distance
        ):
            good.append(
                first
            )

    debug[
        "orb_good_matches"
    ] = len(good)

    if len(good) < 25:
        return moving, debug

    source_points = np.float32(
        [
            kp_m[
                match.queryIdx
            ].pt
            for match
            in good
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    destination_points = np.float32(
        [
            kp_r[
                match.trainIdx
            ].pt
            for match
            in good
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    homography, inlier_mask = (
        cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            3.0,
        )
    )

    if homography is None:
        return moving, debug

    inliers = (
        int(
            inlier_mask.sum()
        )
        if inlier_mask
        is not None
        else 0
    )

    debug[
        "orb_inliers"
    ] = inliers

    if inliers < 18:
        return moving, debug

    # Homography must leave reference corners close to the output canvas.
    corners = np.float32(
        [
            [
                0,
                0,
            ],
            [
                w - 1,
                0,
            ],
            [
                w - 1,
                h - 1,
            ],
            [
                0,
                h - 1,
            ],
        ]
    ).reshape(
        -1,
        1,
        2,
    )

    transformed = (
        cv2.perspectiveTransform(
            corners,
            homography,
        )
        .reshape(
            4,
            2,
        )
    )

    expected = corners.reshape(
        4,
        2,
    )

    corner_error = float(
        np.mean(
            np.linalg.norm(
                transformed
                - expected,
                axis=1,
            )
        )
    )

    debug[
        "orb_corner_error"
    ] = corner_error

    # This is only a fine registration. Reject aggressive warps: the four
    # registration blocks have already established the complete sheet frame.
    if corner_error > MAX_FINE_ALIGNMENT_CORNER_ERROR:
        debug["orb_rejected_reason"] = "fine_alignment_exceeds_geometry_limit"
        return moving, debug

    refined = cv2.warpPerspective(
        moving,
        homography,
        (
            w,
            h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    debug[
        "orb_applied"
    ] = True

    return refined, debug


def _ecc_refine(
    moving: np.ndarray,
    reference: np.ndarray,
    minimum_score: float = 0.75,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Small affine ECC refinement.

    IMPORTANT:
    A low-confidence ECC result is rejected. The previous alignment is kept.
    """
    full_h, full_w = reference.shape[:2]

    small_w = 800
    small_h = int(
        round(
            full_h
            * (
                small_w
                / full_w
            )
        )
    )

    moving_small = cv2.resize(
        _prepare_feature_image(
            moving
        ),
        (
            small_w,
            small_h,
        ),
        interpolation=cv2.INTER_AREA,
    ).astype(
        np.float32
    ) / 255.0

    reference_small = cv2.resize(
        _prepare_feature_image(
            reference
        ),
        (
            small_w,
            small_h,
        ),
        interpolation=cv2.INTER_AREA,
    ).astype(
        np.float32
    ) / 255.0

    mask_small = cv2.resize(
        _alignment_feature_mask(
            full_w,
            full_h,
        ),
        (
            small_w,
            small_h,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    warp_small = np.eye(
        2,
        3,
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )

    debug = {
        "ecc_attempted":
            True,

        "ecc_applied":
            False,

        "ecc_score":
            None,

        "ecc_minimum_score":
            float(
                minimum_score
            ),
    }

    try:
        score, warp_small = (
            cv2.findTransformECC(
                reference_small,
                moving_small,
                warp_small,
                cv2.MOTION_AFFINE,
                criteria,
                inputMask=mask_small,
                gaussFiltSize=5,
            )
        )
    except cv2.error:
        return moving, debug

    debug[
        "ecc_score"
    ] = float(score)

    # Critical safety gate.
    if score < minimum_score:
        return moving, debug

    sx = (
        full_w
        / float(
            small_w
        )
    )

    sy = (
        full_h
        / float(
            small_h
        )
    )

    scale_to_full = np.array(
        [
            [
                sx,
                0,
                0,
            ],
            [
                0,
                sy,
                0,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float32,
    )

    scale_to_small = np.array(
        [
            [
                1.0 / sx,
                0,
                0,
            ],
            [
                0,
                1.0 / sy,
                0,
            ],
            [
                0,
                0,
                1,
            ],
        ],
        dtype=np.float32,
    )

    affine3 = np.vstack(
        [
            warp_small,
            [
                0,
                0,
                1,
            ],
        ]
    ).astype(
        np.float32
    )

    full_affine3 = (
        scale_to_full
        @ affine3
        @ scale_to_small
    )

    full_affine = (
        full_affine3[
            :
            2,
            :
        ]
    )

    corners = np.float32(
        [[0, 0], [full_w - 1, 0], [full_w - 1, full_h - 1], [0, full_h - 1]]
    ).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, full_affine).reshape(4, 2)
    affine_corner_error = float(
        np.mean(
            np.linalg.norm(transformed_corners - corners.reshape(4, 2), axis=1)
        )
    )
    debug["ecc_corner_error"] = affine_corner_error
    if affine_corner_error > MAX_FINE_ALIGNMENT_CORNER_ERROR:
        debug["ecc_rejected_reason"] = "fine_alignment_exceeds_geometry_limit"
        return moving, debug

    refined = cv2.warpAffine(
        moving,
        full_affine,
        (
            full_w,
            full_h,
        ),
        flags=(
            cv2.INTER_LINEAR
            |
            cv2.WARP_INVERSE_MAP
        ),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    debug[
        "ecc_applied"
    ] = True

    return refined, debug


def _draw_marker_debug(
    image: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    names = (
        "TL",
        "TR",
        "BR",
        "BL",
    )

    for name, point in zip(
        names,
        markers,
    ):
        x = int(
            round(
                float(
                    point[0]
                )
            )
        )

        y = int(
            round(
                float(
                    point[1]
                )
            )
        )

        cv2.circle(
            output,
            (
                x,
                y,
            ),
            18,
            (
                0,
                0,
                255,
            ),
            4,
        )

        cv2.putText(
            output,
            name,
            (
                x + 20,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

    polygon = (
        markers
        .astype(
            np.int32
        )
        .reshape(
            -1,
            1,
            2,
        )
    )

    cv2.polylines(
        output,
        [
            polygon
        ],
        True,
        (
            255,
            0,
            0,
        ),
        3,
    )

    return output



# ============================================================
# CANONICAL ORIENTATION
# ============================================================

def _rotate_to_candidate(
    image: np.ndarray,
    rotation: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Rotate a canonical-size image by 0/90/180/270 degrees and resize
    back to the exact canonical canvas.

    This is only used for orientation selection after perspective
    correction, never for JSON coordinate modification.
    """

    rotation = int(rotation) % 360

    if rotation == 0:
        candidate = image.copy()

    elif rotation == 90:
        candidate = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE,
        )

    elif rotation == 180:
        candidate = cv2.rotate(
            image,
            cv2.ROTATE_180,
        )

    elif rotation == 270:
        candidate = cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    else:
        raise ValueError(
            f"Unsupported rotation: {rotation}"
        )

    if (
        candidate.shape[1] != width
        or candidate.shape[0] != height
    ):
        candidate = cv2.resize(
            candidate,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    return candidate


def _header_structure_score(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Compare ONLY the canonical top/header region.

    This avoids the response grid dominating orientation because the
    response section is visually repetitive and can look similar at 180°.
    """
    if candidate.ndim == 3:
        cand_gray = cv2.cvtColor(
            candidate,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        cand_gray = candidate.copy()

    if reference.ndim == 3:
        ref_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        ref_gray = reference.copy()

    h, w = ref_gray.shape[:2]

    # Use only top 32%: Manchester header + instructions + student fields.
    top_h = int(
        round(
            h * 0.32
        )
    )

    cand_top = cand_gray[
        :top_h,
        :
    ]

    ref_top = ref_gray[
        :top_h,
        :
    ]

    target_w = 900

    target_h = int(
        round(
            top_h
            *
            (
                target_w
                /
                float(w)
            )
        )
    )

    cand_top = cv2.resize(
        cand_top,
        (
            target_w,
            target_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    ref_top = cv2.resize(
        ref_top,
        (
            target_w,
            target_h,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # Mild normalization so lighting does not dominate.
    cand_top = cv2.equalizeHist(
        cand_top
    )

    ref_top = cv2.equalizeHist(
        ref_top
    )

    cand_edges = cv2.Canny(
        cand_top,
        45,
        140,
    )

    ref_edges = cv2.Canny(
        ref_top,
        45,
        140,
    )

    # Direct normalized correlation of header structure.
    cand_f = cand_edges.astype(
        np.float32
    )

    ref_f = ref_edges.astype(
        np.float32
    )

    cand_f -= float(
        cand_f.mean()
    )

    ref_f -= float(
        ref_f.mean()
    )

    denominator = float(
        np.linalg.norm(
            cand_f
        )
        *
        np.linalg.norm(
            ref_f
        )
    )

    correlation = (
        float(
            np.sum(
                cand_f
                *
                ref_f
            )
        )
        /
        denominator
        if denominator > 1e-6
        else 0.0
    )

    # ORB header matches provide a second independent orientation signal.
    orb = cv2.ORB_create(
        nfeatures=3000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10,
    )

    kp_c, des_c = orb.detectAndCompute(
        cand_top,
        None,
    )

    kp_r, des_r = orb.detectAndCompute(
        ref_top,
        None,
    )

    good_count = 0
    inlier_count = 0

    if (
        des_c is not None
        and des_r is not None
        and len(kp_c) >= 8
        and len(kp_r) >= 8
    ):
        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING
        )

        pairs = matcher.knnMatch(
            des_c,
            des_r,
            k=2,
        )

        good = []

        for pair in pairs:
            if len(pair) != 2:
                continue

            first, second = pair

            if (
                first.distance
                <
                0.72
                *
                second.distance
            ):
                good.append(
                    first
                )

        good_count = len(
            good
        )

        if good_count >= 8:
            src_pts = np.float32(
                [
                    kp_c[
                        m.queryIdx
                    ].pt
                    for m
                    in good
                ]
            ).reshape(
                -1,
                1,
                2,
            )

            dst_pts = np.float32(
                [
                    kp_r[
                        m.trainIdx
                    ].pt
                    for m
                    in good
                ]
            ).reshape(
                -1,
                1,
                2,
            )

            _, inlier_mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                4.0,
            )

            if inlier_mask is not None:
                inlier_count = int(
                    inlier_mask.sum()
                )

    # Header correlation dominates; ORB/inliers refine the choice.
    score = (
        correlation
        *
        1000.0
        +
        good_count
        *
        1.5
        +
        inlier_count
        *
        4.0
    )

    return float(
        score
    )


def ensure_canonical_orientation(
    image: np.ndarray,
    reference: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict]:
    """
    Force the Manchester header to the TOP.

    Evaluates 0°, 90°, 180°, and 270° rotations against the canonical reference header
    to guarantee proper upright orientation regardless of photo capture orientation.
    """

    candidates = {
        0:
            _rotate_to_candidate(
                image,
                0,
                width,
                height,
            ),
        90:
            _rotate_to_candidate(
                image,
                90,
                width,
                height,
            ),
        180:
            _rotate_to_candidate(
                image,
                180,
                width,
                height,
            ),
        270:
            _rotate_to_candidate(
                image,
                270,
                width,
                height,
            ),
    }

    scores = {
        rotation:
            _header_structure_score(
                candidate,
                reference,
            )
        for rotation, candidate
        in candidates.items()
    }

    best_rotation = max(
        scores,
        key=scores.get,
    )

    oriented = candidates[
        best_rotation
    ]

    return oriented, {
        "selected_rotation":
            int(
                best_rotation
            ),

        "orientation_scores": {
            str(rotation):
                round(
                    float(score),
                    3,
                )
            for rotation, score
            in scores.items()
        },

        "orientation_method":
            "header_structural_matching_0_90_180_270",
    }


def canonicalize_omr(
    image: np.ndarray,
    reference_path: str | Path,
    output_size: Tuple[int, int] = (
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
    ),
    use_orb: bool = True,
    use_ecc: bool = True,
    ecc_minimum_score: float = 0.75,
    debug_dir: Optional[
        str | Path
    ] = None,
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:
    """
    Convert the mobile photo into canonical reference geometry using
    the four printed black registration blocks as the primary anchors.

    Pipeline:
      1. detect four registration blocks in original photo
      2. validate their geometry
      3. homography from detected block centres to canonical centres
      4. optional conservative ORB refinement
      5. optional ECC refinement ONLY when score >= threshold
      6. guarantee exact 1600x2200 output
    """
    width, height = map(
        int,
        output_size,
    )

    reference = cv2.imread(
        str(
            reference_path
        )
    )

    if reference is None:
        raise ValueError(
            "Could not load canonical reference: "
            f"{reference_path}"
        )

    reference = cv2.resize(
        reference,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # The printed corner blocks are the reliable OMR geometry signal.  Their
    # destination coordinates are inset from the template edges, so this
    # single homography preserves the full page margins rather than cropping
    # block-to-block or treating blocks as page corners.
    markers, marker_debug = detect_registration_blocks(image)
    coarse, homography = warp_from_registration_blocks(
        image,
        markers,
        width,
        height,
    )

    # The registration-block homography maps the complete source sheet
    # directly to the template rectangle.  Applying a second cardinal
    # rotation here (then resizing it back) changes the coordinate system and
    # is the source of shifted bubble-debug overlays.  Input orientation is
    # resolved by EXIF normalization and the ordered source markers.
    oriented = coarse
    orientation_debug = {
        "selected_rotation": 0,
        "orientation_method": "registration_block_homography",
        "orientation_correction_applied": False,
    }

    result = coarse

    debug: Dict[
        str,
        Any,
    ] = {
        "alignment_method":
            "registration_blocks",

        "document_detection": {
            "document_detected": True,
            "bounds": {
                name: [
                    round(float(point[0]), 2),
                    round(float(point[1]), 2),
                ]
                for name, point in zip(
                    ("top_left", "top_right", "bottom_right", "bottom_left"),
                    markers,
                )
            },
            "perspective_correction_applied": True,
        },

        "page_detection": {
            "method": "four_omr_registration_blocks",
            "candidate_count": marker_debug["candidate_count"],
            "selected_candidate": 1,
        },

        "output_size": {
            "width":
                width,

            "height":
                height,
        },

        "registration":
            marker_debug,

        "orientation":
            orientation_debug,

        "coarse_homography":
            homography.tolist(),
        "homography":
            homography,
    }

    debug[
        "document_detection"
    ][
        "rotation_angle"
    ] = orientation_debug[
        "selected_rotation"
    ]

    if use_orb:
        result, orb_debug = (
            _orb_refine(
                result,
                reference,
            )
        )

        debug.update(
            orb_debug
        )

    if use_ecc:
        result, ecc_debug = (
            _ecc_refine(
                result,
                reference,
                minimum_score=
                    ecc_minimum_score,
            )
        )

        debug.update(
            ecc_debug
        )

    if (
        result.shape[1]
        != width
        or result.shape[0]
        != height
    ):
        result = cv2.resize(
            result,
            (
                width,
                height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    if debug_dir is not None:
        debug_dir = Path(
            debug_dir
        )

        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Fine alignment is optional, but it must not invalidate the complete
    # page geometry established above.  Validate the final recognition image,
    # not merely the pre-refinement warp.
    final_markers, final_marker_debug = detect_registration_blocks(result)
    final_validation = _validate_canonical_marker_positions(
        final_markers,
        width,
        height,
    )
    debug["registration"]["final_markers"] = final_marker_debug["markers"]
    debug["registration"]["final_canonical_position_validation"] = final_validation

    if debug_dir is not None:
        page_debug_image = _draw_marker_debug(image, markers)
        cv2.polylines(
            page_debug_image,
            [markers.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 0, 255),
            4,
        )
        # Full-page geometry trace.  The selected quadrilateral is repeated
        # in candidates/selected views because detection metadata records the
        # candidate count and score without altering the source image.
        cv2.imwrite(str(debug_dir / "01_raw_camera.jpg"), image)
        cv2.imwrite(str(debug_dir / "02_page_candidates.jpg"), page_debug_image)
        cv2.imwrite(str(debug_dir / "03_selected_a4_page.jpg"), page_debug_image)
        cv2.imwrite(str(debug_dir / "04_a4_perspective_corrected.jpg"), coarse)
        cv2.imwrite(str(debug_dir / "05_canonical_omr.jpg"), result)
        cv2.imwrite(
            str(debug_dir / "06_corner_block_validation.jpg"),
            _draw_marker_debug(result, final_markers),
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "00_registration_detection.jpg"
            ),
            _draw_marker_debug(
                coarse,
                markers,
            ),
        )

        cv2.imwrite(str(debug_dir / "00_a4_page_quad.jpg"), page_debug_image)

        cv2.imwrite(
            str(
                debug_dir
                /
                "01_registration_warp.jpg"
            ),
            coarse,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "02_oriented_canonical.jpg"
            ),
            oriented,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "03_canonical_aligned.jpg"
            ),
            result,
        )

        cv2.imwrite(
            str(
                debug_dir
                /
                "04_reference.jpg"
            ),
            reference,
        )

    return result, debug

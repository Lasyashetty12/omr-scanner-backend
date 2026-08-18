from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 2200


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


def _quad_geometry_score(
    quad: np.ndarray,
    gray: np.ndarray,
    expected_ratio: float,
) -> Optional[float]:
    quad = order_points(quad)
    h, w = gray.shape[:2]
    image_area = float(h * w)

    area = abs(float(cv2.contourArea(quad.reshape(-1, 1, 2))))
    coverage = area / max(image_area, 1.0)

    if coverage < 0.25 or coverage > 0.995:
        return None

    tl, tr, br, bl = quad

    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))

    avg_w = (top_w + bottom_w) / 2.0
    avg_h = (left_h + right_h) / 2.0

    if avg_w <= 0 or avg_h <= 0:
        return None

    ratio = avg_w / avg_h

    # Allow perspective but keep portrait-page geometry.
    if ratio < 0.45 or ratio > 1.02:
        return None

    ratio_error = abs(ratio - expected_ratio) / expected_ratio
    ratio_score = max(0.0, 1.0 - ratio_error)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)

    pixels = gray[mask > 0]
    if pixels.size == 0:
        return None

    brightness = float(np.mean(pixels)) / 255.0
    white_fraction = float(np.mean(pixels > 135))

    # Prefer a large bright portrait sheet with the expected ratio.
    return (
        coverage * 4.0
        + ratio_score * 3.0
        + white_fraction * 2.0
        + brightness
    )


def _candidate_quads_from_contour(contour: np.ndarray) -> list[tuple[np.ndarray, str]]:
    candidates: list[tuple[np.ndarray, str]] = []

    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    if perimeter > 0:
        for factor in (0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040, 0.050):
            approx = cv2.approxPolyDP(
                hull,
                factor * perimeter,
                True,
            )
            if len(approx) == 4:
                candidates.append((approx.reshape(4, 2).astype(np.float32), "contour_quad"))

    # Safe fallback for sheets whose edge merges with another sheet/background.
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    candidates.append((box, "min_area_rect"))

    return candidates


def detect_document_quad(
    image: np.ndarray,
    expected_ratio: float = DEFAULT_WIDTH / DEFAULT_HEIGHT,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, Dict[str, Any]]:
    """
    Find the four OUTER paper corners.

    This intentionally does NOT use the printed registration-marker
    centres as page corners. Those marks sit inside the sheet and mapping
    them to the output edges causes clipping/stretching.

    Returns TL, TR, BR, BL.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Work at a moderate resolution for stable/fast contour detection.
    max_side = 1200
    scale = min(1.0, max_side / float(max(h, w)))

    if scale < 1.0:
        small = cv2.resize(
            gray,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray.copy()

    sh, sw = small.shape[:2]

    blurred = cv2.GaussianBlur(small, (5, 5), 0)

    # Two complementary masks:
    # 1) edge structure
    edges = cv2.Canny(blurred, 35, 130)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=2,
    )

    # 2) bright paper region
    _, white = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
        iterations=2,
    )

    best_quad = None
    best_score = -1e9
    candidate_count = 0
    valid_candidate_count = 0

    for mask in (edges, white):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )[:60]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < sh * sw * 0.18:
                continue

            for quad_small, candidate_source in _candidate_quads_from_contour(contour):
                candidate_count += 1
                quad_full = quad_small / scale

                score = _quad_geometry_score(
                    quad_full,
                    gray,
                    expected_ratio,
                )

                # A real four-corner contour preserves page edges better than
                # minAreaRect's enclosing approximation.  Keep the latter as
                # a fallback for weak/occluded edges, never as a blind winner.
                if score is not None and candidate_source == "min_area_rect":
                    score -= 0.15

                if score is not None:
                    valid_candidate_count += 1
                if score is not None and score > best_score:
                    best_score = score
                    best_quad = order_points(quad_full)

    if best_quad is None:
        raise ValueError(
            "OMR sheet could not be detected clearly. "
            "Please place the complete sheet inside the camera frame and scan again."
        )

    best_quad = best_quad.astype(np.float32)
    if not return_debug:
        return best_quad

    tl, tr, br, bl = best_quad
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    detected_ratio = ((top + bottom) / 2.0) / max((left + right) / 2.0, 1.0)
    area_ratio = abs(float(cv2.contourArea(best_quad.reshape(-1, 1, 2)))) / float(h * w)
    return best_quad, {
        "candidate_count": candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "selected_score": round(float(best_score), 4),
        "a4_ratio_error": round(abs(detected_ratio - expected_ratio) / expected_ratio, 4),
        "page_area_ratio": round(area_ratio, 4),
    }


def warp_document_quad(
    image: np.ndarray,
    quad: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    src = order_points(quad)

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# Backwards-compatible private alias for the standalone canonical pipeline.
_warp_page = warp_document_quad


def _prepare_feature_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _alignment_feature_mask(width: int, height: int) -> np.ndarray:
    """
    Prefer stable printed structure over the variable filled answer bubbles.

    Includes:
      - top/header area
      - left/right border strips
      - central separators / lower signature band
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Header / identity / paper-code area.
    mask[: int(height * 0.36), :] = 255

    # Side strips contain registration/row markers and page rules.
    mask[:, : int(width * 0.11)] = 255
    mask[:, int(width * 0.89) :] = 255

    # Column separator structure without emphasizing all bubble interiors.
    for x_fraction in (0.25, 0.50, 0.75):
        x = int(width * x_fraction)
        half = int(width * 0.025)
        mask[int(height * 0.33) :, max(0, x - half) : min(width, x + half)] = 255

    # Bottom signature/footer structure.
    mask[int(height * 0.90) :, :] = 255

    return mask


def _orb_refine(
    moving: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Match the coarse scan to the canonical sheet using printed features.

    Returns the refined image. If feature matching is not strong enough,
    returns the input unchanged.
    """
    h, w = reference.shape[:2]

    moving_gray = _prepare_feature_image(moving)
    reference_gray = _prepare_feature_image(reference)
    mask = _alignment_feature_mask(w, h)

    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=20,
        patchSize=31,
        fastThreshold=12,
    )

    kp_m, des_m = orb.detectAndCompute(moving_gray, mask)
    kp_r, des_r = orb.detectAndCompute(reference_gray, mask)

    debug = {
        "orb_keypoints_moving": len(kp_m or []),
        "orb_keypoints_reference": len(kp_r or []),
        "orb_good_matches": 0,
        "orb_inliers": 0,
        "orb_applied": False,
    }

    if des_m is None or des_r is None:
        return moving, debug

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_m, des_r, k=2)

    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance:
            good.append(m)

    debug["orb_good_matches"] = len(good)

    if len(good) < 20:
        return moving, debug

    src_pts = np.float32(
        [kp_m[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp_r[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    homography, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        3.0,
    )

    if homography is None:
        return moving, debug

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    debug["orb_inliers"] = inliers

    if inliers < 14:
        return moving, debug

    # Reject obviously destructive homographies.
    corners = np.float32(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ]
    ).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(
        corners,
        homography,
    ).reshape(4, 2)

    transformed_area = abs(
        float(
            cv2.contourArea(
                transformed.reshape(-1, 1, 2)
            )
        )
    )

    expected_area = float(w * h)
    area_ratio = transformed_area / expected_area

    if not 0.75 <= area_ratio <= 1.25:
        return moving, debug

    refined = cv2.warpPerspective(
        moving,
        homography,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    debug["orb_applied"] = True
    return refined, debug


def _ecc_refine(
    moving: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Final small affine refinement. Runs on a half-resolution copy for speed.

    ECC is deliberately only a fine correction; the coarse page warp and
    ORB step must already be reasonably aligned.
    """
    full_h, full_w = reference.shape[:2]

    small_w = 800
    small_h = int(round(full_h * (small_w / full_w)))

    mov_small = cv2.resize(
        _prepare_feature_image(moving),
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0

    ref_small = cv2.resize(
        _prepare_feature_image(reference),
        (small_w, small_h),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0

    mask_small = cv2.resize(
        _alignment_feature_mask(full_w, full_h),
        (small_w, small_h),
        interpolation=cv2.INTER_NEAREST,
    )

    warp_small = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        45,
        1e-5,
    )

    debug = {
        "ecc_applied": False,
        "ecc_score": None,
    }

    try:
        score, warp_small = cv2.findTransformECC(
            ref_small,
            mov_small,
            warp_small,
            cv2.MOTION_AFFINE,
            criteria,
            inputMask=mask_small,
            gaussFiltSize=5,
        )
    except cv2.error:
        return moving, debug

    # Convert the affine transform from small coordinates to full coordinates.
    sx = full_w / float(small_w)
    sy = full_h / float(small_h)

    scale_to_full = np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    scale_to_small = np.array(
        [
            [1.0 / sx, 0, 0],
            [0, 1.0 / sy, 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    affine3 = np.vstack(
        [
            warp_small,
            [0, 0, 1],
        ]
    ).astype(np.float32)

    # findTransformECC returns a warp intended with WARP_INVERSE_MAP.
    full_affine3 = (
        scale_to_full
        @ affine3
        @ scale_to_small
    )

    full_affine = full_affine3[:2, :]

    refined = cv2.warpAffine(
        moving,
        full_affine,
        (full_w, full_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    debug["ecc_applied"] = True
    debug["ecc_score"] = float(score)

    return refined, debug


def canonicalize_omr(
    image: np.ndarray,
    reference_path: str | Path,
    output_size: Tuple[int, int] = (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    use_orb: bool = True,
    use_ecc: bool = True,
    debug_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert a mobile OMR photo to canonical reference geometry.

    The result is suitable for applying fixed JSON bubble coordinates.

    Steps:
      1) outer-page detection
      2) perspective warp to exact output size
      3) ORB/RANSAC registration against canonical reference
      4) small ECC affine refinement
    """
    width, height = map(int, output_size)

    reference_path = Path(reference_path)
    reference = cv2.imread(str(reference_path))

    if reference is None:
        raise ValueError(
            f"Could not load canonical reference image: {reference_path}"
        )

    reference = cv2.resize(
        reference,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    quad = detect_document_quad(
        image,
        expected_ratio=width / float(height),
    )

    coarse = _warp_page(
        image,
        quad,
        width,
        height,
    )

    result = coarse

    debug: Dict[str, Any] = {
        "document_quad": [
            [round(float(x), 2), round(float(y), 2)]
            for x, y in quad
        ],
        "output_size": {
            "width": width,
            "height": height,
        },
    }

    if use_orb:
        result, orb_debug = _orb_refine(
            result,
            reference,
        )
        debug.update(orb_debug)

    if use_ecc:
        result, ecc_debug = _ecc_refine(
            result,
            reference,
        )
        debug.update(ecc_debug)

    # Absolute output-size guarantee.
    if result.shape[1] != width or result.shape[0] != height:
        result = cv2.resize(
            result,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(
            str(debug_dir / "01_coarse_page_warp.jpg"),
            coarse,
        )
        cv2.imwrite(
            str(debug_dir / "02_canonical_aligned.jpg"),
            result,
        )
        cv2.imwrite(
            str(debug_dir / "03_reference.jpg"),
            reference,
        )

        # Draw detected original page corners.
        original_debug = image.copy()
        q = quad.astype(np.int32)
        cv2.polylines(
            original_debug,
            [q.reshape(-1, 1, 2)],
            True,
            (0, 0, 255),
            4,
        )
        cv2.imwrite(
            str(debug_dir / "00_detected_page.jpg"),
            original_debug,
        )

    return result, debug

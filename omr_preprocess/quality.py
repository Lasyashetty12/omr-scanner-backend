from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np


def _clamp_score(value: float) -> float:
    return round(float(np.clip(value, 0.0, 100.0)), 2)


def _quad_metrics(markers: list[list[float]], width: int, height: int) -> tuple[float, float]:
    if len(markers) != 4:
        return 0.0, 0.0

    points = np.asarray(markers, dtype=np.float32).reshape(4, 2)
    area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
    coverage = area / max(float(width * height), 1.0)

    top = float(np.linalg.norm(points[1] - points[0]))
    right = float(np.linalg.norm(points[2] - points[1]))
    bottom = float(np.linalg.norm(points[2] - points[3]))
    left = float(np.linalg.norm(points[3] - points[0]))

    horizontal_balance = min(top, bottom) / max(top, bottom, 1.0)
    vertical_balance = min(left, right) / max(left, right, 1.0)
    perspective_quality = 100.0 * min(horizontal_balance, vertical_balance)

    return coverage, perspective_quality


def assess_document_quality(
    original_bgr: np.ndarray,
    canonical_bgr: np.ndarray,
    alignment_debug: Dict[str, Any],
) -> Dict[str, Any]:
    """Report camera/document quality without modifying recognition input."""
    original_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    canonical_gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)

    # The project has historically exposed quality from the camera image before
    # canonical resizing. Keep that metric scale for the quality gate: resizing
    # changes Laplacian variance substantially and would make the gate fragile.
    sharpness = float(cv2.Laplacian(original_gray, cv2.CV_64F).var())
    brightness = float(np.mean(original_gray))
    contrast = float(np.std(original_gray))
    document_sharpness = float(cv2.Laplacian(canonical_gray, cv2.CV_64F).var())

    background = cv2.GaussianBlur(canonical_gray, (0, 0), sigmaX=45, sigmaY=45)
    illumination_range = float(
        np.percentile(background, 95.0) - np.percentile(background, 5.0)
    )
    # A normal printed OMR sheet has meaningful low-frequency variation from
    # headers, response columns, and margins. Keep this metric conservative so
    # those structures are not mislabeled as a camera shadow.
    illumination_uniformity = _clamp_score(100.0 - illumination_range * 0.65)

    shadow_threshold = float(np.percentile(background, 90.0) - 35.0)
    glare_threshold = float(np.percentile(background, 10.0) + 45.0)
    shadow_percent = round(float(np.mean(background < shadow_threshold) * 100.0), 2)
    glare_percent = round(float(np.mean(background > glare_threshold) * 100.0), 2)

    registration = alignment_debug.get("registration", {})
    markers = registration.get("markers", [])
    coverage, perspective_quality = _quad_metrics(
        markers,
        original_bgr.shape[1],
        original_bgr.shape[0],
    )
    document_detected = len(markers) == 4
    if not document_detected and alignment_debug.get("crop"):
        # JEE uses its existing outer-page path rather than registration blocks.
        document_detected = True
        coverage = 1.0
        perspective_quality = 100.0

    sharpness_score = _clamp_score((sharpness - 150.0) * 100.0 / 850.0)
    brightness_score = _clamp_score(
        100.0 - max(0.0, 130.0 - brightness) * 1.4 - max(0.0, brightness - 240.0) * 1.0
    )
    contrast_score = _clamp_score(100.0 - max(0.0, 20.0 - contrast) * 3.0)
    coverage_score = _clamp_score(coverage * 150.0)

    overall_score = round(
        sharpness_score * 0.50
        + brightness_score * 0.20
        + contrast_score * 0.08
        + illumination_uniformity * 0.10
        + coverage_score * 0.07
        + perspective_quality * 0.05,
        2,
    )

    warnings: list[str] = []
    # These metrics describe image quality, not whether the OMR engine can
    # recover a sheet.  In particular, do not turn a less-than-ideal camera
    # image into a hard failure just because it differs from the reference.
    # Existing document alignment and bubble recognition remain the final
    # authority for borderline images.
    minimum_dimension = min(original_bgr.shape[:2])
    resolution_ok = minimum_dimension >= 600
    severe_lighting = (
        illumination_uniformity < 40.0
        or shadow_percent > 35.0
        or glare_percent > 35.0
    )

    # Only stop scans which are clearly unusable.  A single generic metric is
    # deliberately never sufficient: camera-scale Laplacian variance, mean
    # brightness, and page coverage vary substantially among scans which the
    # established OMR pipeline can still read.
    nearly_uniform = contrast < 6.0
    extreme_exposure = brightness < 20.0 or brightness > 252.0
    no_usable_detail = nearly_uniform and (extreme_exposure or sharpness < 12.0)
    tiny_image = minimum_dimension < 320

    if tiny_image or no_usable_detail:
        classification = "REJECT"
        warnings.append("Image quality is too low to scan reliably. Please scan again.")
    elif (
        sharpness < 150.0
        or brightness < 70.0
        or brightness > 248.0
        or contrast < 15.0
        or not resolution_ok
        or not document_detected
        or coverage < 0.45
        or perspective_quality < 55.0
        or severe_lighting
    ):
        classification = "POOR"
        warnings.append(
            "Image quality is below optimal. For more reliable scanning, use "
            "Document Mode and scan again."
        )
    elif sharpness < 600.0 or brightness < 105.0 or brightness > 245.0 or contrast < 20.0:
        classification = "ACCEPTABLE"
    elif overall_score >= 80.0:
        classification = "GOOD"
    else:
        classification = "ACCEPTABLE"

    if sharpness < 600.0:
        warnings.append(
            f"Sharpness is low ({sharpness:.2f}); OMR recognition may be unreliable."
        )
    if brightness < 70.0:
        warnings.append(f"Document is underexposed ({brightness:.2f}).")
    elif brightness > 248.0:
        warnings.append(f"Document is overexposed ({brightness:.2f}).")
    if not document_detected:
        warnings.append("Could not confidently detect all document registration anchors.")
    if severe_lighting:
        warnings.append("Uneven illumination or a broad shadow is present on the document.")
    if perspective_quality and perspective_quality < 55.0:
        warnings.append("Document perspective is strongly distorted.")

    if not resolution_ok:
        warnings.append("Camera resolution is below the recommended level for bubble visibility.")

    # POOR is a non-blocking advisory.  This preserves the practical operating
    # range of the pre-existing recognition pipeline.
    can_scan = classification != "REJECT"

    return {
        "classification": classification,
        "overall_score": overall_score,
        "can_scan": can_scan,
        "recognition_recommendation": "retake_using_document_mode" if not can_scan else "scan_with_caution" if classification == "POOR" else "ready",
        "warnings": warnings,
        "original_resolution": {
            "width": int(original_bgr.shape[1]),
            "height": int(original_bgr.shape[0]),
        },
        "document_resolution": {
            "width": int(canonical_bgr.shape[1]),
            "height": int(canonical_bgr.shape[0]),
        },
        "document_detected": document_detected,
        "document_coverage_percent": round(coverage * 100.0, 2),
        "perspective_quality": round(perspective_quality, 2),
        "sharpness": round(sharpness, 2),
        "document_sharpness": round(document_sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "illumination_uniformity": illumination_uniformity,
        "illumination_range": round(illumination_range, 2),
        "shadow_percent": shadow_percent,
        "glare_percent": glare_percent,
        "edge_clarity": sharpness_score,
    }

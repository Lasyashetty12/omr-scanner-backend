from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Document mode received an empty image.")

    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _image_characteristics(gray: np.ndarray) -> Dict[str, float]:
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    illumination_range = float(
        np.percentile(background, 95.0)
        - np.percentile(background, 5.0)
    )

    return {
        "brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "illumination_range": round(illumination_range, 2),
        "blur_score": round(
            float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            2,
        ),
    }


def _gentle_illumination_correction(
    gray: np.ndarray,
    characteristics: Dict[str, float],
) -> np.ndarray:
    """Flatten only broad lighting gradients; never create a binary mask."""
    short_side = min(gray.shape[:2])
    sigma = float(np.clip(short_side / 28.0, 30.0, 75.0))
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    paper_level = max(float(np.percentile(background, 92.0)), 1.0)
    normalized = cv2.divide(
        gray,
        np.maximum(background, 1).astype(np.uint8),
        scale=paper_level,
    )

    illumination_strength = float(
        np.clip(
            0.28 + characteristics["illumination_range"] / 150.0,
            0.28,
            0.62,
        )
    )

    # A partial blend avoids the cloud-like artifacts caused by forcing every
    # local background region to the same value, while still lifting shadows.
    return cv2.addWeighted(
        gray,
        1.0 - illumination_strength,
        normalized,
        illumination_strength,
        0,
    )


def _lift_paper_whites(
    gray: np.ndarray,
    characteristics: Dict[str, float],
) -> np.ndarray:
    """Whiten paper smoothly while leaving printing and filled bubbles intact."""
    values = gray.astype(np.float32)
    light_mask = values > 180.0
    lift = float(
        np.clip((210.0 - characteristics["brightness"]) / 120.0, 0.18, 0.42)
    )
    values[light_mask] += (255.0 - values[light_mask]) * lift
    return np.clip(values, 0, 255).astype(np.uint8)


def enhance_color_saturation(
    bgr_image: np.ndarray,
    saturation_factor: float = 1.4,
    saturation_boost: float = 12.0,
) -> np.ndarray:
    """
    Enhance saturation channel in HSV color space to make blue/black/colored pen marks
    stand out clearly from paper background and printed text.
    """
    if bgr_image is None or bgr_image.size == 0 or bgr_image.ndim != 3:
        return bgr_image

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * saturation_factor + saturation_boost, 0.0, 255.0
    )
    enhanced_hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)


def create_document_scan(
    corrected_bgr: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create a scan-like OMR image without changing its geometry.

    This intentionally uses no adaptive/global binarisation or morphological
    background subtraction. Those operations are prone to erasing thin OMR
    rings or producing artificial patches under uneven camera lighting.
    """
    if corrected_bgr is not None and corrected_bgr.ndim == 3:
        corrected_bgr = enhance_color_saturation(corrected_bgr)

    original = _as_gray(corrected_bgr)
    characteristics = _image_characteristics(original)
    lighting = _gentle_illumination_correction(original, characteristics)

    denoise_strength = float(
        np.clip(14.0 + (34.0 - characteristics["contrast"]) * 0.28, 12.0, 22.0)
    )

    denoised = cv2.bilateralFilter(
        lighting,
        d=5,
        sigmaColor=denoise_strength,
        sigmaSpace=denoise_strength,
    )

    clahe = cv2.createCLAHE(
        clipLimit=float(
            np.clip(1.05 + (32.0 - characteristics["contrast"]) / 80.0, 1.05, 1.38)
        ),
        tileGridSize=(16, 16),
    )
    contrasted = clahe.apply(denoised)

    soft = cv2.GaussianBlur(
        contrasted,
        (0, 0),
        sigmaX=0.65,
        sigmaY=0.65,
    )
    sharpen_amount = float(
        np.clip(0.12 + (110.0 - characteristics["blur_score"]) / 900.0, 0.10, 0.20)
    )
    sharpened = cv2.addWeighted(
        contrasted,
        1.0 + sharpen_amount,
        soft,
        -sharpen_amount,
        0,
    )

    whitened = _lift_paper_whites(sharpened, characteristics)
    final = cv2.cvtColor(whitened, cv2.COLOR_GRAY2BGR)

    stages = {
        "original": original,
        "lighting": lighting,
        "denoised": denoised,
        "whitened": whitened,
        "final": final,
    }

    return final, stages


def _write_debug_stages(
    debug_dir: Path,
    stages: Dict[str, np.ndarray],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "05_lighting_corrected.jpg": stages["lighting"],
        "06_denoised.jpg": stages["denoised"],
        "07_final_omr_input.jpg": stages["final"],
    }

    for filename, image in files.items():
        cv2.imwrite(str(debug_dir / filename), image)


def prepare_omr_document_mode(
    corrected_bgr: np.ndarray,
    debug_dir: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply document-mode appearance enhancement after canonical registration.

    The caller has already performed page detection, perspective correction,
    and orientation selection. This function deliberately preserves pixel
    dimensions and does not alter any OMR coordinates.
    """
    document_image, stages = create_document_scan(corrected_bgr)

    if debug_dir is not None:
        _write_debug_stages(Path(debug_dir), stages)

    height, width = document_image.shape[:2]
    debug = {
        "profile": "gentle_document_mode_v2",
        "preview_only": True,
        "recognition_image_modified": False,
        "recognition_source": "canonical_registered",
        "geometry_changed": False,
        "adaptive_threshold_used": False,
        "document_width": int(width),
        "document_height": int(height),
        "stages": [
            "lighting_correction",
            "edge_preserving_denoise",
            "controlled_contrast",
            "controlled_sharpening",
            "paper_whitening",
        ],
        "image_characteristics": _image_characteristics(
            stages["original"]
        ),
    }

    # Pass the enhanced document image to the recognition pipeline while preserving pixel dimensions.
    return document_image, document_image.copy(), debug

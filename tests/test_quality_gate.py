import cv2
import numpy as np

from omr_preprocess.quality import assess_document_quality


def _quality(image):
    return assess_document_quality(
        image,
        image,
        {"crop": {"source": "test"}},
    )


def test_quality_gate_keeps_readable_reference_variants_non_blocking():
    reference = cv2.imread("references/neet_reference.png")
    assert reference is not None

    # Resize only to keep the test quick; it remains well above the quality
    # gate's minimum usable dimension.
    reference = cv2.resize(reference, (800, 1100))
    blurred = cv2.GaussianBlur(reference, (11, 11), 0)
    darker = np.clip(reference.astype(np.float32) * 0.48, 0, 255).astype(np.uint8)

    assert _quality(reference)["can_scan"]
    assert _quality(blurred)["classification"] == "POOR"
    assert _quality(blurred)["can_scan"]
    assert _quality(darker)["can_scan"]


def test_quality_gate_rejects_a_uniform_overexposed_image():
    image = np.full((800, 1100, 3), 255, dtype=np.uint8)
    quality = _quality(image)

    assert quality["classification"] == "REJECT"
    assert not quality["can_scan"]

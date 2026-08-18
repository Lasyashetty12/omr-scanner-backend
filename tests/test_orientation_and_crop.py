import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from scanner import load_image
from omr_preprocess.registration_align import (
    canonicalize_omr,
    ensure_canonical_orientation,
    detect_registration_blocks,
)


def test_exif_load_image_preserves_array(tmp_path):
    # Create a small dummy image
    test_img = np.full((100, 80, 3), 200, dtype=np.uint8)
    img_path = tmp_path / "test_load.jpg"
    cv2.imwrite(str(img_path), test_img)

    loaded = load_image(img_path)
    assert loaded is not None
    assert loaded.shape[:2] == (100, 80)


def test_ensure_canonical_orientation_4_way():
    # Build a simple synthetic reference header image
    height, width = 2200, 1600
    ref = np.full((height, width, 3), 255, dtype=np.uint8)
    # Draw asymmetrical header feature near the top
    cv2.rectangle(ref, (100, 50), (1500, 300), (0, 0, 0), -1)
    cv2.putText(ref, "HEADER TOP", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 5)

    # Test 180° rotated input image
    rotated_180 = cv2.rotate(ref, cv2.ROTATE_180)
    oriented_180, debug_180 = ensure_canonical_orientation(rotated_180, ref, width, height)
    assert debug_180["selected_rotation"] == 180
    assert oriented_180.shape[:2] == (height, width)

    # Test 90° CW rotated input image
    rotated_90 = cv2.rotate(ref, cv2.ROTATE_90_CLOCKWISE)
    oriented_90, debug_90 = ensure_canonical_orientation(rotated_90, ref, width, height)
    assert debug_90["selected_rotation"] in (90, 270)
    assert oriented_90.shape[:2] == (height, width)


def test_landscape_registration_detection():
    # Create a portrait sheet with registration blocks
    h_port, w_port = 1400, 1000
    sheet = np.full((h_port, w_port, 3), 245, dtype=np.uint8)
    # Registration blocks near corners
    cv2.rectangle(sheet, (80, 80), (160, 160), (10, 10, 10), -1) # TL
    cv2.rectangle(sheet, (840, 80), (920, 160), (10, 10, 10), -1) # TR
    cv2.rectangle(sheet, (840, 1240), (920, 1320), (10, 10, 10), -1) # BR
    cv2.rectangle(sheet, (80, 1240), (160, 1320), (10, 10, 10), -1) # BL

    # Rotate 90° CW to create landscape image
    sheet_land = cv2.rotate(sheet, cv2.ROTATE_90_CLOCKWISE)

    # detect_registration_blocks should succeed on landscape input
    markers, debug = detect_registration_blocks(sheet_land)
    assert len(markers) == 4


def test_canonical_registration_does_not_rotate_after_marker_warp(tmp_path):
    """Marker homography is the sole production geometry transform."""
    height, width = 2200, 1600
    sheet = np.full((height, width, 3), 245, dtype=np.uint8)
    for x, y in ((81, 78), (1522, 78), (1523, 2124), (80, 2120)):
        cv2.rectangle(sheet, (x - 20, y - 20), (x + 20, y + 20), (0, 0, 0), -1)
    # An asymmetric feature makes a cardinal post-warp rotation observable.
    cv2.rectangle(sheet, (220, 320), (230, 330), (0, 0, 255), -1)

    # Embed the full sheet in a camera-like frame with perspective and
    # background so registration blocks establish the full-sheet transform.
    frame_height, frame_width = 2500, 2000
    camera = np.full((frame_height, frame_width, 3), 55, dtype=np.uint8)
    page_quad = np.float32([[250, 150], [1760, 250], [1680, 2380], [310, 2280]])
    source_quad = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    camera = cv2.warpPerspective(
        sheet,
        cv2.getPerspectiveTransform(source_quad, page_quad),
        (frame_width, frame_height),
        dst=camera,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(55, 55, 55),
    )

    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), sheet)
    corrected, debug = canonicalize_omr(
        camera,
        reference_path,
        output_size=(width, height),
        use_orb=False,
        use_ecc=False,
    )

    assert corrected.shape[:2] == (height, width)
    assert debug["orientation"]["selected_rotation"] == 0
    assert debug["page_detection"]["method"] == "four_omr_registration_blocks"
    # Red remains in the canonical top-left content region.
    assert corrected[325, 225, 2] > corrected[325, 225, 0]

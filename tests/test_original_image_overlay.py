import cv2
import numpy as np
import json
from pathlib import Path
from scanner import draw_answer_analysis, create_debug_image

def test_debug_overlay_uses_canonical_recognition_coordinates():
    """The downloadable bubble debug must use the recognition image itself."""
    orig_h, orig_w = 3000, 4000
    orig_img = np.full((orig_h, orig_w, 3), 240, dtype=np.uint8)

    # Draw dummy registration marks on original image
    cv2.rectangle(orig_img, (200, 200), (350, 350), (10, 10, 10), -1)
    cv2.rectangle(orig_img, (3650, 200), (3800, 350), (10, 10, 10), -1)

    template = {
        "exam_name": "KCET",
        "total_questions": 5,
        "options": ["A", "B", "C", "D"],
        "bubble_radius": 11,
        "question_y_positions": [819, 840, 862, 883, 905],
    }

    # Dummy homography matrix (4000x3000 -> 1600x2200)
    src_pts = np.float32([[200, 200], [3800, 200], [3800, 2800], [200, 2800]])
    dst_pts = np.float32([[81.2, 78.3], [1522.0, 78.3], [1523.3, 2124.2], [79.9, 2120.4]])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    answers = {
        1: {
            "answer": "A",
            "ml_status": "filled",
            "ml": {
                "options": {
                    "A": {"draw_center": [235, 819]},
                    "B": {"draw_center": [264, 819]},
                    "C": {"draw_center": [293, 819]},
                    "D": {"draw_center": [323, 819]},
                }
            }
        }
    }

    corrected_dummy = np.full((2200, 1600, 3), 255, dtype=np.uint8)
    debug_out = draw_answer_analysis(
        corrected_dummy,
        template,
        answers,
        original_image=orig_img,
        homography=H,
    )

    assert debug_out is not None
    # Direct helper compatibility remains available for diagnostics, while
    # create_debug_image (used by process_omr) selects canonical output.
    assert debug_out.shape[:2] == (orig_h, orig_w)

    canonical_debug = create_debug_image(
        corrected_dummy,
        answers,
        template,
        original_image=orig_img,
        homography=H,
    )
    assert canonical_debug.shape[:2] == corrected_dummy.shape[:2]

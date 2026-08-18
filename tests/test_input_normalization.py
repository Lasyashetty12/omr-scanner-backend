from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from scanner import load_image


def _jpeg_with_orientation(rgb_image, orientation):
    image = Image.fromarray(rgb_image, "RGB")
    exif = Image.Exif()
    exif[274] = orientation
    output = BytesIO()
    image.save(output, format="JPEG", quality=100, subsampling=0, exif=exif)
    return output.getvalue()


def test_raw_bytes_apply_exif_orientation_and_return_bgr():
    # An asymmetric image makes an accidental missing rotation obvious.
    rgb = np.full((30, 50, 3), 255, dtype=np.uint8)
    rgb[2:12, 2:12] = (255, 0, 0)
    data = _jpeg_with_orientation(rgb, 6)

    image, debug, _raw, _oriented = load_image(
        data,
        filename="mobile.jpg",
        mime_type="image/jpeg",
        return_debug=True,
    )

    assert image.dtype == np.uint8
    assert image.shape == (50, 30, 3)
    assert debug["exif_orientation"] == 6
    assert debug["orientation_applied"] is True
    # RGB red must be BGR red after the common normalization boundary.
    assert int(image[2, 20, 2]) > int(image[2, 20, 0])


def test_equivalent_byte_and_path_inputs_have_same_working_pixels(tmp_path):
    image = np.full((1000, 2000, 3), (10, 90, 220), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_path = tmp_path / "reference.jpg"
    image_path.write_bytes(encoded.tobytes())

    from_bytes = load_image(encoded.tobytes())
    from_path = load_image(image_path)

    assert from_bytes.shape == (800, 1600, 3)
    assert np.array_equal(from_bytes, from_path)


def test_input_debug_reports_mobile_geometry_fields():
    rgb = np.full((200, 150, 3), 200, dtype=np.uint8)
    rgb[30:70, 20:60] = (255, 0, 0)
    encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()

    image, debug, _raw, _oriented = load_image(
        encoded,
        filename="mobile.jpg",
        mime_type="image/jpeg",
        return_debug=True,
    )

    assert debug["source"] == "uploaded_bytes"
    assert debug["image_width"] == 150
    assert debug["image_height"] == 200
    assert debug["aspect_ratio"] == 150 / 200
    assert debug["orientation"] in (None, 1)
    assert debug["preprocessed_width"] == image.shape[1]
    assert debug["preprocessed_height"] == image.shape[0]
    assert debug["preprocessed_aspect_ratio"] == image.shape[1] / float(image.shape[0])

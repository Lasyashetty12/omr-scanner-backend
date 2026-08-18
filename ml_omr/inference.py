from pathlib import Path
import json

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "bubble_classifier.onnx"
)

CLASS_PATH = (
    BASE_DIR
    / "models"
    / "class_names.json"
)

IMAGE_SIZE = 48


# ============================================================
# GLOBAL CACHE
# ============================================================

_session = None
_class_names = None
_input_name = None
_output_name = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_classifier():

    global _session
    global _class_names
    global _input_name
    global _output_name


    if _session is None:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "ONNX model not found: "
                f"{MODEL_PATH}"
            )


        _session = (
            ort.InferenceSession(
                str(MODEL_PATH),

                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )


        _input_name = (
            _session
            .get_inputs()[0]
            .name
        )


        _output_name = (
            _session
            .get_outputs()[0]
            .name
        )


    if _class_names is None:

        if CLASS_PATH.exists():

            with open(
                CLASS_PATH,
                "r",
                encoding="utf-8",
            ) as file:

                _class_names = (
                    json.load(
                        file
                    )
                )

        else:

            _class_names = [
                "ambiguous",
                "blank",
                "filled",
            ]


    return (
        _session,
        _class_names,
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_crop(
    crop,
):

    if (
        crop is None
        or crop.size == 0
    ):

        raise ValueError(
            "Empty bubble crop."
        )


    if crop.ndim == 3:

        crop = (
            cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY,
            )
        )


    # --------------------------------------------------------
    # Local contrast normalization
    # --------------------------------------------------------

    clahe = (
        cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(
                4,
                4,
            ),
        )
    )


    crop = clahe.apply(
        crop
    )


    # --------------------------------------------------------
    # Square padding
    # --------------------------------------------------------

    height, width = (
        crop.shape[:2]
    )


    size = max(
        height,
        width,
    )


    canvas = np.full(
        (
            size,
            size,
        ),
        255,
        dtype=np.uint8,
    )


    y_offset = (
        size - height
    ) // 2


    x_offset = (
        size - width
    ) // 2


    canvas[
        y_offset:
        y_offset + height,

        x_offset:
        x_offset + width

    ] = crop


    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = (
        cv2.resize(
            canvas,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image = (
        resized.astype(
            np.float32
        )
        / 255.0
    )


    # Keras model was trained as:
    #
    # NHWC:
    # (batch, 48, 48, 1)
    #
    # ONNX export will normally preserve this.
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=-1,
    )


    return image


# ============================================================
# SOFTMAX SAFETY
# ============================================================

def softmax(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float32,
    )


    shifted = (
        values
        - np.max(
            values,
            axis=-1,
            keepdims=True,
        )
    )


    exponentials = (
        np.exp(
            shifted
        )
    )


    return (
        exponentials
        /
        np.sum(
            exponentials,
            axis=-1,
            keepdims=True,
        )
    )


# ============================================================
# BATCH CLASSIFICATION
# ============================================================

def classify_batch(
    crops,
):

    if not crops:

        return []


    session, class_names = (
        load_classifier()
    )


    batch = np.stack(
        [
            preprocess_crop(
                crop
            )

            for crop
            in crops
        ],
        axis=0,
    ).astype(
        np.float32
    )


    inputs = (
        session.get_inputs()
    )


    input_shape = (
        inputs[0].shape
    )


    # --------------------------------------------------------
    # Some ONNX converters may output NCHW:
    # (N, 1, 48, 48)
    #
    # Detect and transpose if required.
    # --------------------------------------------------------

    if (
        len(input_shape) == 4
        and input_shape[1] == 1
    ):

        batch = np.transpose(
            batch,
            (
                0,
                3,
                1,
                2,
            ),
        )


    outputs = (
        session.run(
            [_output_name],
            {
                _input_name:
                    batch
            },
        )[0]
    )


    predictions = np.asarray(
        outputs,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Your Keras model ends in softmax, so ONNX will usually
    # already return probabilities.
    #
    # This check protects against logits.
    # --------------------------------------------------------

    row_sums = (
        np.sum(
            predictions,
            axis=1,
        )
    )


    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-3,
    ):

        predictions = (
            softmax(
                predictions
            )
        )


    results = []


    for probabilities in predictions:

        index = int(
            np.argmax(
                probabilities
            )
        )


        results.append(
            {
                "label":
                    class_names[
                        index
                    ],

                "confidence":
                    float(
                        probabilities[
                            index
                        ]
                    ),

                "probabilities": {

                    class_names[i]:
                        float(
                            probabilities[
                                i
                            ]
                        )

                    for i
                    in range(
                        len(
                            class_names
                        )
                    )
                },
            }
        )


    return results
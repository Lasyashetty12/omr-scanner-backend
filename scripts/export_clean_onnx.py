from pathlib import Path

import tensorflow as tf
import tf2onnx


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

KERAS_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "bubble_classifier.keras"
)

ONNX_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "bubble_classifier.onnx"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

trained_model = (
    tf.keras.models.load_model(
        KERAS_MODEL_PATH
    )
)


# ============================================================
# FIND THE ACTUAL CLASSIFIER
#
# Your training model was:
#
# input
#   ↓
# augmentation
#   ↓
# classifier
#
# We must REMOVE augmentation before ONNX export.
# ============================================================

classifier = None


for layer in trained_model.layers:

    print(
        "Layer:",
        layer.name,
        type(layer).__name__
    )

    # The nested Sequential CNN contains Conv2D layers.
    if isinstance(
        layer,
        tf.keras.Sequential
    ):

        has_conv = any(
            isinstance(
                child,
                tf.keras.layers.Conv2D
            )
            for child
            in layer.layers
        )

        if has_conv:

            classifier = layer

            break


if classifier is None:

    raise RuntimeError(
        "Could not find inference classifier inside trained model."
    )


print(
    "Found classifier:",
    classifier.name
)


# ============================================================
# BUILD CLEAN INFERENCE MODEL
# ============================================================

input_tensor = tf.keras.Input(
    shape=(
        48,
        48,
        1,
    ),
    name="bubble_input",
)


output_tensor = classifier(
    input_tensor,
    training=False,
)


inference_model = (
    tf.keras.Model(
        inputs=input_tensor,
        outputs=output_tensor,
        name="bubble_inference",
    )
)


# ============================================================
# TEST OUTPUT
# ============================================================

inference_model.summary()


# ============================================================
# EXPORT ONNX
# ============================================================

input_signature = [

    tf.TensorSpec(
        shape=(
            None,
            48,
            48,
            1,
        ),
        dtype=tf.float32,
        name="bubble_input",
    )

]


tf2onnx.convert.from_keras(

    inference_model,

    input_signature=input_signature,

    opset=13,

    output_path=str(
        ONNX_MODEL_PATH
    ),

)


print(
    "Clean ONNX model saved:",
    ONNX_MODEL_PATH
)
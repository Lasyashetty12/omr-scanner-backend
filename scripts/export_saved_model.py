import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "bubble_classifier.keras"
)

EXPORT_PATH = (
    BASE_DIR
    / "models"
    / "bubble_saved_model"
)

model = tf.keras.models.load_model(
    MODEL_PATH
)

model.export(
    EXPORT_PATH
)

print(
    "SavedModel exported to:",
    EXPORT_PATH
)
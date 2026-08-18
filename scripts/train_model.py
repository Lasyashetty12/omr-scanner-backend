
from pathlib import Path
import json
import tensorflow as tf
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml_omr.model import build_model, IMAGE_SIZE

TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "val"
MODEL_DIR = BASE_DIR / "models"

BATCH_SIZE = 64
EPOCHS = 30
SEED = 42

def load_dataset(path, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    class_names = ds.class_names

    ds = ds.map(
        lambda x, y: (
            tf.cast(x, tf.float32) / 255.0,
            y
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    return ds, class_names

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, class_names = load_dataset(TRAIN_DIR, True)
    val_ds, val_class_names = load_dataset(VAL_DIR, False)

    if class_names != val_class_names:
        raise RuntimeError(
            f"Class mismatch: {class_names} vs {val_class_names}"
        )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomTranslation(0.04, 0.04),
        tf.keras.layers.RandomContrast(0.15),
    ])

    classifier = build_model()

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1)
    )

    x = augmentation(inputs)
    outputs = classifier(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "bubble_classifier.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5,
            min_lr=1e-5,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    loss, accuracy = model.evaluate(val_ds, verbose=0)

    print(f"Validation accuracy: {accuracy:.4f}")
    print("Saved:", MODEL_DIR / "bubble_classifier.keras")

if __name__ == "__main__":
    main()

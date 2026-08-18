
"""
Build bubble-level training data from the synthetic mobile dataset.

Expected synthetic dataset structure:
    omr_synthetic_mobile_dataset/
        dataset.json
        neet/images/*.jpg
        kcet/images/*.jpg
        templates/neet.json
        templates/kcet.json

Usage:
    python scripts/build_training_dataset.py PATH_TO_SYNTHETIC_DATASET

Example:
    python scripts/build_training_dataset.py E:\datasets\omr_synthetic_mobile_dataset
"""

from pathlib import Path
import sys
import json
import random
import shutil
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "dataset"
TEMPLATES = BASE_DIR / "templates"

random.seed(42)

def generate_coordinates(template):
    total = int(template["total_questions"])
    per_col = int(template["questions_per_column"])
    options = template["options"]
    columns = template["columns"]
    ys = template["question_y_positions"]

    coords = {}
    for q in range(1, total + 1):
        ci = (q - 1) // per_col
        ri = (q - 1) % per_col
        y = int(ys[ri])
        coords[q] = {
            option: (int(columns[ci][option]), y)
            for option in options
        }
    return coords

def warp_from_label(image, page_quad, width, height):
    src = np.float32(page_quad)
    dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height))

def crop_square(gray, x, y, radius=16):
    h, w = gray.shape[:2]
    return gray[
        max(0, y-radius):min(h, y+radius+1),
        max(0, x-radius):min(w, x+radius+1),
    ]

def option_class(question_label, option):
    status = question_label["status"]
    answer = question_label.get("answer")

    if status == "blank":
        return "blank"

    if status == "answered":
        return "filled" if option == answer else "blank"

    if status == "multiple":
        return "filled" if option in (answer or []) else "blank"

    if status == "ambiguous":
        return "ambiguous" if option == answer else "blank"

    return "blank"

def ensure_dirs():
    for split in ("train", "val"):
        for cls in ("blank", "filled", "ambiguous"):
            (OUTPUT / split / cls).mkdir(parents=True, exist_ok=True)

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/build_training_dataset.py PATH_TO_SYNTHETIC_DATASET"
        )

    synthetic_root = Path(sys.argv[1]).resolve()
    dataset_json = synthetic_root / "dataset.json"

    if not dataset_json.exists():
        raise FileNotFoundError(dataset_json)

    ensure_dirs()

    records = json.loads(dataset_json.read_text(encoding="utf-8"))

    # Split by whole sheet, not individual crop.
    random.shuffle(records)
    split_index = int(len(records) * 0.80)

    train_records = records[:split_index]
    val_records = records[split_index:]

    for split, subset in (("train", train_records), ("val", val_records)):
        counters = {"blank": 0, "filled": 0, "ambiguous": 0}

        for sheet_index, record in enumerate(subset, 1):
            exam = record["exam"].lower()
            template_path = TEMPLATES / f"{exam}.json"

            template = json.loads(
                template_path.read_text(encoding="utf-8")
            )

            image_path = (
                synthetic_root
                / exam
                / "images"
                / record["image"]
            )

            image = cv2.imread(str(image_path))
            if image is None:
                print("Skipping unreadable:", image_path)
                continue

            width = int(template["sheet_width"])
            height = int(template["sheet_height"])

            corrected = warp_from_label(
                image=image,
                page_quad=record["conditions"]["page_quad"],
                width=width,
                height=height,
            )

            gray = cv2.cvtColor(
                corrected,
                cv2.COLOR_BGR2GRAY,
            )

            coordinates = generate_coordinates(template)

            for q, option_map in coordinates.items():
                q_label = record["answers"][str(q)]

                for option, (x, y) in option_map.items():
                    cls = option_class(q_label, option)

                    crop = crop_square(
                        gray,
                        x,
                        y,
                        radius=16,
                    )

                    filename = (
                        f"{exam}_sheet{sheet_index:04d}"
                        f"_q{q:03d}_{option}.png"
                    )

                    out_path = OUTPUT / split / cls / filename
                    cv2.imwrite(str(out_path), crop)
                    counters[cls] += 1

        print(split, counters)

if __name__ == "__main__":
    main()

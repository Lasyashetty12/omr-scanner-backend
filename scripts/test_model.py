
from pathlib import Path
import sys
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml_omr.inference import classify_batch

def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python scripts/test_model.py bubble1.png [bubble2.png ...]"
        )

    crops = []

    for path in sys.argv[1:]:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Could not read:", path)
            continue

        crops.append(image)

    results = classify_batch(crops)

    for path, result in zip(sys.argv[1:], results):
        print(path)
        print(result)
        print()

if __name__ == "__main__":
    main()

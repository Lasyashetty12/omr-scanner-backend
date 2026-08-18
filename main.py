# main.py

import os
import json
import cv2

from scanner import process_omr
from scorer import calculate_score
from answer_key import ANSWER_KEY


INPUT_IMAGE = "input/sample.jpg"
OUTPUT_DIRECTORY = "output"


def create_output_directory():
    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )


def save_debug_images(
    processing_result
):
    cv2.imwrite(
        os.path.join(
            OUTPUT_DIRECTORY,
            "01_marker_detection.jpg"
        ),
        processing_result[
            "marker_debug"
        ]
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIRECTORY,
            "02_corrected_sheet.jpg"
        ),
        processing_result[
            "corrected"
        ]
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIRECTORY,
            "03_threshold.jpg"
        ),
        processing_result[
            "threshold"
        ]
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIRECTORY,
            "04_bubble_debug.jpg"
        ),
        processing_result[
            "debug"
        ]
    )


def print_answer_details(
    detected_answers
):
    print()
    print(
        "=" * 75
    )

    print(
        "DETECTED BUBBLES"
    )

    print(
        "=" * 75
    )

    for question_number in sorted(
        detected_answers.keys()
    ):
        data = detected_answers[
            question_number
        ]

        scores = data[
            "scores"
        ]

        print(
            f"Q{question_number:02d}  "
            f"A={scores['A']:.2f}  "
            f"B={scores['B']:.2f}  "
            f"C={scores['C']:.2f}  "
            f"D={scores['D']:.2f}  "
            f"=> {data['answer']}"
        )


def print_final_result(
    result
):
    print()
    print(
        "=" * 75
    )

    print(
        "QUESTION-WISE RESULT"
    )

    print(
        "=" * 75
    )

    for question_number in sorted(
        result[
            "questions"
        ].keys()
    ):
        item = result[
            "questions"
        ][question_number]

        print(
            f"Q{question_number:02d} "
            f"Detected={item['detected']:8} "
            f"Key={item['correct_answer']} "
            f"Status={item['status']:8} "
            f"Marks={item['marks']}"
        )

    print()
    print(
        "=" * 75
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 75
    )

    print(
        f"Correct  : "
        f"{result['correct']}"
    )

    print(
        f"Wrong    : "
        f"{result['wrong']}"
    )

    print(
        f"Blank    : "
        f"{result['blank']}"
    )

    print(
        f"Multiple : "
        f"{result['multiple']}"
    )

    print(
        f"Marks    : "
        f"{result['score']}"
    )

    print(
        "=" * 75
    )


def save_json_result(
    result
):
    output_file = os.path.join(
        OUTPUT_DIRECTORY,
        "result.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4
        )


def main():
    create_output_directory()

    try:
        processing_result = (
            process_omr(
                INPUT_IMAGE
            )
        )

        detected_answers = (
            processing_result[
                "answers"
            ]
        )

        result = calculate_score(
            detected_answers,
            ANSWER_KEY
        )

        save_debug_images(
            processing_result
        )

        print_answer_details(
            detected_answers
        )

        print_final_result(
            result
        )

        save_json_result(
            result
        )

        print()
        print(
            "Files saved inside output/"
        )

    except Exception as error:
        print()
        print(
            "OMR processing failed."
        )

        print(
            f"Reason: {error}"
        )


if __name__ == "__main__":
    main()
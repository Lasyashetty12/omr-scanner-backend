# app.py

import json
import os
import uuid

import cv2

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Depends,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from config import (
    BASE_DIR,
    UPLOAD_DIR,
    RESULT_DIR,
    TEMPLATE_DIR,
    STATIC_DIR,
)

from database import init_db, get_db
from database import OMRResult, QuestionResult
from db_operations import save_complete_omr_scan

from scanner import process_omr

from scorer import (
    calculate_score,
    calculate_jee_score,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="OMR Scanner API",
    version="2.1.0",
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()


# ============================================================
# PATHS
# ============================================================

ANSWER_KEY_DIR = os.path.join(
    BASE_DIR,
    "answer_keys",
)


# ============================================================
# STATIC FILES
# ============================================================

if os.path.exists(STATIC_DIR):

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR
        ),
        name="static",
    )


# ============================================================
# HELPERS
# ============================================================

def safe_filename(name):

    return os.path.basename(
        str(name)
    )


def save_json(
    path,
    data,
):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def load_answer_key_for_exam(
    exam_name,
    identifier,
):

    exam_name = (
        str(exam_name)
        .strip()
        .lower()
    )

    identifier = (
        safe_filename(
            str(identifier)
            .strip()
            .upper()
        )
    )

    if not identifier:

        raise ValueError(
            "Answer key identifier is empty."
        )

    answer_key_path = os.path.join(
        ANSWER_KEY_DIR,
        exam_name,
        f"{identifier}.json",
    )

    if not os.path.exists(
        answer_key_path
    ):

        raise ValueError(
            f"No answer key found for "
            f"{exam_name.upper()} "
            f"paper/series {identifier}."
        )

    try:

        with open(
            answer_key_path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

    except json.JSONDecodeError:

        raise ValueError(
            f"Invalid answer key JSON: "
            f"{answer_key_path}"
        )

    return data


def save_debug_images(
    scan_id,
    processing,
):

    def _fast_write(path, img):
        if img is None:
            return
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    corrected = processing.get("corrected")
    if corrected is not None:
        _fast_write(os.path.join(RESULT_DIR, f"{scan_id}_corrected.jpg"), corrected)

    debug_img = processing.get("debug")
    if debug_img is not None:
        _fast_write(os.path.join(RESULT_DIR, f"{scan_id}_bubble_debug.jpg"), debug_img)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    index_path = os.path.join(
        STATIC_DIR,
        "index.html",
    )

    if os.path.exists(
        index_path
    ):

        return FileResponse(
            index_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )

    return {
        "status": "ok",
        "message": "OMR Scanner API is running",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "OMR Scanner",
        "version": "2.1.0",
    }


# ============================================================
# SCAN OMR
# ============================================================

@app.post("/scan")
async def scan_omr(

    image: UploadFile = File(...),

    exam: str = Form(...),

    stream: str = Form("pcmb"),

    db: Session = Depends(get_db),

):

    # ========================================================
    # VALIDATE EXAM
    # ========================================================

    exam = (
        exam
        .strip()
        .lower()
    )

    allowed_exams = [
        "neet",
        "kcet",
        "jee",
    ]

    if exam not in allowed_exams:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exam must be "
                "NEET, KCET or JEE."
            ),
        )


    # ========================================================
    # TEMPLATE
    # ========================================================

    template_path = os.path.join(
        TEMPLATE_DIR,
        f"{exam}.json",
    )

    if not os.path.exists(
        template_path
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                f"Template not found "
                f"for {exam.upper()}."
            ),
        )


    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    original_filename = (
        image.filename
        or "camera_omr.jpg"
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    allowed_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
    ]

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG and PNG "
                "images are supported."
            ),
        )


    # ========================================================
    # CREATE SCAN ID
    # ========================================================

    scan_id = str(
        uuid.uuid4()
    )

    upload_filename = (
        f"{scan_id}{extension}"
    )

    upload_path = os.path.join(
        UPLOAD_DIR,
        upload_filename,
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        contents = await image.read()

        if not contents:

            raise ValueError(
                "Captured image is empty."
            )

        with open(
            upload_path,
            "wb",
        ) as file:

            file.write(
                contents
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save "
                "captured OMR image: "
                f"{str(error)}"
            ),
        )


    # ========================================================
    # PROCESS OMR
    # ========================================================

    try:

        processing = process_omr(
            contents,
            template_path,
            input_filename=original_filename,
            input_mime_type=image.content_type,
            diagnostic_dir=(
                os.path.join(RESULT_DIR, f"{scan_id}_input_diagnostics")
                if os.environ.get("OMR_DEBUG_INPUT")
                else None
            ),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "OMR processing failed: "
                f"{str(error)}"
            ),
        )


    # ========================================================
    # BASIC RESULT
    # ========================================================

    result = {

        "scan_id":
            scan_id,

        "status":
            "processed",

        "exam":
            exam.upper(),

        "quality":
            processing.get(
                "quality"
            ),

        "input": processing.get("input_debug"),

    }


    # ========================================================
    # NEET
    # ========================================================

    if exam == "neet":

        paper_code_data = (
            processing.get(
                "paper_code"
            )
        )

        if not paper_code_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "NEET question paper "
                    "code could not be detected."
                ),
            )


        paper_code = (
            paper_code_data.get(
                "value"
            )
        )


        if not paper_code:

            raise HTTPException(
                status_code=400,
                detail=(
                    "NEET question paper "
                    "code is empty."
                ),
            )


        # ----------------------------------------------------
        # AUTO LOAD ANSWER KEY
        # ----------------------------------------------------

        try:

            answer_key_data = (
                load_answer_key_for_exam(
                    "neet",
                    paper_code,
                )
            )

        except ValueError as error:

            raise HTTPException(
                status_code=404,
                detail=str(error),
            )


        if "answers" not in answer_key_data:

            raise HTTPException(
                status_code=500,
                detail=(
                    "NEET answer key does not "
                    "contain 'answers'."
                ),
            )


        detected_answers = (
            processing.get(
                "answers",
                {},
            )
        )


        marking = (
            answer_key_data.get(
                "marking",
                {},
            )
        )


        score_data = calculate_score(

            detected_answers=
                detected_answers,

            answer_key=
                answer_key_data[
                    "answers"
                ],

            correct_marks=
                marking.get(
                    "correct",
                    4,
                ),

            wrong_marks=
                marking.get(
                    "wrong",
                    -1,
                ),

            blank_marks=
                marking.get(
                    "blank",
                    0,
                ),

            multiple_marks=
                marking.get(
                    "multiple",
                    -1,
                ),

        )


        result.update(
            {

                "paper_code":
                    paper_code,

                "paper_code_details":
                    paper_code_data,

                "score":
                    score_data[
                        "score"
                    ],

                "correct":
                    score_data[
                        "correct"
                    ],

                "wrong":
                    score_data[
                        "wrong"
                    ],

                "blank":
                    score_data[
                        "blank"
                    ],

                "multiple":
                    score_data[
                        "multiple"
                    ],

                "uncertain":
                    score_data[
                        "uncertain"
                    ],

                "answers":
                    detected_answers,

                "question_results":
                    score_data[
                        "questions"
                    ],

            }
        )


    # ========================================================
    # KCET
    # ========================================================

    elif exam == "kcet":

        paper_code_data = (
            processing.get(
                "paper_code"
            )
        )


        if not paper_code_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "KCET question paper "
                    "code could not be detected."
                ),
            )


        paper_code = (
            paper_code_data.get(
                "value"
            )
        )


        if not paper_code:

            raise HTTPException(
                status_code=400,
                detail=(
                    "KCET question paper "
                    "code is empty."
                ),
            )


        # ----------------------------------------------------
        # AUTO LOAD ANSWER KEY
        # ----------------------------------------------------

        try:

            answer_key_data = (
                load_answer_key_for_exam(
                    "kcet",
                    paper_code,
                )
            )

        except ValueError as error:

            raise HTTPException(
                status_code=404,
                detail=str(error),
            )


        if "answers" not in answer_key_data:

            raise HTTPException(
                status_code=500,
                detail=(
                    "KCET answer key does not "
                    "contain 'answers'."
                ),
            )


        detected_answers = (
            processing.get(
                "answers",
                {},
            )
        )


        marking = (
            answer_key_data.get(
                "marking",
                {},
            )
        )


        kcet_answer_key = answer_key_data["answers"]
        if stream and stream.lower().strip() == "pcm":
            kcet_answer_key = {
                k: v for k, v in kcet_answer_key.items()
                if int(k) <= 180
            }

        score_data = calculate_score(

            detected_answers=
                detected_answers,

            answer_key=
                kcet_answer_key,

            correct_marks=
                marking.get(
                    "correct",
                    1,
                ),

            wrong_marks=
                marking.get(
                    "wrong",
                    0,
                ),

            blank_marks=
                marking.get(
                    "blank",
                    0,
                ),

            multiple_marks=
                marking.get(
                    "multiple",
                    0,
                ),

        )


        result.update(
            {

                "stream":
                    (stream or "PCMB").upper(),

                "paper_code":
                    paper_code,

                "paper_code_details":
                    paper_code_data,

                "score":
                    score_data[
                        "score"
                    ],

                "correct":
                    score_data[
                        "correct"
                    ],

                "wrong":
                    score_data[
                        "wrong"
                    ],

                "blank":
                    score_data[
                        "blank"
                    ],

                "multiple":
                    score_data[
                        "multiple"
                    ],

                "uncertain":
                    score_data[
                        "uncertain"
                    ],

                "answers":
                    detected_answers,

                "question_results":
                    score_data[
                        "questions"
                    ],

            }
        )


    # ========================================================
    # JEE
    # ========================================================

    elif exam == "jee":

        series_data = (
            processing.get(
                "jee_series"
            )
        )


        if not series_data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "JEE series/code could "
                    "not be detected."
                ),
            )


        series = (
            series_data.get(
                "value"
            )
        )


        if not series:

            raise HTTPException(
                status_code=400,
                detail=(
                    "JEE series/code is empty."
                ),
            )


        # ----------------------------------------------------
        # AUTO LOAD ANSWER KEY
        # ----------------------------------------------------

        try:

            answer_key_data = (
                load_answer_key_for_exam(
                    "jee",
                    series,
                )
            )

        except ValueError as error:

            raise HTTPException(
                status_code=404,
                detail=str(error),
            )


        detected = (
            processing.get(
                "answers",
                {},
            )
        )


        mcq_detected = (
            detected.get(
                "mcq",
                {},
            )
        )


        numerical_detected = (
            detected.get(
                "numerical",
                {},
            )
        )


        # ----------------------------------------------------
        # JEE SCORING
        # ----------------------------------------------------

        try:

            score_data = calculate_jee_score(

                detected_answers=
                    detected,

                answer_key=
                    answer_key_data,

            )

        except Exception as error:

            # Allows calibration/testing even if
            # JEE answer key format is incomplete.

            result.update(
                {

                    "series":
                        series,

                    "series_details":
                        series_data,

                    "mcq_answers":
                        mcq_detected,

                    "numerical_answers":
                        numerical_detected,

                    "score":
                        None,

                    "correct":
                        None,

                    "wrong":
                        None,

                    "blank":
                        None,

                    "multiple":
                        None,

                    "message":
                        (
                            "JEE sheet detected, "
                            "but scoring could not "
                            "be completed: "
                            f"{str(error)}"
                        ),

                }
            )

        else:

            result.update(
                {

                    "series":
                        series,

                    "series_details":
                        series_data,

                    "score":
                        score_data.get(
                            "score"
                        ),

                    "correct":
                        score_data.get(
                            "correct"
                        ),

                    "wrong":
                        score_data.get(
                            "wrong"
                        ),

                    "blank":
                        score_data.get(
                            "blank"
                        ),

                    "multiple":
                        score_data.get(
                            "multiple",
                            0,
                        ),

                    "mcq_answers":
                        mcq_detected,

                    "numerical_answers":
                        numerical_detected,

                    "score_details":
                        score_data,

                }
            )


    # ========================================================
    # SAVE DEBUG IMAGES
    # ========================================================

    try:

        save_debug_images(
            scan_id,
            processing,
        )

    except Exception as error:

        print(
            "Debug image save warning:",
            error,
        )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_path = os.path.join(
        RESULT_DIR,
        f"{scan_id}.json",
    )


    try:

        save_json(
            result_path,
            result,
        )

    except Exception as error:

        print(
            "Result save warning:",
            error,
        )


    # ========================================================
    # SAVE TO DATABASE
    # ========================================================

    try:

        omr_result = save_complete_omr_scan(
            db,
            scanner_result=result,
            student_name=None,
            student_roll_number=None,
            student_class=None,
            student_section=None,
            capture_source="camera",
        )

        if not omr_result or not omr_result.id:
            raise ValueError("Database insertion returned no valid result ID.")

        result["database_id"] = omr_result.id

    except Exception as db_error:

        print(
            "Database save failure:",
            db_error,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Database persistence failed: {str(db_error)}",
        )


    result["original_image_url"] = f"/uploads/{upload_filename}"
    result["corrected_image_url"] = f"/uploads/{upload_filename}"
    result["bubble_debug_image_url"] = f"/results/{scan_id}_bubble_debug.jpg"

    return result


# ============================================================
# GET UPLOADED IMAGE (AS CLICKED)
# ============================================================

@app.get(
    "/uploads/{filename}"
)
def get_upload_image(
    filename: str,
):

    filename = safe_filename(
        filename
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        filename,
    )

    if not os.path.exists(
        image_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Uploaded image not found.",
        )

    return FileResponse(
        image_path,
    )


# ============================================================
# GET RESULT IMAGE
# ============================================================

@app.get(
    "/results/{filename}"
)
def get_result_image(
    filename: str,
):

    filename = safe_filename(
        filename
    )

    image_path = os.path.join(
        RESULT_DIR,
        filename,
    )

    if not os.path.exists(
        image_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Result image not found.",
        )

    return FileResponse(
        image_path,
        media_type="image/jpeg",
    )


# ============================================================
# GET RESULT & RESULT PAGE
# ============================================================

@app.get(
    "/result.html"
)
def get_result_html_page():
    result_html_path = os.path.join(
        STATIC_DIR,
        "result.html",
    )
    if os.path.exists(result_html_path):
        return FileResponse(
            result_html_path,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    raise HTTPException(
        status_code=404,
        detail="Result page template not found.",
    )


@app.get(
    "/result/{scan_id}"
)
def get_result(
    scan_id: str,
):
    # If scan_id is a numeric DB result ID, serve the dedicated HTML result page
    if scan_id.isdigit():
        result_html_path = os.path.join(
            STATIC_DIR,
            "result.html",
        )
        if os.path.exists(result_html_path):
            return FileResponse(
                result_html_path,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )

    scan_id_safe = safe_filename(
        scan_id
    )

    result_path = os.path.join(
        RESULT_DIR,
        f"{scan_id_safe}.json",
    )

    if not os.path.exists(
        result_path
    ):
        raise HTTPException(
            status_code=404,
            detail="Result not found.",
        )

    with open(
        result_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


# ============================================================
# API INFO
# ============================================================

@app.get("/api")
def api_info():

    return {

        "service":
            "OMR Scanner",

        "version":
            "2.1.0",

        "supported_exams": [
            "NEET",
            "JEE",
            "KCET",
        ],

        "workflow":
            (
                "Select exam -> "
                "capture OMR using camera -> "
                "detect paper code/series -> "
                "load answer key automatically -> "
                "calculate score"
            ),

        "endpoints": {

            "scan":
                "POST /scan",

            "health":
                "GET /health",

            "result":
                "GET /result/{scan_id}",

        },

    }


# ============================================================
# DATABASE QUERY ENDPOINTS
# ============================================================

@app.get("/api/omr-results")
def get_omr_results(
    class_name: str = None,
    section: str = None,
    exam_type: str = None,
    db: Session = Depends(get_db),
):
    """
    Get OMR results with optional filters.
    """
    from database import Student, Exam
    
    query = db.query(
        OMRResult.id,
        OMRResult.score,
        OMRResult.correct,
        OMRResult.wrong,
        OMRResult.blank,
        OMRResult.multiple,
        OMRResult.uncertain,
        OMRResult.created_at,
        Student.name,
        Student.roll_number,
        Student.class_name,
        Student.section,
        Exam.exam_type,
        Exam.paper_code,
    ).join(
        Student,
        OMRResult.student_id == Student.id
    ).join(
        Exam,
        OMRResult.exam_id == Exam.id
    )
    
    if class_name and class_name != "all":
        query = query.filter(Student.class_name == class_name)
    
    if section and section != "all":
        query = query.filter(Student.section == section)
    
    if exam_type and exam_type != "all":
        query = query.filter(Exam.exam_type == exam_type.upper())
    
    results = query.order_by(OMRResult.created_at.desc()).all()
    
    return [
        {
            "id": row.id,
            "student_name": row.name,
            "roll_number": row.roll_number,
            "class": row.class_name,
            "section": row.section,
            "exam": row.exam_type,
            "paper_code": row.paper_code,
            "score": row.score,
            "correct": row.correct,
            "wrong": row.wrong,
            "blank": row.blank,
            "multiple": row.multiple,
            "uncertain": row.uncertain,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in results
    ]


@app.get("/api/omr-results/{result_id}")
def get_omr_result_detail(
    result_id: int,
    db: Session = Depends(get_db),
):
    """
    Get detailed OMR result including question-wise breakdown.
    """
    from database import Student, Exam
    
    result = db.query(OMRResult).filter(
        OMRResult.id == result_id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Result not found",
        )
    
    student = db.query(Student).filter(
        Student.id == result.student_id
    ).first()
    
    exam = db.query(Exam).filter(
        Exam.id == result.exam_id
    ).first()
    
    questions = db.query(QuestionResult).filter(
        QuestionResult.omr_result_id == result_id
    ).order_by(QuestionResult.question_number).all()
    
    return {
        "id": result.id,
        "student": {
            "name": student.name if (student and student.name) else "Student",
            "roll_number": student.roll_number if (student and student.roll_number) else "-",
            "class": student.class_name if (student and student.class_name) else "-",
            "section": student.section if (student and student.section) else "-",
            "batch": student.batch if (student and student.batch) else "-",
        },
        "exam": {
            "type": (exam.exam_type if (exam and exam.exam_type) else "OMR Exam").upper(),
            "paper_code": exam.paper_code if (exam and exam.paper_code) else "-",
            "paper_series": exam.paper_series if (exam and exam.paper_series) else "-",
            "exam_date": exam.exam_date.isoformat() if (exam and exam.exam_date) else None,
            "session": exam.session if (exam and exam.session) else "-",
        },
        "performance": {
            "score": result.score,
            "correct": result.correct,
            "wrong": result.wrong,
            "blank": result.blank,
            "multiple": result.multiple,
            "uncertain": result.uncertain,
            "total_questions": result.total_questions or len(questions),
            "stream": result.stream or "PCMB",
        },
        "questions": [
            {
                "question_number": q.question_number,
                "marked_answer": q.marked_answer,
                "correct_answer": q.correct_answer,
                "status": q.status,
            }
            for q in questions
        ],
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }


@app.get("/api/classes")
def get_classes(db: Session = Depends(get_db)):
    """
    Get list of unique classes from students.
    """
    from database import Student
    
    classes = db.query(Student.class_name).distinct().filter(
        Student.class_name.isnot(None)
    ).all()
    
    return [c[0] for c in sorted(classes) if c[0]]


@app.get("/api/sections")
def get_sections(db: Session = Depends(get_db)):
    """
    Get list of unique sections from students.
    """
    from database import Student
    
    sections = db.query(Student.section).distinct().filter(
        Student.section.isnot(None)
    ).all()
    
    return [s[0] for s in sorted(sections) if s[0]]


@app.get("/api/exams")
def get_exams(db: Session = Depends(get_db)):
    """
    Get list of unique exam types.
    """
    from database import Exam
    
    exams = db.query(Exam.exam_type).distinct().all()
    
    return [e[0] for e in sorted(exams) if e[0]]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
    app,
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
)


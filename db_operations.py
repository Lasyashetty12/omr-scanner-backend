"""
Database operations for OMR results.
"""

import json
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from database import (
    Student,
    Exam,
    OMRResult,
    QuestionResult,
    Scan,
)


def get_or_create_student(
    db: Session,
    name: Optional[str] = None,
    roll_number: Optional[str] = None,
    class_name: Optional[str] = None,
    section: Optional[str] = None,
    batch: Optional[str] = None,
) -> Student:
    """
    Get or create a student record.
    
    Avoid duplicates by checking unique constraint on (roll_number, class_name).
    """
    
    # If we have roll_number and class, use that as the unique key
    if roll_number and class_name:
        student = db.query(Student).filter(
            Student.roll_number == roll_number,
            Student.class_name == class_name,
        ).first()
        
        if student:
            return student
    
    # Create new student
    student = Student(
        name=name,
        roll_number=roll_number,
        class_name=class_name,
        section=section,
        batch=batch,
    )
    
    db.add(student)
    db.flush()
    
    return student


def get_or_create_exam(
    db: Session,
    exam_type: str,
    paper_series: Optional[str] = None,
    paper_code: Optional[str] = None,
    exam_date: Optional[str] = None,
    session: Optional[str] = None,
) -> Exam:
    """
    Get or create an exam record.
    
    Avoid duplicates by checking unique constraint on (exam_type, paper_code).
    """
    
    # If we have exam_type and paper_code, try to find existing exam
    if exam_type and paper_code:
        exam = db.query(Exam).filter(
            Exam.exam_type == exam_type,
            Exam.paper_code == paper_code,
        ).first()
        
        if exam:
            return exam
    
    # Create new exam
    exam = Exam(
        exam_type=exam_type,
        paper_series=paper_series,
        paper_code=paper_code,
        exam_date=exam_date,
        session=session,
    )
    
    db.add(exam)
    db.flush()
    
    return exam


def save_omr_result(
    db: Session,
    student_id: int,
    exam_id: int,
    scanner_result: Dict[str, Any],
) -> OMRResult:
    """
    Save OMR result from scanner to database.
    
    The scanner_result is the final evaluation from scanner.py.
    Copy values directly without recalculation.
    """
    
    # Extract scores directly from scanner result
    score = scanner_result.get("score", 0)
    correct = scanner_result.get("correct", 0)
    wrong = scanner_result.get("wrong", 0)
    blank = scanner_result.get("blank", 0)
    multiple = scanner_result.get("multiple", 0)
    uncertain = scanner_result.get("uncertain", 0)
    stream = scanner_result.get("stream")
    
    # Calculate total from question results if available
    question_results = scanner_result.get("question_results", {})
    total_questions = len(question_results) if question_results else None
    
    # Store the entire result as JSON for reference
    raw_result_json = json.dumps(scanner_result)
    
    omr_result = OMRResult(
        student_id=student_id,
        exam_id=exam_id,
        score=score,
        correct=correct,
        wrong=wrong,
        blank=blank,
        multiple=multiple,
        uncertain=uncertain,
        total_questions=total_questions,
        stream=stream,
        raw_result_json=raw_result_json,
    )
    
    db.add(omr_result)
    db.flush()
    
    return omr_result


def save_question_results(
    db: Session,
    omr_result_id: int,
    scanner_result: Dict[str, Any],
) -> None:
    """
    Save question-wise results from scanner.
    
    Extract from scanner_result["question_results"] which contains
    the answer key validation for each question.
    """
    
    question_results = scanner_result.get("question_results")
    if not question_results and "answers" in scanner_result:
        answers_dict = scanner_result.get("answers", {})
        if isinstance(answers_dict, dict):
            question_results = {}
            for q_num, ans_data in answers_dict.items():
                if isinstance(ans_data, dict):
                    question_results[str(q_num)] = {
                        "marked": ans_data.get("marked") or ans_data.get("answer"),
                        "answer": ans_data.get("answer") or ans_data.get("correct_answer"),
                        "status": ans_data.get("status", "unknown")
                    }
                else:
                    question_results[str(q_num)] = {
                        "marked": ans_data,
                        "answer": "unknown",
                        "status": "unknown"
                    }
    
    if not question_results:
        return
    
    for question_num_str, question_data in question_results.items():
        try:
            question_num = int(question_num_str)
        except (ValueError, TypeError):
            continue
        
        # Status is already determined by scanner
        status = question_data.get("status", "unknown")
        marked = question_data.get("marked")
        correct_answer = question_data.get("answer") or question_data.get("correct_answer")
        
        result = QuestionResult(
            omr_result_id=omr_result_id,
            question_number=question_num,
            marked_answer=str(marked) if (marked is not None and marked != "None") else None,
            correct_answer=str(correct_answer) if (correct_answer and correct_answer != "None") else "unknown",
            status=status,
        )
        
        db.add(result)
    
    db.flush()


def save_scan_metadata(
    db: Session,
    omr_result_id: int,
    capture_source: str = "camera",
    image_reference: Optional[str] = None,
) -> Scan:
    """
    Save scan metadata (capture source, image reference).
    """
    
    scan = Scan(
        omr_result_id=omr_result_id,
        capture_source=capture_source,
        image_reference=image_reference,
    )
    
    db.add(scan)
    db.flush()
    
    return scan


def save_complete_omr_scan(
    db: Session,
    scanner_result: Dict[str, Any],
    student_name: Optional[str] = None,
    student_roll_number: Optional[str] = None,
    student_class: Optional[str] = None,
    student_section: Optional[str] = None,
    capture_source: str = "camera",
) -> OMRResult:
    """
    Atomic transaction to save complete OMR scan result.
    
    Either all data saves successfully, or the transaction is rolled back.
    """
    
    try:
        # Extract exam info from result
        exam_type = scanner_result.get("exam", "UNKNOWN").upper()
        paper_code = scanner_result.get("paper_code") or scanner_result.get("series")
        
        # Get or create student
        student = get_or_create_student(
            db,
            name=student_name,
            roll_number=student_roll_number,
            class_name=student_class,
            section=student_section,
        )
        
        # Get or create exam
        exam = get_or_create_exam(
            db,
            exam_type=exam_type,
            paper_code=paper_code,
        )
        
        # Save OMR result
        omr_result = save_omr_result(
            db,
            student.id,
            exam.id,
            scanner_result,
        )
        
        # Save question-wise results
        save_question_results(
            db,
            omr_result.id,
            scanner_result,
        )
        
        # Save scan metadata
        save_scan_metadata(
            db,
            omr_result.id,
            capture_source=capture_source,
        )
        
        # Commit the transaction
        db.commit()
        
        return omr_result
        
    except Exception as error:
        db.rollback()
        raise error

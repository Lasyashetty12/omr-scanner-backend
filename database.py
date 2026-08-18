"""
Database models and initialization for OMR Scanner persistence.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
)

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import DATABASE_URL


# ============================================================
# DATABASE SETUP
# ============================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ============================================================
# MODELS
# ============================================================

class Student(Base):
    """
    Student information.
    """
    __tablename__ = "students"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(
        String(255),
        nullable=True,
    )

    roll_number = Column(
        String(50),
        nullable=True,
        index=True,
    )

    class_name = Column(
        String(50),
        nullable=True,
        index=True,
    )

    section = Column(
        String(50),
        nullable=True,
        index=True,
    )

    batch = Column(
        String(100),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "roll_number",
            "class_name",
            name="uq_roll_class"
        ),
    )


class Exam(Base):
    """
    Exam/Test information.
    """
    __tablename__ = "exams"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    exam_type = Column(
        String(50),
        nullable=False,
        index=True,
    )

    paper_series = Column(
        String(100),
        nullable=True,
    )

    paper_code = Column(
        String(100),
        nullable=True,
    )

    exam_date = Column(
        DateTime,
        nullable=True,
    )

    session = Column(
        String(100),
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "exam_type",
            "paper_code",
            name="uq_exam_paper"
        ),
    )


class OMRResult(Base):
    """
    OMR evaluation result from scanner.
    
    Values come directly from scanner.py result object.
    Do NOT recalculate scoring here.
    """
    __tablename__ = "omr_results"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    student_id = Column(
        Integer,
        ForeignKey("students.id"),
        nullable=False,
        index=True,
    )

    exam_id = Column(
        Integer,
        ForeignKey("exams.id"),
        nullable=False,
        index=True,
    )

    score = Column(
        Float,
        nullable=False,
    )

    correct = Column(
        Integer,
        nullable=False,
    )

    wrong = Column(
        Integer,
        nullable=False,
    )

    blank = Column(
        Integer,
        nullable=False,
    )

    multiple = Column(
        Integer,
        nullable=False,
    )

    uncertain = Column(
        Integer,
        nullable=False,
    )

    total_questions = Column(
        Integer,
        nullable=True,
    )

    stream = Column(
        String(50),
        nullable=True,
    )

    raw_result_json = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )


class QuestionResult(Base):
    """
    Question-wise OMR evaluation result.
    
    Stores the exact marked and correct answers as determined
    by the existing scanner logic.
    """
    __tablename__ = "question_results"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    omr_result_id = Column(
        Integer,
        ForeignKey("omr_results.id"),
        nullable=False,
        index=True,
    )

    question_number = Column(
        Integer,
        nullable=False,
    )

    marked_answer = Column(
        String(10),
        nullable=True,
    )

    correct_answer = Column(
        String(10),
        nullable=False,
    )

    status = Column(
        String(20),
        nullable=False,
    )


class Scan(Base):
    """
    Scan metadata about the capture itself.
    """
    __tablename__ = "scans"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    omr_result_id = Column(
        Integer,
        ForeignKey("omr_results.id"),
        nullable=False,
        index=True,
    )

    image_reference = Column(
        String(255),
        nullable=True,
    )

    capture_source = Column(
        String(20),
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    Dependency for FastAPI route handlers.
    Yields a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Export models and session for use in other modules
__all__ = [
    "engine",
    "SessionLocal",
    "Base",
    "Student",
    "Exam",
    "OMRResult",
    "QuestionResult",
    "Scan",
    "init_db",
    "get_db",
]

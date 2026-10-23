"""SQLite persistence and reporting for independently verified image checks."""

import csv
import io
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import DB_PATH, TARGET_CLASSES

REVIEW_PENDING = "Awaiting expert review"
REVIEW_STATUSES = {
    REVIEW_PENDING,
    "Confirmed",
    "Incorrect",
    "Needs more information",
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS result_validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_ref TEXT NOT NULL UNIQUE,
    stored_filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    crop TEXT NOT NULL,
    predicted_class TEXT NOT NULL,
    predicted_display_name TEXT NOT NULL,
    prediction_confidence REAL NOT NULL,
    suggested_reference_label TEXT,
    verified_label TEXT,
    label_source TEXT,
    source_description TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expert_review_status TEXT NOT NULL DEFAULT 'Awaiting expert review',
    expert_diagnosis TEXT,
    expert_comments TEXT,
    expert_reviewed_at TEXT,
    reviewer_identity TEXT
);
CREATE INDEX IF NOT EXISTS idx_validation_created ON result_validations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_validation_crop ON result_validations(crop);
CREATE INDEX IF NOT EXISTS idx_validation_verified ON result_validations(verified_label);
"""


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _connection(db_path: Optional[str] = None):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_validation_db(db_path: Optional[str] = None) -> None:
    """Create the additive validation table; does not alter existing tables."""
    with _connection(db_path) as conn:
        conn.executescript(SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(result_validations)")}
        if "suggested_reference_label" not in columns:
            conn.execute("ALTER TABLE result_validations ADD COLUMN suggested_reference_label TEXT")


def add_validation_record(
    *,
    image_ref: str,
    stored_filename: str,
    original_filename: str,
    crop: str,
    predicted_class: str,
    predicted_display_name: str,
    prediction_confidence: float,
    source_description: str,
    suggested_reference_label: Optional[str] = None,
    verified_label: Optional[str] = None,
    label_source: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    init_validation_db(db_path)
    with _connection(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO result_validations
            (image_ref, stored_filename, original_filename, crop, predicted_class,
             predicted_display_name, prediction_confidence, suggested_reference_label,
             verified_label, label_source, source_description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                image_ref,
                stored_filename,
                original_filename,
                crop,
                predicted_class,
                predicted_display_name,
                float(prediction_confidence),
                suggested_reference_label,
                verified_label,
                label_source,
                source_description.strip(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        return int(cursor.lastrowid)


def confirm_reference_label(
    record_id: int,
    verified_label: str,
    label_source: str,
    db_path: Optional[str] = None,
) -> None:
    """Record a human-confirmed label and provenance, never an inferred label."""
    label = verified_label.strip()
    source = label_source.strip()
    if not label or not source:
        raise ValueError("A verified label and its source are both required.")
    if len(label) > 180 or len(source) > 500:
        raise ValueError("Verified label or source is too long.")
    with _connection(db_path) as conn:
        cursor = conn.execute(
            "UPDATE result_validations SET verified_label = ?, label_source = ? WHERE id = ?",
            (label, source, record_id),
        )
        if cursor.rowcount != 1:
            raise LookupError("Validation record was not found.")


def record_expert_review(
    record_id: int,
    status: str,
    diagnosis: str,
    comments: str,
    reviewer_identity: str,
    db_path: Optional[str] = None,
) -> None:
    if status not in REVIEW_STATUSES - {REVIEW_PENDING}:
        raise ValueError("Choose a completed expert review status.")
    diagnosis = diagnosis.strip()
    reviewer_identity = reviewer_identity.strip()
    comments = comments.strip()
    if not diagnosis or not reviewer_identity:
        raise ValueError("Reviewer name and expert diagnosis are required.")
    if len(diagnosis) > 300 or len(comments) > 2000 or len(reviewer_identity) > 120:
        raise ValueError("Review information exceeds the allowed length.")
    with _connection(db_path) as conn:
        cursor = conn.execute(
            """UPDATE result_validations
               SET expert_review_status = ?, expert_diagnosis = ?, expert_comments = ?,
                   expert_reviewed_at = ?, reviewer_identity = ?
               WHERE id = ?""",
            (
                status,
                diagnosis,
                comments,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                reviewer_identity,
                record_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LookupError("Validation record was not found.")


def _row_with_status(row: sqlite3.Row) -> Dict[str, Any]:
    record = dict(row)
    verified = record.get("verified_label")
    if not verified:
        record["match_status"] = "awaiting"
        record["match_label"] = "Awaiting verification"
    elif verified not in TARGET_CLASSES:
        record["match_status"] = "unsupported"
        record["match_label"] = "Unsupported reference label"
    elif verified == record["predicted_class"]:
        record["match_status"] = "match"
        record["match_label"] = "Match"
    else:
        record["match_status"] = "mismatch"
        record["match_label"] = "Does not match"
    if record.get("expert_review_status") not in REVIEW_STATUSES:
        record["expert_review_status"] = REVIEW_PENDING
    return record


def get_validation_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    init_validation_db(db_path)
    class_placeholders = ",".join("?" for _ in TARGET_CLASSES)
    with _connection(db_path) as conn:
        totals = conn.execute(
            f"""SELECT COUNT(*) AS total,
                      SUM(CASE WHEN verified_label IN ({class_placeholders}) THEN 1 ELSE 0 END) AS eligible,
                      SUM(CASE WHEN verified_label IN ({class_placeholders}) AND verified_label = predicted_class THEN 1 ELSE 0 END) AS correct,
                      SUM(CASE WHEN expert_review_status != ? AND expert_reviewed_at IS NOT NULL
                                   AND reviewer_identity IS NOT NULL AND reviewer_identity != '' THEN 1 ELSE 0 END) AS reviewed
               FROM result_validations""",
            (*TARGET_CLASSES, *TARGET_CLASSES, REVIEW_PENDING),
        ).fetchone()
    eligible = int(totals["eligible"] or 0)
    correct = int(totals["correct"] or 0)
    return {
        "total_records": int(totals["total"] or 0),
        "images_tested": eligible,
        "correct_results": correct,
        "incorrect_results": eligible - correct,
        "test_accuracy_pct": (correct / eligible * 100.0) if eligible else None,
        "expert_reviewed": int(totals["reviewed"] or 0),
    }


def get_validation_crops(db_path: Optional[str] = None) -> List[str]:
    init_validation_db(db_path)
    with _connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT crop FROM result_validations ORDER BY crop COLLATE NOCASE"
        ).fetchall()
    return [row["crop"] for row in rows]


def list_validations(
    *, crop: str = "all", match_status: str = "all", page: int = 1,
    per_page: int = 20, db_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    init_validation_db(db_path)
    clauses = []
    params: List[Any] = []
    if crop and crop != "all":
        clauses.append("crop = ?")
        params.append(crop)
    if match_status == "awaiting":
        clauses.append("(verified_label IS NULL OR verified_label = '')")
    elif match_status == "match":
        clauses.append("verified_label = predicted_class")
    elif match_status == "mismatch":
        clauses.append("verified_label IS NOT NULL AND verified_label != '' AND verified_label != predicted_class AND verified_label IN (%s)" % ",".join("?" for _ in TARGET_CLASSES))
        params.extend(TARGET_CLASSES)
    elif match_status == "unsupported":
        clauses.append("verified_label IS NOT NULL AND verified_label != '' AND verified_label NOT IN (%s)" % ",".join("?" for _ in TARGET_CLASSES))
        params.extend(TARGET_CLASSES)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    offset = (max(page, 1) - 1) * per_page
    with _connection(db_path) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM result_validations" + where, params).fetchone()[0])
        rows = conn.execute(
            "SELECT * FROM result_validations" + where + " ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, per_page, offset],
        ).fetchall()
    return [_row_with_status(row) for row in rows], total


def get_validation_by_id(record_id: int, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_validation_db(db_path)
    with _connection(db_path) as conn:
        row = conn.execute("SELECT * FROM result_validations WHERE id = ?", (record_id,)).fetchone()
    return _row_with_status(row) if row else None


def get_validation_by_ref(image_ref: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_validation_db(db_path)
    with _connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM result_validations WHERE image_ref = ?", (image_ref,)
        ).fetchone()
    return _row_with_status(row) if row else None


def validation_csv(db_path: Optional[str] = None) -> str:
    rows, _ = list_validations(page=1, per_page=1_000_000, db_path=db_path)
    output = io.StringIO(newline="")
    writer = csv.writer(output)

    def safe_cell(value: Any) -> Any:
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        return value

    writer.writerow([
        "Image Reference", "Crop", "Verified Label", "AI Prediction", "Match Status",
        "Label Source", "Expert Review Status", "Submitted Date",
    ])
    for row in rows:
        writer.writerow([safe_cell(value) for value in [
            row["image_ref"], row["crop"], row["verified_label"] or "",
            row["predicted_display_name"], row["match_label"], row["label_source"] or "",
            row["expert_review_status"], row["created_at"],
        ]])
    return output.getvalue()

"""
CropCare AI — SQLite prediction history database.
"""

import os
import sys
import sqlite3
import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            filename    TEXT    NOT NULL,
            predicted_class  TEXT NOT NULL,
            display_name     TEXT NOT NULL,
            confidence  REAL    NOT NULL,
            top_predictions  TEXT,       -- JSON string
            arch        TEXT,
            img_size    INTEGER,
            symptoms    TEXT,
            management  TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(filename: str, result: Dict) -> int:
    """
    Save a prediction result to the database.
    Returns the row id of the new record.
    """
    import json
    conn = get_connection()
    cur = conn.execute(
        """INSERT INTO predictions
           (timestamp, filename, predicted_class, display_name,
            confidence, top_predictions, arch, img_size, symptoms, management)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            result.get("predicted_class", ""),
            result.get("display_name", ""),
            result.get("confidence", 0.0),
            json.dumps(result.get("top_predictions", [])),
            result.get("arch", ""),
            result.get("img_size", 0),
            result.get("disease_info", {}).get("symptoms", ""),
            result.get("disease_info", {}).get("management", ""),
        )
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_all_predictions(limit: int = 50) -> List[Dict]:
    """Return the most recent predictions, newest first."""
    import json
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        try:
            d["top_predictions"] = json.loads(d.get("top_predictions") or "[]")
        except Exception:
            d["top_predictions"] = []
        results.append(d)
    return results


def get_prediction_by_id(pred_id: int) -> Optional[Dict]:
    """Return a single prediction by ID."""
    import json
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM predictions WHERE id = ?", (pred_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    try:
        d["top_predictions"] = json.loads(d.get("top_predictions") or "[]")
    except Exception:
        d["top_predictions"] = []
    return d


def get_stats() -> Dict:
    """Return aggregate statistics for the about/dashboard page."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    class_counts = conn.execute(
        "SELECT predicted_class, display_name, COUNT(*) as cnt "
        "FROM predictions GROUP BY predicted_class ORDER BY cnt DESC"
    ).fetchall()
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "avg_confidence": round(avg_conf or 0, 4),
        "class_counts": [dict(r) for r in class_counts],
    }

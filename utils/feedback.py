"""
utils/feedback.py
Saves feedback and usage logs to CSV files.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

FEEDBACK_FILE = Path("data/feedback_log.csv")
USAGE_FILE    = Path("data/usage_log.csv")
SEARCH_HISTORY_FILE = Path("data/search_history.csv")


def _ensure_file(path: Path, headers: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def save_feedback(
    question: str,
    answer: str,
    rating: str,
    comment: Optional[str] = None,
):
    _ensure_file(FEEDBACK_FILE, ["timestamp", "rating", "question", "answer_preview", "comment"])
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rating,
            question[:200],
            answer[:200],
            comment or "",
        ])


def save_usage(
    question: str,
    answered: bool,
    sources: list,
    language: str = "English",
):
    _ensure_file(USAGE_FILE, ["timestamp", "question", "answered", "source_files", "language"])
    source_names = "; ".join(set(s["file"] for s in sources)) if sources else "none"
    with open(USAGE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question[:200],
            "yes" if answered else "no",
            source_names,
            language,
        ])


def load_feedback() -> list:
    if not FEEDBACK_FILE.exists():
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_usage() -> list:
    if not USAGE_FILE.exists():
        return []
    with open(USAGE_FILE, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_search_history(question: str, answer_preview: str):
    _ensure_file(SEARCH_HISTORY_FILE, ["timestamp", "question", "answer_preview"])
    with open(SEARCH_HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question[:200],
            answer_preview[:150],
        ])


def load_search_history(limit: int = 50) -> list:
    if not SEARCH_HISTORY_FILE.exists():
        return []
    with open(SEARCH_HISTORY_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return list(reversed(rows))[:limit]


def clear_search_history():
    if SEARCH_HISTORY_FILE.exists():
        SEARCH_HISTORY_FILE.unlink()
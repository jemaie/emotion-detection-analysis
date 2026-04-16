"""
Multi-rater storage module.

Ratings are stored per rater under:
    ratings/{rater_name}/caller_concat/{conv_id}.json
    ratings/{rater_name}/caller_segments/{conv_id}/{segment_stem}.json

Each JSON:
    { "true_emotion": "...", "phase": "...", "notes": "...", "timestamp": "..." }
"""

from pathlib import Path
import json
import logging
import shutil
import filelock
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

RATINGS_DIR = Path("ratings")
DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "conversations.json"

CONCAT_DIR = DATA_DIR / "caller_concat_24kHz"
SEGMENTS_DIR = DATA_DIR / "caller_segments_24kHz"

ADMIN_NAME = "admin-jm"


def _rater_dir(rater_name: str) -> Path:
    """Sanitise rater name into a safe directory name."""
    safe = rater_name.strip().replace(" ", "_").lower()
    safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-"))
    return RATINGS_DIR / safe


# ── manifest ────────────────────────────────────────────────────────

def load_manifest() -> list[dict]:
    """Load the conversations manifest."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ── reading ─────────────────────────────────────────────────────────

def _rating_path(rater_name: str, conv_id: str, segment: str | None = None) -> Path:
    base = _rater_dir(rater_name)
    if segment:
        stem = Path(segment).stem
        return base / "caller_segments" / conv_id / f"{stem}.json"
    return base / "caller_concat" / f"{conv_id}.json"


def read_rating(rater_name: str, conv_id: str, segment: str | None = None) -> dict | None:
    """Read a single rating. Returns None if not yet rated."""
    p = _rating_path(rater_name, conv_id, segment)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {p}: {e}")
    return None


def write_rating(rater_name: str, conv_id: str, rating: dict, segment: str | None = None):
    """Write (or overwrite) a rating."""
    p = _rating_path(rater_name, conv_id, segment)
    p.parent.mkdir(parents=True, exist_ok=True)
    rating["timestamp"] = datetime.now(timezone.utc).isoformat()
    lock = filelock.FileLock(str(p) + ".lock", timeout=10)
    with lock:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rating, f, indent=2, ensure_ascii=False)


def delete_rater_ratings(rater_name: str) -> bool:
    """Delete all ratings for a given rater. Returns True if deleted."""
    d = _rater_dir(rater_name)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


# ── aggregation helpers ─────────────────────────────────────────────

def get_all_rater_names() -> list[str]:
    """List all rater directory names that exist."""
    if not RATINGS_DIR.exists():
        return []
    return sorted([d.name for d in RATINGS_DIR.iterdir() if d.is_dir()])


def get_rater_progress(rater_name: str, manifest: list[dict]) -> dict:
    """Returns { 'concat_rated': int, 'concat_total': int, 'segments_rated': int, 'segments_total': int }."""
    concat_total = len(manifest)
    segments_total = sum(len(c["segments"]) for c in manifest)

    concat_rated = 0
    segments_rated = 0

    for conv in manifest:
        cid = conv["conv_id"]
        if read_rating(rater_name, cid) is not None:
            concat_rated += 1
        for seg in conv["segments"]:
            if read_rating(rater_name, cid, seg) is not None:
                segments_rated += 1

    return {
        "concat_rated": concat_rated,
        "concat_total": concat_total,
        "segments_rated": segments_rated,
        "segments_total": segments_total,
    }


def get_all_ratings_for_conv(conv_id: str, segments: list[str]) -> dict:
    """
    Returns all ratings across all raters for a given conversation.
    {
        "concat": { rater_name: rating_dict, ... },
        "segments": { segment_name: { rater_name: rating_dict, ... }, ... }
    }
    """
    result = {"concat": {}, "segments": {}}
    for rater in get_all_rater_names():
        r = read_rating(rater, conv_id)
        if r:
            result["concat"][rater] = r
        for seg in segments:
            sr = read_rating(rater, conv_id, seg)
            if sr:
                if seg not in result["segments"]:
                    result["segments"][seg] = {}
                result["segments"][seg][rater] = sr
    return result

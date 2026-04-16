"""
Export all ratings to CSV for evaluation / inter-rater analysis.

Outputs two CSV files:
  - ratings_concat.csv   (one row per rater × conversation)
  - ratings_segments.csv  (one row per rater × segment)

Run:
    python export_ratings.py
"""

import csv
import json
from pathlib import Path

RATINGS_DIR = Path("ratings")
DATA_DIR = Path("data")
MANIFEST_PATH = DATA_DIR / "conversations.json"
OUTPUT_DIR = Path("exports")


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    raters = sorted([d.name for d in RATINGS_DIR.iterdir() if d.is_dir()]) if RATINGS_DIR.exists() else []
    if not raters:
        print("No ratings found.")
        return

    print(f"Found raters: {', '.join(raters)}")

    # ── Concat CSV ─────────────────────────────────────────────────
    concat_rows = []
    for conv in manifest:
        cid = conv["conv_id"]
        for rater in raters:
            p = RATINGS_DIR / rater / "caller_concat" / f"{cid}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    r = json.load(f)
                concat_rows.append({
                    "conv_id": cid,
                    "rater": rater,
                    "emotion": r.get("true_emotion", ""),
                    "notes": r.get("notes", ""),
                    "timestamp": r.get("timestamp", ""),
                })

    concat_path = OUTPUT_DIR / "ratings_concat.csv"
    with open(concat_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["conv_id", "rater", "emotion", "notes", "timestamp"])
        w.writeheader()
        w.writerows(concat_rows)
    print(f"Wrote {len(concat_rows)} concat ratings to {concat_path}")

    # ── Segments CSV ───────────────────────────────────────────────
    seg_rows = []
    for conv in manifest:
        cid = conv["conv_id"]
        for seg in conv.get("segments", []):
            stem = Path(seg).stem
            for rater in raters:
                p = RATINGS_DIR / rater / "caller_segments" / cid / f"{stem}.json"
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        r = json.load(f)
                    seg_rows.append({
                        "conv_id": cid,
                        "segment": seg,
                        "rater": rater,
                        "emotion": r.get("true_emotion", ""),
                        "notes": r.get("notes", ""),
                        "timestamp": r.get("timestamp", ""),
                    })

    seg_path = OUTPUT_DIR / "ratings_segments.csv"
    with open(seg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["conv_id", "segment", "rater", "emotion", "notes", "timestamp"])
        w.writeheader()
        w.writerows(seg_rows)
    print(f"Wrote {len(seg_rows)} segment ratings to {seg_path}")

    # ── Agreement summary ──────────────────────────────────────────
    print("\n=== Inter-Rater Summary (Conversations) ===")
    for conv in manifest:
        cid = conv["conv_id"]
        ratings = {}
        for rater in raters:
            p = RATINGS_DIR / rater / "caller_concat" / f"{cid}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    r = json.load(f)
                ratings[rater] = r.get("true_emotion", "?")

        if len(ratings) > 1:
            emotions = list(ratings.values())
            agree = len(set(emotions)) == 1
            status = "✅ AGREE" if agree else "❌ DISAGREE"
            rating_str = ", ".join(f"{k}={v}" for k, v in ratings.items())
            print(f"  {cid}: {status}  [{rating_str}]")
        elif len(ratings) == 1:
            rater, emotion = list(ratings.items())[0]
            print(f"  {cid}: only {rater} → {emotion}")


if __name__ == "__main__":
    main()

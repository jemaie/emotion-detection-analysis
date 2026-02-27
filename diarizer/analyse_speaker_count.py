#!/usr/bin/env python3
"""
Analyze diarization outputs and summarize how many speakers were detected.

Scans: out/diarized/*.json
Reads: JSON with key "segments", each segment has key "speaker"
Outputs:
  - console summary (counts for 0/1/2/3/4/5+ speakers)
  - out/index/speaker_counts.csv
  - out/index/speaker_counts_summary.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


@dataclass(frozen=True)
class FileStats:
    file: str
    num_speakers: int
    speakers: Tuple[str, ...]
    num_segments: int


def _safe_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read JSON: {path} ({e})")
        return None


def _extract_speakers(d: Dict[str, Any]) -> Tuple[Set[str], int]:
    segments = d.get("segments", [])
    if not isinstance(segments, list):
        return set(), 0

    speakers: Set[str] = set()
    for seg in segments:
        if isinstance(seg, dict):
            spk = seg.get("speaker")
            if isinstance(spk, str) and spk.strip():
                speakers.add(spk.strip())

    return speakers, len(segments)


def bucket_label(n: int) -> str:
    """Bucket for summary: 0,1,2,3,4, or 5+."""
    if n >= 5:
        return "5+"
    return str(n)


def analyze_directory(out_dir: Path) -> None:
    diarized_dir = out_dir / "diarized"
    index_dir = out_dir / "index"
    
    csv_path = index_dir / "speaker_counts.csv"
    summary_json_path = index_dir / "speaker_counts_summary.json"

    if not diarized_dir.exists():
        print(f"[WARN] Missing folder: {diarized_dir}")
        return

    index_dir.mkdir(parents=True, exist_ok=True)

    diar_files = sorted(diarized_dir.glob("*.json"))
    if not diar_files:
        print(f"[INFO] No JSON files found in {diarized_dir}")
        return

    rows: List[FileStats] = []
    bucket_counts = Counter()
    failed = 0

    for p in diar_files:
        data = _safe_load_json(p)
        if data is None:
            failed += 1
            continue

        speakers, num_segments = _extract_speakers(data)
        speakers_sorted = tuple(sorted(speakers))
        n = len(speakers_sorted)

        rows.append(
            FileStats(
                file=p.name,
                num_speakers=n,
                speakers=speakers_sorted,
                num_segments=num_segments,
            )
        )
        bucket_counts[bucket_label(n)] += 1

    # Write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "num_speakers", "num_segments", "speakers"])
        for r in sorted(rows, key=lambda x: (-x.num_speakers, x.file)):
            w.writerow([r.file, r.num_speakers, r.num_segments, ";".join(r.speakers)])

    # Write JSON summary
    summary = {
        "diarized_dir": str(diarized_dir),
        "num_files_total": len(diar_files),
        "num_files_parsed": len(rows),
        "num_files_failed": failed,
        "bucket_counts": dict(bucket_counts),
        "examples": {
            "0": [r.file for r in rows if r.num_speakers == 0][:10],
            "1": [r.file for r in rows if r.num_speakers == 1][:10],
            "2": [r.file for r in rows if r.num_speakers == 2][:10],
            "3": [r.file for r in rows if r.num_speakers == 3][:10],
            "4": [r.file for r in rows if r.num_speakers == 4][:10],
            "5+": [r.file for r in rows if r.num_speakers >= 5][:10],
        },
        "top_speaker_sets": [
            {"speakers": list(s), "count": c}
            for s, c in Counter(r.speakers for r in rows).most_common(10)
        ],
    }
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print console summary
    print(f"\n=== Speaker count summary for {out_dir.name} ===")
    print(f"Diarized files total: {len(diar_files)}")
    print(f"Parsed: {len(rows)} | Failed: {failed}")
    for k in ["0", "1", "2", "3", "4", "5+"]:
        print(f"  {k} speaker(s): {bucket_counts.get(k, 0)}")

    print(f"\nWrote CSV:     {csv_path}")
    print(f"Wrote summary: {summary_json_path}")

    # Show a few “3+ speakers” examples for quick inspection
    more = [r for r in rows if r.num_speakers >= 3]
    if more:
        print("\nExamples with 3+ speakers (first 10):")
        for r in more[:10]:
            print(f"  {r.file}: {r.num_speakers} speakers -> {r.speakers}")


def main() -> None:
    base_dir = Path("OUT_COMPARISON")
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"[ERROR] Directory '{base_dir}' not found.")
        return
        
    for out_dir in sorted(base_dir.iterdir()):
        if out_dir.is_dir() and out_dir.name.startswith("out"):
            analyze_directory(out_dir)


if __name__ == "__main__":
    main()

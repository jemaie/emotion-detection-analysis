"""
Copies the relevant audio data and conversation manifest from the labeler folder
into the deploy folder so it can be pushed to a Git repository.

Run from the repository root:
    python deploy/copy_data.py
"""

import shutil
import json
from pathlib import Path

LABELER = Path("labeler")
DEPLOY  = Path("deploy")

CONCAT_SRC   = LABELER / "data" / "caller_concat_24kHz"
SEGMENTS_SRC = LABELER / "data" / "caller_segments_24kHz"
EVAL_CONCAT  = LABELER / "output" / "caller_concat"
EVAL_SEGS    = LABELER / "output" / "caller_segments"

CONCAT_DST   = DEPLOY / "data" / "caller_concat_24kHz"
SEGMENTS_DST = DEPLOY / "data" / "caller_segments_24kHz"

def main():
    # Discover all conversation IDs that have evaluation JSONs
    conv_ids = [f.stem for f in sorted(EVAL_CONCAT.glob("*.json"))]
    print(f"Found {len(conv_ids)} evaluated conversations.")

    # --- Copy concat audio ---
    CONCAT_DST.mkdir(parents=True, exist_ok=True)
    for cid in conv_ids:
        src = CONCAT_SRC / f"{cid}.wav"
        dst = CONCAT_DST / f"{cid}.wav"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  [concat] {cid}.wav")
        else:
            print(f"  [concat] MISSING {cid}.wav")

    # --- Copy segment audio ---
    total_segs = 0
    for cid in conv_ids:
        seg_src = SEGMENTS_SRC / cid
        seg_dst = SEGMENTS_DST / cid
        if seg_src.exists():
            seg_dst.mkdir(parents=True, exist_ok=True)
            for wav in sorted(seg_src.glob("*.wav")):
                shutil.copy2(wav, seg_dst / wav.name)
                total_segs += 1
        else:
            print(f"  [segments] MISSING dir for {cid}")

    print(f"  [segments] Copied {total_segs} segment files across {len(conv_ids)} conversations.")

    # --- Build conversations manifest ---
    manifest = []
    for cid in conv_ids:
        seg_dir = SEGMENTS_DST / cid
        segments = sorted([f.name for f in seg_dir.glob("*.wav")]) if seg_dir.exists() else []
        manifest.append({
            "conv_id": cid,
            "concat_file": f"{cid}.wav",
            "segments": segments,
        })

    manifest_path = DEPLOY / "data" / "conversations.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written to {manifest_path} ({len(manifest)} conversations)")

    # Print total size
    total_bytes = sum(f.stat().st_size for f in CONCAT_DST.rglob("*.wav"))
    total_bytes += sum(f.stat().st_size for f in SEGMENTS_DST.rglob("*.wav"))
    print(f"Total audio data: {total_bytes / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from run_batch_scripts.audio_io import normalize_to_wav16k_mono
from run_batch_scripts.diarize_pyannote import diarize_transcribe as diarize_transcribe_pyannote
from run_batch_scripts.role_assign import assign_roles
from run_batch_scripts.segment_postprocess import postprocess_caller_segments
from run_batch_scripts.extract_audio import extract_segments_ffmpeg, concat_wavs_ffmpeg

RAW_DIR = Path("../aufnahmen")
NORM_DIR = Path("data/normalized")
REFS_DIR = Path("data/refs")
OUT_DIR = Path("output")

# Segmenting params (tuned based on prior experiments)
TRIM_MS = 0
MERGE_GAP_MS = 300
MIN_SEG_DUR_S = 0.8

def load_agent_ref() -> List[Path]:
    """Load exactly one reference file for Pyannote padding."""
    refs = sorted([p for p in REFS_DIR.glob("*.wav")])
    return refs[:1]

def list_input_files() -> List[Path]:
    """Gather all raw audio files."""
    files = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            files.append(p)
    return sorted(files)

def create_dirs(base: Path) -> dict:
    """Create directory structure for flattened output."""
    dirs = {
        "diar": base / "diarized",
        "caller_segs": base / "caller_segments",
        "caller_concat": base / "caller_concat",
        "index": base / "index"
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def main() -> None:
    agent_refs = load_agent_ref()
    
    audio_files = list_input_files()
    if not audio_files:
        raise RuntimeError(f"No input files found under {RAW_DIR}")

    if not agent_refs:
        print(f"Warning: No agent reference WAVs found in {REFS_DIR}.")
        return

    print(f"=== Starting pipeline with Pyannote (1 reference: {agent_refs[0].name}) ===")
    print(f"Found {len(audio_files)} files to process.")

    dirs = create_dirs(OUT_DIR)
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    
    for src in tqdm(audio_files, desc="Extracting Caller Audio"):
        call_id = src.stem

        norm_path = NORM_DIR / f"{call_id}.wav"

        # 1) Normalize
        if not norm_path.exists():
            normalize_to_wav16k_mono(src, norm_path)

        diar_path = dirs["diar"] / f"{call_id}.json"
        summary_path = dirs["index"] / f"{call_id}.summary.json"

        # 2) Diarize+transcribe (cache)
        if diar_path.exists():
            try:
                diarized = json.loads(diar_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"\nError reading cached diarization for {call_id}: {e}")
                continue
        else:
            try:
                diarized = diarize_transcribe_pyannote(norm_path, agent_ref_paths=agent_refs)
                diar_path.write_text(json.dumps(diarized, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"\nError running pyannote on {call_id}: {e}")
                continue

        segments = diarized.get("segments", []) or []

        # 3) Role assignment
        speaker_to_role, speaker_durations, flags = assign_roles(diarized)

        # 4) Post-process caller segments (trim/merge/drop)
        caller_segments, stats = postprocess_caller_segments(
            diarized_segments=segments,
            speaker_to_role=speaker_to_role,
            trim_ms=TRIM_MS,
            merge_gap_ms=MERGE_GAP_MS,
            min_seg_dur_s=MIN_SEG_DUR_S,
        )

        # 5) Extract per-segment + concat
        seg_dir = dirs["caller_segs"] / call_id
        # Note: extract_segments_ffmpeg might be skipping if files exist, but we pass it anyway.
        seg_paths = extract_segments_ffmpeg(norm_path, caller_segments, seg_dir)

        concat_path = dirs["caller_concat"] / f"{call_id}.wav"
        if seg_paths:
            concat_wavs_ffmpeg(seg_paths, concat_path)

        # Summaries / audit info
        summary = {
            "call_id": call_id,
            "provider": "pyannote",
            "source_file": str(src),
            "normalized_file": str(norm_path),
            "diarized_file": str(diar_path),
            "speaker_to_role": speaker_to_role,
            "speaker_durations_sec": speaker_durations,
            "flags": flags,
            "num_segments_total": len(segments),
            "num_segments_caller_raw": stats["num_raw_caller"],
            "num_segments_caller_dropped": stats["num_dropped"],
            "num_segments_caller_dropped_trim_dur": stats.get("num_dropped_trim_dur", 0),
            "num_segments_caller_dropped_overlap": stats.get("num_dropped_overlap", 0),
            "num_segments_caller_merged": stats["num_merged_into"],
            "num_segments_caller_final": len(caller_segments),
            "caller_segments_dir": str(seg_dir),
            "caller_concat_file": str(concat_path) if seg_paths else None,
            "segment_params": {
                "trim_ms": TRIM_MS,
                "merge_gap_ms": MERGE_GAP_MS,
                "min_seg_dur_s": MIN_SEG_DUR_S,
            },
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

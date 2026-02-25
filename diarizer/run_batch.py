import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from audio_io import normalize_to_wav16k_mono
from diarize_openai import diarize_transcribe
from role_assign import assign_roles
from segment_postprocess import postprocess_caller_segments
from extract_audio import extract_segments_ffmpeg, concat_wavs_ffmpeg

RAW_DIR = Path("../aufnahmen20")
NORM_DIR = Path("data/normalized")
REFS_DIR = Path("data/refs")

OUT_DIAR = Path("out/diarized")
OUT_CALLER_SEGS = Path("out/caller_segments")
OUT_CALLER_CONCAT = Path("out/caller_concat")
OUT_INDEX = Path("out/index")

# Segmenting params (tune later if needed)
TRIM_MS = 250
MERGE_GAP_MS = 300
MIN_SEG_DUR_S = 0.7

def load_agent_refs() -> List[Path]:
    # Use up to 4 refs; order doesn't matter much
    refs = sorted([p for p in REFS_DIR.glob("*.wav")])
    return refs[:4]

def list_input_files() -> List[Path]:
    files = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            files.append(p)
    return sorted(files)

def main() -> None:
    OUT_DIAR.mkdir(parents=True, exist_ok=True)
    OUT_CALLER_SEGS.mkdir(parents=True, exist_ok=True)
    OUT_CALLER_CONCAT.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.mkdir(parents=True, exist_ok=True)

    agent_refs = load_agent_refs()
    # if not agent_refs:
    #     raise RuntimeError("No agent reference WAVs found in data/refs. Add agent_01.wav ...")

    audio_files = list_input_files()
    if not audio_files:
        raise RuntimeError("No input files found under data/raw.")

    for src in tqdm(audio_files, desc="Batch"):
        call_id = src.stem

        norm_path = NORM_DIR / f"{call_id}.wav"
        diar_path = OUT_DIAR / f"{call_id}.json"
        summary_path = OUT_INDEX / f"{call_id}.summary.json"

        # 1) Normalize
        if not norm_path.exists():
            normalize_to_wav16k_mono(src, norm_path)

        # 2) Diarize+transcribe (cache)
        if diar_path.exists():
            diarized = json.loads(diar_path.read_text(encoding="utf-8"))
        else:
            diarized = diarize_transcribe(norm_path, agent_ref_paths=agent_refs)
            diar_path.write_text(json.dumps(diarized, ensure_ascii=False, indent=2), encoding="utf-8")

        segments = diarized.get("segments", []) or []

        # 3) Role assignment
        speaker_to_role, speaker_durations, flags = assign_roles(diarized)

        # 4) Post-process caller segments (trim/merge/drop)
        caller_segments = postprocess_caller_segments(
            diarized_segments=segments,
            speaker_to_role=speaker_to_role,
            trim_ms=TRIM_MS,
            merge_gap_ms=MERGE_GAP_MS,
            min_seg_dur_s=MIN_SEG_DUR_S,
        )

        # 5) Extract per-segment + concat
        seg_dir = OUT_CALLER_SEGS / call_id
        seg_paths = extract_segments_ffmpeg(norm_path, caller_segments, seg_dir)

        concat_path = OUT_CALLER_CONCAT / f"{call_id}.wav"
        if seg_paths:
            concat_wavs_ffmpeg(seg_paths, concat_path)

        # Summaries / audit info
        summary = {
            "call_id": call_id,
            "source_file": str(src),
            "normalized_file": str(norm_path),
            "diarized_file": str(diar_path),
            "speaker_to_role": speaker_to_role,
            "speaker_durations_sec": speaker_durations,
            "flags": flags,
            "num_segments_total": len(segments),
            "num_segments_caller_raw": sum(1 for s in segments if speaker_to_role.get(s.get("speaker", "unknown")) == "caller"),
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

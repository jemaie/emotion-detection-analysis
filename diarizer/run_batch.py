import json
import concurrent.futures
from pathlib import Path
from typing import List

from tqdm import tqdm

from run_batch_scripts.audio_io import normalize_to_wav16k_mono
from run_batch_scripts.diarize_openai import diarize_transcribe as diarize_transcribe_openai
#from run_batch_scripts.diarize_google import diarize_transcribe as diarize_transcribe_google
from run_batch_scripts.diarize_pyannote import diarize_transcribe as diarize_transcribe_pyannote
from run_batch_scripts.role_assign import assign_roles
from run_batch_scripts.segment_postprocess import postprocess_caller_segments
from run_batch_scripts.extract_audio import extract_segments_ffmpeg, concat_wavs_ffmpeg

RAW_DIR = Path("../aufnahmen25")
NORM_DIR = Path("data/normalized")
REFS_DIR = Path("data/refs")

# Segmenting params (tune later if needed)
TRIM_MS = 200
MERGE_GAP_MS = 300
MIN_SEG_DUR_S = 0.8

def load_agent_refs() -> List[Path]:
    # Use up to 4 refs; ensure sorted order (e.g., ref_01, ref_02)
    refs = sorted([p for p in REFS_DIR.glob("*.wav")])
    return refs[:4]

def list_input_files() -> List[Path]:
    files = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            files.append(p)
    return sorted(files)

def main() -> None:
    all_agent_refs = load_agent_refs()
    
    audio_files = list_input_files()
    if not audio_files:
        raise RuntimeError("No input files found under data/raw.")

    if not all_agent_refs:
        print("Warning: No agent reference WAVs found in data/refs.")
        return

    for num_refs in range(0, len(all_agent_refs) + 1):
        agent_refs = all_agent_refs[:num_refs]
        print(f"\n=== Processing with {num_refs} reference(s) ===")

        # Output bases
        openai_base = Path(f"out_openai/{num_refs}_refs")
        google_base = Path(f"out_google")
        pyannote_base = Path(f"out_pyannote/{num_refs}_refs")

        def create_dirs(base: Path) -> dict:
            dirs = {
                "diar": base / "diarized",
                "caller_segs": base / "caller_segments",
                "caller_concat": base / "caller_concat",
                "index": base / "index"
            }
            for d in dirs.values():
                d.mkdir(parents=True, exist_ok=True)
            return dirs

        dirs_openai = create_dirs(openai_base)
        dirs_google = create_dirs(google_base)
        dirs_pyannote = create_dirs(pyannote_base)

        for src in tqdm(audio_files, desc=f"Batch ({num_refs} refs)"):
            call_id = src.stem

            norm_path = NORM_DIR / f"{call_id}.wav"

            # 1) Normalize
            if not norm_path.exists():
                normalize_to_wav16k_mono(src, norm_path)

            def process_diarizer(provider: str, dirs: dict, diarize_fn: callable) -> None:
                if not diarize_fn:
                    print(f"Skipping {provider} (not available).")
                    return

                diar_path = dirs["diar"] / f"{call_id}.json"
                summary_path = dirs["index"] / f"{call_id}.summary.json"

                # 2) Diarize+transcribe (cache)
                if diar_path.exists():
                    diarized = json.loads(diar_path.read_text(encoding="utf-8"))
                else:
                    try:
                        if provider in ["openai", "pyannote"]:
                            diarized = diarize_fn(norm_path, agent_ref_paths=agent_refs)
                        else:
                            diarized = diarize_fn(norm_path)  # Google doesn't use refs
                        diar_path.write_text(json.dumps(diarized, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception as e:
                        print(f"Error running {provider} on {call_id}: {e}")
                        return

                segments = diarized.get("segments", []) or []

                # 3) Role assignment
                speaker_to_role, speaker_durations, flags = assign_roles(diarized)

                # 4) Post-process caller segments (trim/merge/drop)
                current_trim_ms = 0 if provider == "openai" else TRIM_MS
                
                caller_segments, stats = postprocess_caller_segments(
                    diarized_segments=segments,
                    speaker_to_role=speaker_to_role,
                    trim_ms=current_trim_ms,
                    merge_gap_ms=MERGE_GAP_MS,
                    min_seg_dur_s=MIN_SEG_DUR_S,
                )

                # 5) Extract per-segment + concat
                seg_dir = dirs["caller_segs"] / call_id
                seg_paths = extract_segments_ffmpeg(norm_path, caller_segments, seg_dir)

                concat_path = dirs["caller_concat"] / f"{call_id}.wav"
                if seg_paths:
                    concat_wavs_ffmpeg(seg_paths, concat_path)

                # Summaries / audit info
                summary = {
                    "call_id": call_id,
                    "provider": provider,
                    "source_file": str(src),
                    "normalized_file": str(norm_path),
                    "diarized_file": str(diar_path),
                    "speaker_to_role": speaker_to_role,
                    "speaker_durations_sec": speaker_durations,
                    "flags": flags,
                    "num_segments_total": len(segments),
                    "num_segments_caller_raw": stats["num_raw_caller"],
                    "num_segments_caller_dropped": stats["num_dropped"],
                    "num_segments_caller_merged": stats["num_merged_into"],
                    "num_segments_caller_final": len(caller_segments),
                    "caller_segments_dir": str(seg_dir),
                    "caller_concat_file": str(concat_path) if seg_paths else None,
                    "segment_params": {
                        "trim_ms": current_trim_ms,
                        "merge_gap_ms": MERGE_GAP_MS,
                        "min_seg_dur_s": MIN_SEG_DUR_S,
                    },
                }
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            # Run diarizers in parallel for this audio file
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(process_diarizer, "openai", dirs_openai, diarize_transcribe_openai),
                    # executor.submit(process_diarizer, "google", dirs_google, diarize_transcribe_google),
                    executor.submit(process_diarizer, "pyannote", dirs_pyannote, diarize_transcribe_pyannote)
                ]
                
                # Wait for all providers to finish processing this file before moving to the next
                for future in concurrent.futures.as_completed(futures):
                    # We can optionally handle exceptions here if they weren't caught inside process_diarizer 
                    _ = future.result()

if __name__ == "__main__":
    main()

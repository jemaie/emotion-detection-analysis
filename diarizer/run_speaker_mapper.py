import json
from pathlib import Path
from tqdm import tqdm

from run_batch_scripts.diarize_pyannote import diarize_transcribe
from run_batch_scripts.role_assign import assign_roles
from run_batch_scripts.segment_postprocess import postprocess_caller_segments
from run_batch_scripts.extract_audio import extract_segments_ffmpeg, concat_wavs_ffmpeg
from speaker_mapping import map_speakers_to_roles

TRIM_MS = 200
MERGE_GAP_MS = 300
MIN_SEG_DUR_S = 0.8

def main():
    norm_dir = Path("data/normalized")
    ref_path = Path("data/refs/agent_ref_01.wav")
    providers = ["pyannote", "openai"]

    for provider in providers:
        base_dir = Path(f"out_{provider}/0_refs")
        if not base_dir.exists():
            print(f"Directory {base_dir} does not exist.")
            continue

        diar_dir = base_dir / "diarized"
        out_base = Path("out_mapped") / provider
        dirs = {
            "diar": out_base / "diarized",
            "caller_segs": out_base / "caller_segments",
            "caller_concat": out_base / "caller_concat",
            "index": out_base / "index"
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        json_files = list(diar_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found to map for {provider}.")
            continue

        for json_path in tqdm(json_files, desc=f"Mapping Speakers ({provider})"):
            call_id = json_path.stem
            wav_path = norm_dir / f"{call_id}.wav"
            
            if not wav_path.exists():
                print(f"Skipping {call_id}: wav missing")
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                diarized_json = json.load(f)

            # 1. Map offline
            mapped_json = map_speakers_to_roles(diarized_json, wav_path, ref_path)

            mapped_path = dirs["diar"] / f"{call_id}.json"
            mapped_path.write_text(json.dumps(mapped_json, ensure_ascii=False, indent=2), encoding="utf-8")

            segments = mapped_json.get("segments", [])

            # 2. Role assignment
            speaker_to_role, speaker_durations, flags = assign_roles(mapped_json)

            # Determine specific trim per provider
            current_trim_ms = 0 if provider == "openai" else TRIM_MS

            # 3. Post-process 
            caller_segments, stats = postprocess_caller_segments(
                diarized_segments=segments,
                speaker_to_role=speaker_to_role,
                trim_ms=current_trim_ms,
                merge_gap_ms=MERGE_GAP_MS,
                min_seg_dur_s=MIN_SEG_DUR_S,
            )

            # 4. Extract
            seg_dir = dirs["caller_segs"] / call_id
            seg_paths = extract_segments_ffmpeg(wav_path, caller_segments, seg_dir)

            concat_path = dirs["caller_concat"] / f"{call_id}.wav"
            if seg_paths:
                concat_wavs_ffmpeg(seg_paths, concat_path)

            # Summaries / audit info
            summary_path = dirs["index"] / f"{call_id}.summary.json"
            summary = {
                "call_id": call_id,
                "provider": f"offline_mapped_{provider}",
                "source_json": str(json_path),
                "speaker_to_role": speaker_to_role,
                "speaker_durations_sec": speaker_durations,
                "flags": flags,
                "num_segments_caller_raw": stats["num_raw_caller"],
                "num_segments_caller_dropped": stats["num_dropped"],
                "num_segments_caller_dropped_trim_dur": stats.get("num_dropped_trim_dur", 0),
                "num_segments_caller_dropped_overlap": stats.get("num_dropped_overlap", 0),
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

if __name__ == "__main__":
    main()

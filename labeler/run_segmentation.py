import json
from pathlib import Path
from tqdm import tqdm

from scripts_core.segmentation_utils import (
    assign_roles, 
    postprocess_caller_segments, 
    extract_segments_ffmpeg, 
    concat_wavs_ffmpeg
)

INPUT_DIAR_DIR = Path("data/diarized")
INPUT_NORM_16K = Path("data/normalized_16kHz")
INPUT_NORM_24K = Path("data/normalized_24kHz")

OUTPUT_SEGS_16K = Path("data/caller_segments_16kHz")
OUTPUT_CONCAT_16K = Path("data/caller_concat_16kHz")
OUTPUT_SEGS_24K = Path("data/caller_segments_24kHz")
OUTPUT_CONCAT_24K = Path("data/caller_concat_24kHz")

TRIM_MS = 0
MERGE_GAP_MS = 300
MIN_SEG_DUR_S = 0.8


def main():
    if not INPUT_DIAR_DIR.exists():
        print(f"Error: {INPUT_DIAR_DIR} does not exist.")
        return

    json_files = list(INPUT_DIAR_DIR.glob("*.json"))
    print(f"Found {len(json_files)} diarized JSON files.")

    OUTPUT_SEGS_16K.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONCAT_16K.mkdir(parents=True, exist_ok=True)
    OUTPUT_SEGS_24K.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONCAT_24K.mkdir(parents=True, exist_ok=True)

    for json_path in tqdm(json_files, desc="Processing Audio Segments"):
        call_id = json_path.stem

        try:
            diar_data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Failed to load {json_path}: {e}")
            continue

        segments = diar_data.get("segments", [])
        if not segments:
            print(f"Skipping {call_id}: No segments found.")
            continue

        # 1. Role Assignment
        speaker_to_role, _, _ = assign_roles(diar_data)

        # 2. Post-processing
        caller_segments, _ = postprocess_caller_segments(
            diarized_segments=segments,
            speaker_to_role=speaker_to_role,
            trim_ms=TRIM_MS,
            merge_gap_ms=MERGE_GAP_MS,
            min_seg_dur_s=MIN_SEG_DUR_S,
        )

        if not caller_segments:
            print(f"Skipping {call_id}: No valid caller segments after post-processing.")
            continue

        # 3. Process 16kHz audio
        norm_16k_path = INPUT_NORM_16K / f"{call_id}.wav"
        if norm_16k_path.exists():
            seg_dir_16k = OUTPUT_SEGS_16K / call_id
            seg_paths_16k = extract_segments_ffmpeg(norm_16k_path, caller_segments, seg_dir_16k)
            if seg_paths_16k:
                concat_path_16k = OUTPUT_CONCAT_16K / f"{call_id}.wav"
                concat_wavs_ffmpeg(seg_paths_16k, concat_path_16k)

        # 4. Process 24kHz audio
        norm_24k_path = INPUT_NORM_24K / f"{call_id}.wav"
        if norm_24k_path.exists():
            seg_dir_24k = OUTPUT_SEGS_24K / call_id
            seg_paths_24k = extract_segments_ffmpeg(norm_24k_path, caller_segments, seg_dir_24k)
            if seg_paths_24k:
                concat_path_24k = OUTPUT_CONCAT_24K / f"{call_id}.wav"
                concat_wavs_ffmpeg(seg_paths_24k, concat_path_24k)

if __name__ == "__main__":
    main()

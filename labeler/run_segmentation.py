import json
from pathlib import Path
from tqdm import tqdm

from scripts_core.segmentation_utils import (
    assign_roles, 
    postprocess_caller_segments, 
    extract_segments_ffmpeg, 
    concat_wavs_ffmpeg,
    parse_time,
    parse_segment_midpoint
)

INPUT_DIAR_DIR = Path("data/diarized")
INPUT_NORM_16K = Path("data/normalized_16kHz")
INPUT_NORM_24K = Path("data/normalized_24kHz")
PHASES_ANALYSIS_FILE = Path("output/phases_analysis.json")

OUTPUT_SEGS_16K = Path("data/caller_segments_16kHz")
OUTPUT_CONCAT_16K = Path("data/caller_concat_16kHz")
OUTPUT_PHASES_16K = Path("data/caller_phases_16kHz")

OUTPUT_SEGS_24K = Path("data/caller_segments_24kHz")
OUTPUT_CONCAT_24K = Path("data/caller_concat_24kHz")
OUTPUT_PHASES_24K = Path("data/caller_phases_24kHz")

TRIM_MS = 0
MERGE_GAP_MS = 300
MIN_SEG_DUR_S = 0.8


def process_phases(call_id, seg_paths, output_phases_dir, phases_data):
    """Slices extracted segments into phase-level concatenated wavs."""
    conv_id_with_wav = call_id + ".wav"
    conv_phases = phases_data.get(conv_id_with_wav, {}).get("phases", [])
    if not conv_phases:
        return
        
    out_dir = output_phases_dir / call_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, phase in enumerate(conv_phases):
        start_s = parse_time(phase.get("start_time"))
        end_s = parse_time(phase.get("end_time"))
        
        if start_s is None or end_s is None:
            continue
            
        phase_name_raw = phase.get("phase_name", f"phase_{idx}")
        numeric_id = phase_name_raw.split(".")[0].strip() if "." in phase_name_raw else "0"
        
        # Collect paths whose midpoint falls into the phase boundaries
        phase_subset = []
        for p in seg_paths:
            mid = parse_segment_midpoint(p.name)
            if start_s <= mid <= end_s:
                phase_subset.append(p)
                
        if phase_subset:
            # Re-sort to be safe, then concat
            phase_subset.sort(key=lambda x: parse_segment_midpoint(x.name))
            concat_path = out_dir / f"phase_{idx+1}_{numeric_id}_{start_s:.2f}_{end_s:.2f}.wav"
            concat_wavs_ffmpeg(phase_subset, concat_path)

def main():
    if not INPUT_DIAR_DIR.exists():
        print(f"Error: {INPUT_DIAR_DIR} does not exist.")
        return

    json_files = list(INPUT_DIAR_DIR.glob("*.json"))
    print(f"Found {len(json_files)} diarized JSON files.")

    OUTPUT_SEGS_16K.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONCAT_16K.mkdir(parents=True, exist_ok=True)
    OUTPUT_PHASES_16K.mkdir(parents=True, exist_ok=True)
    
    OUTPUT_SEGS_24K.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONCAT_24K.mkdir(parents=True, exist_ok=True)
    OUTPUT_PHASES_24K.mkdir(parents=True, exist_ok=True)
    
    phases_data = {}
    if PHASES_ANALYSIS_FILE.exists():
        try:
            with open(PHASES_ANALYSIS_FILE, 'r', encoding='utf-8') as f:
                phases_data = json.load(f)
        except Exception as e:
            print(f"Failed to load phases: {e}")

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
            
            # Smart-cache: skip segment extraction if segments exist
            if seg_dir_16k.exists() and any(seg_dir_16k.iterdir()):
                seg_paths_16k = sorted(list(seg_dir_16k.glob("*.wav")), key=lambda x: str(x.name))
            else:
                seg_paths_16k = extract_segments_ffmpeg(norm_16k_path, caller_segments, seg_dir_16k)
                
            if seg_paths_16k:
                concat_path_16k = OUTPUT_CONCAT_16K / f"{call_id}.wav"
                if not concat_path_16k.exists():
                    concat_wavs_ffmpeg(seg_paths_16k, concat_path_16k)
                if phases_data:
                    process_phases(call_id, seg_paths_16k, OUTPUT_PHASES_16K, phases_data)

        # 4. Process 24kHz audio
        norm_24k_path = INPUT_NORM_24K / f"{call_id}.wav"
        if norm_24k_path.exists():
            seg_dir_24k = OUTPUT_SEGS_24K / call_id
            
            # Smart-cache: skip segment extraction if segments exist
            if seg_dir_24k.exists() and any(seg_dir_24k.iterdir()):
                seg_paths_24k = sorted(list(seg_dir_24k.glob("*.wav")), key=lambda x: str(x.name))
            else:
                seg_paths_24k = extract_segments_ffmpeg(norm_24k_path, caller_segments, seg_dir_24k)
                
            if seg_paths_24k:
                concat_path_24k = OUTPUT_CONCAT_24K / f"{call_id}.wav"
                if not concat_path_24k.exists():
                    concat_wavs_ffmpeg(seg_paths_24k, concat_path_24k)
                if phases_data:
                    process_phases(call_id, seg_paths_24k, OUTPUT_PHASES_24K, phases_data)

if __name__ == "__main__":
    main()

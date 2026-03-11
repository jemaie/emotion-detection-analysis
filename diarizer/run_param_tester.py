"""
Compare segmentation parameters side-by-side.

Reads cached diarized JSONs from existing output folders,
re-runs post-processing + audio extraction with BOTH parameter sets,
and writes results to out_param_test/ so nothing is overwritten.

Output structure:
  out_param_test/
    <method>/
      current/          (trim=200, merge_gap=300, min_seg=0.6)
        caller_segments/<call_id>/seg_XXXX_*.wav
        caller_concat/<call_id>.wav
      proposed/         (trim=100, merge_gap=300, min_seg=0.4)
        caller_segments/<call_id>/seg_XXXX_*.wav
        caller_concat/<call_id>.wav
"""

import json
from pathlib import Path

from run_batch_scripts.role_assign import assign_roles
from run_batch_scripts.segment_postprocess import postprocess_caller_segments
from run_batch_scripts.extract_audio import extract_segments_ffmpeg, concat_wavs_ffmpeg

# ─── Config ──────────────────────────────────────────────────────────────
NORM_DIR = Path("data/normalized")
OUT_BASE = Path("out_param_test")

# The 3 contender methods and where their diarized JSONs live
METHODS = {
    "pyannote_mapped": Path("out_mapped/pyannote/diarized"),
    "openai_mapped":   Path("out_mapped/openai/diarized"),
    "openai_2refs":    Path("out_openai/2_refs/diarized"),
}

# 3 representative test calls:
#   - high segment count / heavy filtering
#   - medium count / high drop rate
#   - small call / OpenAI over-segmentation visible
TEST_CALLS = [
    "conv__+4915203230182_22-08-2024_8_06_41",
    "conv__+49713182652_09-09-2024_10_08_43",
    "conv__+49706353138_26-07-2024_9_13_32",
]

# Parameter sets to compare
PARAM_SETS = {
    "current": {"trim_ms": 200, "merge_gap_ms": 300, "min_seg_dur_s": 0.6},
    "var_1":   {"trim_ms": 100, "merge_gap_ms": 300, "min_seg_dur_s": 0.4},
    "var_2":   {"trim_ms": 250, "merge_gap_ms": 450, "min_seg_dur_s": 0.8},
    "mixed":   {"trim_ms": 200, "merge_gap_ms": 300, "min_seg_dur_s": 0.8},
}


def process(method_name: str, diar_dir: Path, call_id: str, param_name: str, params: dict):
    """Run post-processing + extraction for one (method, call, param_set) combo."""
    diar_path = diar_dir / f"{call_id}.json"
    if not diar_path.exists():
        print(f"  SKIP {method_name}/{call_id} — diarized JSON not found")
        return

    norm_path = NORM_DIR / f"{call_id}.wav"
    if not norm_path.exists():
        print(f"  SKIP {call_id} — normalized WAV not found")
        return

    diarized = json.loads(diar_path.read_text(encoding="utf-8"))
    segments = diarized.get("segments", []) or []

    # Role assignment (same as original pipeline)
    speaker_to_role, speaker_durations, flags = assign_roles(diarized)

    # Post-process with the given param set
    caller_segments = postprocess_caller_segments(
        diarized_segments=segments,
        speaker_to_role=speaker_to_role,
        **params,
    )

    # Output dirs
    out_method = OUT_BASE / method_name / param_name
    seg_dir = out_method / "caller_segments" / call_id
    concat_dir = out_method / "caller_concat"
    concat_dir.mkdir(parents=True, exist_ok=True)

    # Extract per-segment WAVs
    seg_paths = extract_segments_ffmpeg(norm_path, caller_segments, seg_dir)

    # Concatenate
    concat_path = concat_dir / f"{call_id}.wav"
    if seg_paths:
        concat_wavs_ffmpeg(seg_paths, concat_path)

    # Compute stats
    raw_caller = sum(
        1 for s in segments
        if speaker_to_role.get(s.get("speaker", "unknown")) == "caller"
    )
    total_dur = sum(s["end"] - s["start"] for s in caller_segments)

    print(f"  {method_name:20s} | {param_name:8s} | raw={raw_caller:3d} -> final={len(caller_segments):3d} "
          f"(drop {100*(1 - len(caller_segments)/raw_caller) if raw_caller else 0:.0f}%) | "
          f"dur={total_dur:.1f}s | {len(seg_paths)} WAVs")


def main():
    print("=" * 80)
    print("Segmentation Parameter A/B Test")
    print("=" * 80)

    for pname, pset in PARAM_SETS.items():
        print(f"\n  {pname}: trim={pset['trim_ms']}ms, merge_gap={pset['merge_gap_ms']}ms, "
              f"min_seg={pset['min_seg_dur_s']}s")

    for call_id in TEST_CALLS:
        print(f"\n{'-' * 80}")
        print(f"Call: {call_id}")
        print(f"{'-' * 80}")

        for method_name, diar_dir in METHODS.items():
            for param_name, params in PARAM_SETS.items():
                process(method_name, diar_dir, call_id, param_name, params)

    print(f"\n{'=' * 80}")
    print(f"Done. Results in: {OUT_BASE.resolve()}")
    print(f"\nTo compare, listen to:")
    print(f"  out_param_test/<method>/current/caller_concat/<call_id>.wav")
    print(f"  out_param_test/<method>/proposed/caller_concat/<call_id>.wav")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

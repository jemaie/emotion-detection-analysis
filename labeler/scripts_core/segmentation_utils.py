import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def assign_roles(diarized: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, Any]]:
    """
    Returns:
      speaker_to_role: maps speaker label -> role (agent/caller/unknown)
      speaker_durations: total speech duration per speaker
      flags: info for auditing
    """
    durations = defaultdict(float)
    for seg in diarized.get("segments", []):
        spk = seg.get("speaker", "unknown")
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end > start:
            durations[spk] += (end - start)

    speaker_to_role: Dict[str, str] = {}
    flags: Dict[str, Any] = {
        "agent_matched_by_reference": False,
        "fallback_used": False,
        "num_speakers": len(durations),
    }

    agent_speaker_id = diarized.get("agent_speaker_id")

    # Match by explicitly provided agent_speaker_id (e.g., from offline mapping)
    if agent_speaker_id and agent_speaker_id in durations:
        flags["agent_matched_by_reference"] = True
        speaker_to_role[agent_speaker_id] = "agent"
        other_speakers = [s for s in durations.keys() if s != agent_speaker_id]
        if other_speakers:
            caller_spk = max(other_speakers, key=lambda s: durations[s])
            speaker_to_role[caller_spk] = "caller"
            for s in other_speakers:
                if s != caller_spk:
                    speaker_to_role[s] = "other"
        return speaker_to_role, dict(durations), flags

    # Fallback for online diarizers that rewrite the speaker directly to "agent"
    if "agent" in durations:
        flags["agent_matched_by_reference"] = True
        speaker_to_role["agent"] = "agent"
        other_speakers = [s for s in durations.keys() if s != "agent"]
        if other_speakers:
            caller_spk = max(other_speakers, key=lambda s: durations[s])
            speaker_to_role[caller_spk] = "caller"
            for s in other_speakers:
                if s != caller_spk:
                    speaker_to_role[s] = "other"
        return speaker_to_role, dict(durations), flags

    # Anything else: unknown (should be rare)
    flags["fallback_used"] = True
    for s in durations.keys():
        speaker_to_role[s] = "unknown"
    return speaker_to_role, dict(durations), flags


def postprocess_caller_segments(
    diarized_segments: List[Dict[str, Any]],
    speaker_to_role: dict,
    trim_ms: int = 0,
    merge_gap_ms: int = 300,
    min_seg_dur_s: float = 0.8,
):
    """
    Filters out pieces overlapping with the agent by splitting caller segments into sub-segments.
    Trims boundaries, drops tiny segments, and merges small gaps.
    """
    trim = trim_ms / 1000.0
    merge_gap = merge_gap_ms / 1000.0

    stats = {
        "num_raw_caller": 0,
        "num_dropped_trim_dur": 0,
        "num_dropped_overlap": 0,
        "num_merged_into": 0,
        "num_dropped": 0, # total effectively dropped or completely removed
    }

    # Extract agent segments (sorted)
    agent_segs = []
    for seg in diarized_segments:
        spk = seg.get("speaker", "unknown")
        if speaker_to_role.get(spk) == "agent":
            agent_segs.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0))
            })
    agent_segs.sort(key=lambda x: x["start"])

    # Extract raw caller segments
    raw_caller = []
    for seg in diarized_segments:
        spk = seg.get("speaker", "unknown")
        if speaker_to_role.get(spk) == "caller":
            raw_caller.append({
                "speaker": spk,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": (seg.get("text") or "").strip(),
            })
            stats["num_raw_caller"] += 1

    # Apply trim and split overlaps
    caller_processed = []
    for c in raw_caller:
        c_start = c["start"] + trim
        c_end = c["end"] - trim
        
        if c_end <= c_start:
            stats["num_dropped_trim_dur"] += 1
            stats["num_dropped"] += 1
            continue

        # Start with the whole piece as 1 valid chunk
        chunks = [{"speaker": c["speaker"], "start": c_start, "end": c_end, "text": c["text"], "overlapped": False}]
        
        for a in agent_segs:
            a_start, a_end = a["start"], a["end"]
            new_chunks = []
            for chunk in chunks:
                chk_start, chk_end = chunk["start"], chunk["end"]
                
                # Check for overlap
                if a_start < chk_end and a_end > chk_start:
                    chunk["overlapped"] = True
                    # Left side
                    if chk_start < a_start:
                        new_chunks.append({"speaker": chunk["speaker"], "start": chk_start, "end": a_start, "text": chunk["text"], "overlapped": True})
                    # Right side
                    if a_end < chk_end:
                        new_chunks.append({"speaker": chunk["speaker"], "start": a_end, "end": chk_end, "text": chunk["text"], "overlapped": True})
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
            
        # Add resulted chunks
        for chunk in chunks:
            if chunk["overlapped"]:
                stats["num_dropped_overlap"] += 1 # We mark sub-chunks born from overlap splitting for metrics
            caller_processed.append(chunk)

    # Filter out chunks that fall under min_seg_dur_s
    caller_filtered = []
    for chk in caller_processed:
        if (chk["end"] - chk["start"]) < min_seg_dur_s:
            stats["num_dropped_trim_dur"] += 1
            stats["num_dropped"] += 1
        else:
            caller_filtered.append(chk)

    if not caller_filtered:
        return [], stats

    # Sort and merge remaining gaps
    caller_filtered.sort(key=lambda s: s["start"])
    merged = []
    cur = caller_filtered[0].copy()
    
    for nxt in caller_filtered[1:]:
        gap = nxt["start"] - cur["end"]
        if gap <= merge_gap:
            # Merge
            cur["end"] = max(cur["end"], nxt["end"])
            if nxt["text"]:
                if cur["text"]:
                    cur["text"] += " " + nxt["text"]
                else:
                    cur["text"] = nxt["text"]
            stats["num_merged_into"] += 1
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)

    # Final duration check after any potential merges
    final = []
    for m in merged:
        if (m["end"] - m["start"]) >= min_seg_dur_s:
            final.append(m)

    return final, stats


def extract_segments_ffmpeg(
    audio_wav: Path,
    segments: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    """
    Extract each segment as its own WAV file (copied, no re-encode).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        out_path = out_dir / f"seg_{i:04d}_{start:.2f}_{end:.2f}.wav"

        # Note: Added -y to overwrite existing chunks safely
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_wav),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_paths.append(out_path)

    return out_paths


def concat_wavs_ffmpeg(segment_paths: List[Path], out_path: Path) -> None:
    """
    Concatenate WAVs using ffmpeg concat demuxer.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not segment_paths:
        return

    # Create concat list file
    list_file = out_path.parent / (out_path.stem + "_concat_list.txt")
    lines = [f"file '{p.resolve()}'" for p in segment_paths]
    list_file.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Cleanup
    try:
        list_file.unlink()
    except OSError:
        pass


def parse_time(t_str):
    """Parses MM:SS string or float nan to seconds."""
    if t_str == "-" or t_str is None: return None
    import math
    if isinstance(t_str, float) and math.isnan(t_str): return None
    t_str = str(t_str)
    parts = t_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return None

def parse_segment_midpoint(segment_name):
    """Extracts midpoint from segment filename (e.g. seg_0000_10.5_15.5.wav or phase_1_2_10.5_15.5.wav)."""
    parts = str(segment_name).split('_')
    if len(parts) >= 4:
        try:
            start = float(parts[-2])
            end_part = parts[-1].replace('.wav', '').replace('.json', '')
            end = float(end_part)
            return (start + end) / 2.0
        except ValueError:
            pass
    return 0.0

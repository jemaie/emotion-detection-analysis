from typing import List, Dict, Any

def postprocess_caller_segments(
    diarized_segments: List[Dict[str, Any]],
    speaker_to_role: dict,
    trim_ms: int = 250,
    merge_gap_ms: int = 300,
    min_seg_dur_s: float = 0.7,
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

    # Since splitting a raw segment into 2 pieces creates +1 total segment, 'num_dropped' needs to behave predictably. 
    # For now, we will simply rely on the detailed metrics in eval_agents.
    return final, stats



from typing import List, Dict, Any

def postprocess_caller_segments(
    diarized_segments: List[Dict[str, Any]],
    speaker_to_role: dict,
    trim_ms: int = 250,
    merge_gap_ms: int = 300,
    min_seg_dur_s: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Filters to caller, trims boundaries, drops tiny segments, merges small gaps.
    Returns new list of segments with updated start/end and merged text concatenation.
    """
    trim = trim_ms / 1000.0
    merge_gap = merge_gap_ms / 1000.0

    # 1) filter to caller
    caller = []
    for seg in diarized_segments:
        spk = seg.get("speaker", "unknown")
        if speaker_to_role.get(spk) != "caller":
            continue
        start = float(seg.get("start", 0.0)) + trim
        end = float(seg.get("end", 0.0)) - trim
        if end <= start:
            continue
        if (end - start) < min_seg_dur_s:
            continue
        caller.append({
            "speaker": spk,
            "start": start,
            "end": end,
            "text": (seg.get("text") or "").strip(),
        })

    if not caller:
        return []

    # 2) sort
    caller.sort(key=lambda s: s["start"])

    # 3) merge small gaps
    merged: List[Dict[str, Any]] = []
    cur = caller[0].copy()
    for nxt in caller[1:]:
        gap = nxt["start"] - cur["end"]
        if gap <= merge_gap:
            # merge
            cur["end"] = max(cur["end"], nxt["end"])
            if nxt["text"]:
                if cur["text"]:
                    cur["text"] += " " + nxt["text"]
                else:
                    cur["text"] = nxt["text"]
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)

    return merged

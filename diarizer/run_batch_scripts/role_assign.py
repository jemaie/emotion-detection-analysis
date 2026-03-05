from collections import defaultdict
from typing import Dict, Any, Tuple

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
        for s in durations.keys():
            if s != agent_speaker_id:
                speaker_to_role[s] = "caller"
        return speaker_to_role, dict(durations), flags

    # Fallback for online diarizers that rewrite the speaker directly to "agent"
    if "agent" in durations:
        flags["agent_matched_by_reference"] = True
        speaker_to_role["agent"] = "agent"
        for s in durations.keys():
            if s != "agent":
                speaker_to_role[s] = "caller"
        return speaker_to_role, dict(durations), flags

    # Fallback: assume 2 speakers and agent tends to speak less
    # if len(durations) == 2:
    #     flags["fallback_used"] = True
    #     spk_sorted = sorted(durations.keys(), key=lambda s: durations[s])  # shortest first
    #     agent_spk, caller_spk = spk_sorted[0], spk_sorted[1]
    #     speaker_to_role[agent_spk] = "agent"
    #     speaker_to_role[caller_spk] = "caller"
    #     return speaker_to_role, dict(durations), flags

    # Anything else: unknown (should be rare)
    flags["fallback_used"] = True
    for s in durations.keys():
        speaker_to_role[s] = "unknown"
    return speaker_to_role, dict(durations), flags
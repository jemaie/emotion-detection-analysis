from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pyannoteai.sdk import Client

load_dotenv()

# Cache voiceprints in memory so we don't recalculate them for every file
_voiceprint_cache: Dict[str, str] = {}

def diarize_transcribe(wav_path: Path, agent_ref_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """
    Calls Pyannote AI platform for speaker identification (if agent_ref_paths provided) 
    or speaker diarization (if none provided).
    Returns diarized_json.
    Output includes diarized segments with start/end/speaker.
    """
    client = Client()

    # 1. Upload main audio
    media_url = client.upload(str(wav_path))

    job_id = None
    voiceprints = {}

    # 2. Upload refs and create voiceprints
    if agent_ref_paths and len(agent_ref_paths) > 0:
        for i, ref_path in enumerate(agent_ref_paths):
            ref_str = str(ref_path)
            
            if ref_str not in _voiceprint_cache:
                ref_media_url = client.upload(ref_str)
                ref_job_id = client.voiceprint(ref_media_url)
                ref_result = client.retrieve(ref_job_id)
                
                # Retrieve the base64 encoded voiceprint from the output
                vp = ref_result.get("output", {}).get("voiceprint")
                if vp:
                    _voiceprint_cache[ref_str] = vp

            if ref_str in _voiceprint_cache:
                voiceprints[f"agent_{i}"] = _voiceprint_cache[ref_str]

    # 3. Call identify or diarize
    if voiceprints:
        job_id = client.identify(
            media_url, 
            voiceprints=voiceprints, 
            min_speakers=2,
            max_speakers=3
        )
    else:
        job_id = client.diarize(
            media_url, 
            min_speakers=2,
            max_speakers=3
        )

    # 4. Wait for and retrieve results
    result = client.retrieve(job_id)
    output = result.get("output", {})
    
    # 'identify' returns an 'identification' array; 'diarize' returns a 'diarization' array.
    segments_list = output.get("identification") or output.get("diarization", [])

    standard_segments = []
    
    # 5. Parse output to standard format
    for seg in segments_list:
        spk = seg.get("speaker", "unknown")
        
        # 'identify' replaces the generic speaker label with the voiceprint label e.g., 'agent_0'.
        if spk.startswith("agent"):
            spk = "agent"

        standard_segments.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "speaker": spk
        })

    return {"segments": standard_segments}

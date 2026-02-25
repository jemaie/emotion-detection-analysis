import base64
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def _to_data_url_wav(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return "data:audio/wav;base64," + b64

def diarize_transcribe(wav_path: Path, agent_ref_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """
    Calls OpenAI diarization-capable transcription endpoint and returns diarized_json.
    Output includes diarized segments with start/end/speaker + text.
    """
    extra_body = None
    if agent_ref_paths:
        extra_body = {
            "known_speaker_names": ["agent"] * len(agent_ref_paths),
            "known_speaker_references": [_to_data_url_wav(p) for p in agent_ref_paths],
        }

    with wav_path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe-diarize",
            file=f,
            response_format="diarized_json",
            chunking_strategy="auto",
            extra_body=extra_body,
        )

    return resp.model_dump()

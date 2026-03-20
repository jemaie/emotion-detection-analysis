import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from google.cloud import speech_v1p1beta1 as speech

def diarize_transcribe(wav_path: Path, agent_ref_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """
    Calls Google Cloud Speech-to-Text V1P1Beta1 API with Speaker Diarization.
    Note: agent_ref_paths is ignored since Google API does not support reference audio for diarization natively.
    Returns the raw Google response as a Python dictionary.
    """
    client = speech.SpeechClient()

    with wav_path.open("rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=2,
        max_speaker_count=2,
    )

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="de-DE",
        diarization_config=diarization_config,
    )

    # Note: For longer audio, we should use long_running_recognize,
    # but based on the context of run_batch logic where it processes segments,
    # recognize might suffice or we use long_running_recognize to be safe.
    # Let's use long_running_recognize to avoid 1 minute limits.
    operation = client.long_running_recognize(config=config, audio=audio)

    # print(f"Waiting for Google Speech-to-Text operation to complete for {wav_path.name}...")
    response = operation.result(timeout=600)

    try:
        return type(response).to_dict(response)
    except AttributeError:
        # Fallback in case it's not a proto-plus subclass with to_dict
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(response._pb if hasattr(response, '_pb') else response)


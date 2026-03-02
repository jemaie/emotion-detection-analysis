import os
import copy
from pathlib import Path
from typing import Dict, Any

import torch
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.audio import Audio
from pyannote.core import Segment
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

load_dotenv()

# Cache model to avoid reloading
_embedding_model = None

def get_embedding_model() -> Inference:
    global _embedding_model
    if _embedding_model is None:
        model = Model.from_pretrained("pyannote/embedding", token=os.environ.get("HF_TOKEN"))
        _embedding_model = Inference(model, window="whole")
    return _embedding_model

def get_segment_embedding(wav_path: Path, start: float, end: float) -> Any:
    """Extracts the numerical identity embedding for a specific time window in a wav file."""
    inference = get_embedding_model()
    # Handle floating point inaccuracies by capping end to real duration
    audio = Audio()
    file_duration = audio.get_duration(str(wav_path))
    capped_end = min(end, file_duration)
    segment = Segment(start, capped_end)
    # The Inference object's crop method computes the embedding for the specific segment
    embedding = inference.crop(str(wav_path), segment)
    return embedding

def map_speakers_to_roles(diarized_json: Dict[str, Any], wav_path: Path, ref_path: Path) -> Dict[str, Any]:
    """
    Takes an unmapped diarized JSON containing unmapped speaker labels (e.g., SPEAKER_XX, A, B),
    extracts representative embeddings for each speaker, compares them
    against the ref_path using Cosine Similarity, and maps the closest match
    to the 'agent' role.
    """
    segments = diarized_json.get("segments", [])
    if not segments:
        return diarized_json

    # 1. Get reference embedding
    # Assuming the reference is purely the agent speaking, use the entire file duration.
    import librosa
    ref_duration = librosa.get_duration(path=str(ref_path))
    ref_embedding = get_segment_embedding(ref_path, 0.0, ref_duration)

    # 2. Extract representative segments for each speaker
    # A simple approach: find the longest segment for each speaker as the representative
    speaker_longest_seg: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk: continue
            
        dur = seg["end"] - seg["start"]
        if spk not in speaker_longest_seg or dur > (speaker_longest_seg[spk]["end"] - speaker_longest_seg[spk]["start"]):
            speaker_longest_seg[spk] = {"start": seg["start"], "end": seg["end"]}

    # 3. Compute cosine distances
    spk_distances = {}
    for spk, time_seg in speaker_longest_seg.items():
        try:
            spk_emb = get_segment_embedding(wav_path, time_seg["start"], time_seg["end"])
            # Scipy cosine distance metric (0 is identical, 1 is orthogonal, 2 is opposite)
            # Distance is 1 - Cosine Similarity
            distance = cosine(ref_embedding, spk_emb)
            spk_distances[spk] = distance
        except Exception as e:
            print(f"Failed to extract embedding for {spk}: {e}")
            spk_distances[spk] = 999.0 # Arbitrary high distance on failure

    # 4. Map the closest speaker to "agent", ignore others
    mapped_json = copy.deepcopy(diarized_json)
    if not spk_distances:
        return mapped_json

    # The speaker with the minimum distance is the most similar to the reference
    best_match_spk = min(spk_distances, key=spk_distances.get)

    for seg in mapped_json["segments"]:
        if seg.get("speaker") == best_match_spk:
            seg["speaker"] = "agent"
        else:
            seg["speaker"] = "unknown" # Leave others to be mapped by role_assign.py natively

    return mapped_json

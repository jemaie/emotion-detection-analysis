import torch
import logging
from transformers import pipeline
from speechbrain.inference.interfaces import foreign_class

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, name: str):
        self.name = name
    
    def predict(self, audio_path: str) -> dict:
        raise NotImplementedError

class HuggingFaceModel(BaseModel):
    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        self.model_id = model_id
        try:
            # Basic audio classification pipeline
            self.classifier = pipeline(
                "audio-classification", 
                model=self.model_id, 
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded Hugging Face model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {model_id}: {e}")
            self.classifier = None
            
    def predict(self, audio_path: str) -> dict:
        if not self.classifier:
            return {"emotion": "error_loading", "confidence": 0.0}
        
        try:
            result = self.classifier(audio_path)
            # Result is usually a list of dicts: [{'score': 0.9, 'label': 'angry'}, ...]
            if result:
                best = max(result, key=lambda x: x['score'])
                return {"emotion": best['label'], "confidence": best['score']}
            return {"emotion": "unknown", "confidence": 0.0}
        except Exception as e:
            logger.error(f"Inference failed for {self.name} on {audio_path}: {e}")
            return {"emotion": "error", "confidence": 0.0}

class SpeechBrainModel(BaseModel):
    def __init__(self, name: str, source: str):
        super().__init__(name)
        self.source = source
        try:
            # Specific loading procedure for Speechbrain IEMOCAP
            self.classifier = foreign_class(
                source=self.source, 
                pymodule_file="custom_interface.py", 
                classname="CustomEncoderWav2vec2Classifier",
                run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            logger.info(f"Loaded SpeechBrain model: {source} on {'cuda' if torch.cuda.is_available() else 'cpu'}")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain model {source}: {e}")
            self.classifier = None
            
    def predict(self, audio_path: str) -> dict:
        if not self.classifier:
            return {"emotion": "error_loading", "confidence": 0.0}
        try:
            # Returns: out_prob, score, index, text_lab
            out_prob, score, index, text_lab = self.classifier.classify_file(audio_path)
            # text_lab is usually a list with one element
            emotion = text_lab[0] if isinstance(text_lab, list) else text_lab
            confidence = score.item() if hasattr(score, 'item') else float(score)
            return {"emotion": emotion, "confidence": confidence}
        except Exception as e:
            logger.error(f"Inference failed for {self.name} on {audio_path}: {e}")
            return {"emotion": "error", "confidence": 0.0}


def get_model_factories() -> list[dict]:
    factories = []
    
    # 1. ehcalabres wav2vec2
    factories.append({"name": "ehcalabres", "factory": lambda: HuggingFaceModel("ehcalabres", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")})
    
    # 2. SpeechBrain IEMOCAP
    factories.append({"name": "speechbrain", "factory": lambda: SpeechBrainModel("speechbrain", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")})
    
    # 3. Multilingual / German HuBERT Pipeline
    factories.append({"name": "hubert_german", "factory": lambda: HuggingFaceModel("hubert_german", "superb/hubert-large-superb-er")}) 
    
    # 4. Emotion2Vec
    factories.append({"name": "emotion2vec", "factory": lambda: HuggingFaceModel("emotion2vec", "emotion2vec/emotion2vec_plus_base")})
    
    return factories

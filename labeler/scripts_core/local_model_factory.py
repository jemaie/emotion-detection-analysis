from pathlib import Path
import torch
import logging

from transformers import pipeline
from speechbrain.inference.interfaces import foreign_class
from funasr import AutoModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - LOCAL - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

class BaseModel:
    def __init__(self, name: str):
        self.name = name
    
    def predict(self, audio_path: Path) -> dict:
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
                device=0 if torch.cuda.is_available() else -1,
                top_k=None,
                function_to_apply="softmax"
            )
            logger.info(f"Loaded Hugging Face model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {model_id}: {e}")
            self.classifier = None
            
    def predict(self, audio_path: Path) -> dict:
        if not self.classifier:
            return {"emotion": "error_loading", "confidence": 0.0}
        
        try:
            # Convert to string for the pipeline
            result = self.classifier(str(audio_path))
            # Result is usually a list of dicts: [{'score': 0.9, 'label': 'angry'}, ...]
            if result:
                best = max(result, key=lambda x: x['score'])
                all_scores = {item["label"]: float(item["score"]) for item in result}
                top_scores = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))
                return {
                    "emotion": best['label'], 
                    "confidence": float(best['score']),
                    "scores": top_scores
                }
            return {"emotion": "unknown", "confidence": 0.0, "scores": {}}
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
            
    def predict(self, audio_path: Path) -> dict:
        if not self.classifier:
            return {"emotion": "error_loading", "confidence": 0.0}
        try:
            # Returns: out_prob, score, index, text_lab
            # Convert to string for the classifier
            out_prob, score, index, text_lab = self.classifier.classify_file(str(audio_path))
            # text_lab is usually a list with one element
            emotion = text_lab[0] if isinstance(text_lab, list) else text_lab
            confidence = score.item() if hasattr(score, 'item') else float(score)
            
            # Preserve whole distribution
            # labels are in self.classifier.hparams.label_encoder.ind2lab
            # out_prob is [batch, num_classes]
            scores = {}
            if hasattr(self.classifier, "hparams") and hasattr(self.classifier.hparams, "label_encoder"):
                labels = self.classifier.hparams.label_encoder.ind2lab
                probs = out_prob[0].cpu().numpy()
                all_scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
                scores = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))
            
            return {
                "emotion": emotion, 
                "confidence": confidence,
                "scores": scores
            }
        except Exception as e:
            logger.error(f"Inference failed for {self.name} on {audio_path}: {e}")
            return {"emotion": "error", "confidence": 0.0}


class FunASRModel(BaseModel):
    def __init__(self, name: str, model_id: str):
        super().__init__(name)
        self.model_id = model_id
        try:
            # Load FunASR model, using hub="hf" to ensure we pull from Hugging Face
            self.model = AutoModel(
                model=self.model_id, 
                hub="hf",
                disable_pbar=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"Loaded FunASR model: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load FunASR model {model_id}: {e}")
            self.model = None
            
    def predict(self, audio_path: Path) -> dict:
        if not self.model:
            return {"emotion": "error_loading", "confidence": 0.0}
        
        try:
            res = self.model.generate(
                input=str(audio_path),
                extract_embedding=False
            )
            
            if res and isinstance(res, list) and len(res) > 0:
                prediction = res[0]
                
                labels = prediction.get('labels')
                scores = prediction.get('scores')
                
                if labels and scores:
                    max_idx = scores.index(max(scores))
                    # Strip Chinese prefix (e.g. "愤怒/angry" -> "angry")
                    clean_labels = [l.split("/")[-1] for l in labels]
                    all_scores = {clean_labels[i]: float(scores[i]) for i in range(len(clean_labels))}
                    top_scores = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))
                    return {
                        "emotion": clean_labels[max_idx],
                        "confidence": float(scores[max_idx]),
                        "scores": top_scores
                    }
                
                return {"emotion": "unknown", "confidence": 0.0, "scores": {}}
        except Exception as e:
            logger.error(f"Inference failed for {self.name} on {audio_path}: {e}")
            return {"emotion": "error", "confidence": 0.0}


def get_model_factories() -> list[dict]:
    factories = []
    factories.append({"name": "ehcalabres/wav2vec2", "factory": lambda: HuggingFaceModel("ehcalabres/wav2vec2", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")})
    factories.append({"name": "speechbrain/wav2vec2", "factory": lambda: SpeechBrainModel("speechbrain/wav2vec2", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")})
    factories.append({"name": "superb/wav2vec2_base", "factory": lambda: HuggingFaceModel("superb/wav2vec2_base", "superb/wav2vec2-base-superb-er")})
    factories.append({"name": "superb/wav2vec2_large", "factory": lambda: HuggingFaceModel("superb/wav2vec2_large", "superb/wav2vec2-large-superb-er")})
    factories.append({"name": "superb/hubert_base", "factory": lambda: HuggingFaceModel("superb/hubert_base", "superb/hubert-base-superb-er")})
    factories.append({"name": "superb/hubert_large", "factory": lambda: HuggingFaceModel("superb/hubert_large", "superb/hubert-large-superb-er")})
    factories.append({"name": "iic/emotion2vec_base", "factory": lambda: FunASRModel("iic/emotion2vec_base", "iic/emotion2vec_plus_base")})
    factories.append({"name": "iic/emotion2vec_large", "factory": lambda: FunASRModel("iic/emotion2vec_large", "iic/emotion2vec_plus_large")})
    return factories

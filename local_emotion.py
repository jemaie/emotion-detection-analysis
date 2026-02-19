from transformers import pipeline
import numpy as np

class LocalEmotionDetector:
    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        # ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition is a good candidate
        # or "superb/wav2vec2-large-superb-er"
        self.model_name = model_name
        self.classifier = None

    def load_model(self):
        print(f"Loading local model: {self.model_name}...")
        try:
            self.classifier = pipeline("audio-classification", model=self.model_name)
            print("Local model loaded successfully.")
        except Exception as e:
            print(f"Failed to load local model: {e}")

    def predict(self, audio_path):
        """
        Predict emotion from audio file path or numpy array.
        Pipeline handles loading from file path best.
        """
        if self.classifier is None:
            self.load_model()
            
        # pipeline supports file path directly
        result = self.classifier(audio_path)
        # Result is list of dicts: [{'score': 0.9, 'label': 'angry'}, ...]
        return result


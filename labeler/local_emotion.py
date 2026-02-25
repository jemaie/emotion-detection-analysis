from transformers import pipeline

class LocalEmotionDetector:
    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", log_callback=None):
        # ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition is a good candidate
        # or "superb/wav2vec2-large-superb-er"
        self.model_name = model_name
        self.classifier = None
        self.log_callback = log_callback

    def log(self, message):
        if self.log_callback:
            # Blue [LOCAL]
            self.log_callback(f"\033[94m[LOCAL]\033[0m {message}")

    def load_model(self):
        self.log(f"Loading local model: {self.model_name}...")
        try:
            self.classifier = pipeline("audio-classification", model=self.model_name)
            self.log("Local model loaded successfully.")
        except Exception as e:
            self.log(f"Failed to load local model: {e}")

    def predict(self, audio_path):
        """
        Predict emotion from audio file path or numpy array.
        Pipeline handles loading from file path best.
        """
        if self.classifier is None:
            self.load_model()
            
        self.log(f"Predicting emotion locally for: {audio_path}")
        # pipeline supports file path directly
        result = self.classifier(audio_path)
        self.log(f"Local prediction finished.")
        # Result is list of dicts: [{'score': 0.9, 'label': 'angry'}, ...]
        return result


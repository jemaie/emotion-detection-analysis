import pandas as pd
import os
from datetime import datetime

class ResultsLogger:
    def __init__(self, filename="results.csv"):
        self.filename = filename
        self.columns = [
            "timestamp", "filename", "openai_emotion", "openai_confidence",
            "local_emotion", "local_confidence", "user_preference", "comments"
        ]
        
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)

    def log_result(self, filename, openai_res, local_res, user_pref, comments=""):
        """
        Logs a single experiment result.
        openai_res: dict {'emotion': 'happy', 'confidence': 0.9}
        local_res: dict {'emotion': 'happy', 'score': 0.8}
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "openai_emotion": openai_res.get('emotion', 'unknown'),
            "openai_confidence": openai_res.get('confidence', 0.0),
            "local_emotion": local_res[0]['label'] if isinstance(local_res, list) and local_res else 'unknown',
            "local_confidence": local_res[0]['score'] if isinstance(local_res, list) and local_res else 0.0,
            "user_preference": user_pref,
            "comments": comments
        }
        
        df = pd.DataFrame([entry])
        df.to_csv(self.filename, mode='a', header=False, index=False)
        print(f"Result logged to {self.filename}")

    def get_processed_filenames(self):
        """Returns a set of filenames that have already been processed."""
        if not os.path.exists(self.filename):
            return set()
        try:
            df = pd.read_csv(self.filename)
            # Check if headers match what we expect or just look for 'filename'
            if 'filename' in df.columns:
                return set(df['filename'].astype(str).tolist())
        except Exception as e:
            print(f"Error reading results log: {e}")
        return set()

import asyncio
import os
import json
from audio_manager import AudioManager
from openai_client import OpenAIRealtimeClient
from local_emotion import LocalEmotionDetector
from results_logger import ResultsLogger

class EmotionOrchestrator:
    def __init__(self, log_callback=None):
        self.audio = AudioManager(log_callback=log_callback)
        self.openai = OpenAIRealtimeClient(log_callback=log_callback)
        self.local = LocalEmotionDetector(log_callback=log_callback)
        self.logger = ResultsLogger(log_callback=log_callback)
        self.current_file = None
        self.log_callback = log_callback

    def log(self, message):
        if self.log_callback:
            # Cyan [ORCHESTRATOR]
            self.log_callback(f"\033[96m[ORCHESTRATOR]\033[0m {message}")

    def get_next_file(self, folder_path="aufnahmen"):
        """
        Scans the folder for .mp4 files and returns the first one
        that hasn't been logged in results.csv.
        """
        if not os.path.exists(folder_path):
            self.log(f"Folder not found: {folder_path}")
            return None
            
        # Get all MP4 files
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
        all_files.sort() # Ensure deterministic order
        
        # Get processed files
        processed = self.logger.get_processed_filenames()
        
        # normalized check (just filename, assuming unique filenames in the folder)
        for f in all_files:
            # We log the full path or just the filename? 
            # In run_analysis below, we log what is passed. 
            # To be safe, let's check if the filename is 'in' the set of processed entries
            # or if the full path is there.
            # The logger logs 'file_path'.
            
            # Use absolute path for robustness if needed, but simple filename match is preferred for the user request "title as the key"
            if f not in processed and os.path.join(folder_path, f) not in processed:
                return os.path.join(folder_path, f)
        
        return None

    async def run_analysis(self, file_path=None, status_callback=None, log_callback=None):
        """
        Runs the full analysis pipeline.
        If file_path is None, it automatically picks the next unprocessed file.
        """
        if log_callback:
            self.log_callback = log_callback
            # Propagate to sub-modules
            self.audio.log_callback = log_callback
            self.openai.log_callback = log_callback
            self.local.log_callback = log_callback
            self.logger.log_callback = log_callback
            
        if status_callback: status_callback("Finding next file...")
        
        if file_path is None:
            file_path = self.get_next_file()
            
        if not file_path:
            if status_callback: status_callback("No more files.")
            return {"error": "No more unprocessed files found."}

        if not os.path.exists(file_path):
            if status_callback: status_callback("File not found.")
            return {"error": "File not found"}
            
        self.current_file = file_path

        if status_callback: status_callback(f"Loading {os.path.basename(file_path)}...")
        self.log(f"Starting analysis for: {file_path}")

        # Load audio (used for both playback and sending)
        self.log("Loading audio data...")
        data, fs = self.audio.load_audio(file_path)
        self.log(f"Audio loaded: {len(data)} samples at {fs}Hz")

        if status_callback: status_callback("Analyzing (OpenAI + Local)...")
        # Create tasks
        # 1. Playback (we can await this or run safely in background if we want parallel processing)
        # However, we want the user to listen WHILE processing happens.
        
        self.log("Starting playback...")
        playback_task = asyncio.create_task(self.audio.play_audio(file_path))
        
        # 2. OpenAI Analysis
        self.log("Starting OpenAI Analysis task...")
        self.openai.log_callback = self.log_callback
        openai_task = asyncio.create_task(self.openai.analyze_stream(data, fs))
        
        # 3. Local Analysis (CPU/GPU intensive, might block event loop if not threaded,
        # but HF pipeline is okay-ish or we can run in executor)
        self.log("Starting Local Analysis task...")
        self.local.log_callback = self.log_callback
        loop = asyncio.get_running_loop()
        local_task = loop.run_in_executor(None, self.local.predict, file_path)

        # Wait for analysis to finish (playback handles its own timing)
        self.log("Waiting for analysis results...")
        openai_result_raw, local_result = await asyncio.gather(openai_task, local_task)
        self.log("Analysis tasks completed.")
        
        if status_callback: status_callback("Parsing results...")
        # Parse OpenAI result if it's a string JSON or similar
        openai_result = {"emotion": "unknown", "confidence": 0.0}
        if openai_result_raw:
            try:
                # The prompt asks for JSON, but it might be wrapped in text.
                # Simple heuristic: find first { and last }
                start = openai_result_raw.find('{')
                end = openai_result_raw.rfind('}')
                if start != -1 and end != -1:
                    json_str = openai_result_raw[start:end+1]
                    openai_result = json.loads(json_str)
            except Exception as e:
                self.log(f"Failed to parse OpenAI JSON: {e} | Raw: {openai_result_raw}")

        if status_callback: status_callback("Ready for feedback.")
        return {
            "file": file_path,
            "openai": openai_result,
            "local": local_result
        }

    def save_feedback(self, result_data, user_pref, comments=""):
        # Log just the filename or full path? User said "title as key". 
        # Using basename is cleaner for the "key" concept.
        filename_only = os.path.basename(result_data["file"])
        
        self.logger.log_result(
            filename_only,
            result_data["openai"],
            result_data["local"],
            user_pref,
            comments
        )

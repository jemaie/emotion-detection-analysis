import sounddevice as sd
import soundfile as sf
import asyncio
import numpy as np

class AudioManager:
    """
    Handles audio file loading and playback.
    """
    def __init__(self):
        self.current_stream = None

    def load_audio(self, file_path):
        """
        Loads an audio file and returns data and samplerate.
        Supports .mp4 via moviepy and other formats via soundfile.
        """
        if file_path.lower().endswith('.mp4'):
            try:
                from moviepy import AudioFileClip
                clip = AudioFileClip(file_path)
                data = clip.to_soundarray()
                samplerate = clip.fps
                clip.close()
                return data, samplerate
            except ImportError:
                print("MoviePy not installed. Cannot process MP4.")
                raise
            except Exception as e:
                print(f"Error loading MP4: {e}")
                raise
        
        # Fallback for WAV etc.
        data, samplerate = sf.read(file_path)
        return data, samplerate

    async def play_audio(self, file_path):
        """Plays an audio file asynchronously."""
        data, fs = self.load_audio(file_path)
        
        # sounddevice.play is non-blocking, but we want to know when it finishes?
        # Actually sd.play() is fire and forget unless we use sd.wait().
        # To make it async friendly and stoppable, we might want a stream.
        
        event = asyncio.Event()

        def callback(outdata, frames, time, status):
            if status:
                print(status)
            # This is for output stream, but we are just playing a file.
            # simpler to use sd.play and sleep for duration, or sd.wait() in a thread.
            pass

        # Calculate duration
        duration = len(data) / fs
        print(f"Playing audio: {file_path} ({duration:.2f}s)")
        
        sd.play(data, fs)
        
        # We'll sleep for the duration to simulate async wait without blocking the event loop entirely
        # minimal implementation for now. ideally we use a stream and callback for precise timing/stop.
        await asyncio.sleep(duration)
        sd.stop()
        print("Audio playback finished.")

    def stop(self):
        sd.stop()

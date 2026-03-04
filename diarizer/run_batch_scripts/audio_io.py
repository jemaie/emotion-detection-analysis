import subprocess
from pathlib import Path

def normalize_to_wav16k_mono(src: Path, dst: Path) -> None:
    """
    Normalize any common audio/video input to WAV, 16kHz, mono.
    Uses ffmpeg.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

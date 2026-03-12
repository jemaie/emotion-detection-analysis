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

def chunk_wav_ffmpeg(src: Path, out_dir: Path, segment_time: int = 30) -> list[Path]:
    """
    Split a WAV file into smaller chunks (default 30 seconds) using ffmpeg.
    Returns a sorted list of the generated chunk paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # E.g. out_dir / "chunk_000.wav"
    out_pattern = out_dir / "chunk_%03d.wav"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-c", "copy",
        str(out_pattern)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Return the generated chunks sorted by name so they are in order
    return sorted(list(out_dir.glob("chunk_*.wav")))

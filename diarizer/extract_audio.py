import subprocess
from pathlib import Path
from typing import List, Dict, Any

def extract_segments_ffmpeg(
    audio_wav: Path,
    segments: List[Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    """
    Extract each segment as its own WAV file (copied, no re-encode).
    Assumes audio_wav is WAV PCM (ffmpeg will still handle correctly).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])
        out_path = out_dir / f"seg_{i:04d}_{start:.2f}_{end:.2f}.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_wav),
            "-ss", f"{start:.3f}",
            "-to", f"{end:.3f}",
            "-c", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out_paths.append(out_path)

    return out_paths

def concat_wavs_ffmpeg(segment_paths: List[Path], out_path: Path) -> None:
    """
    Concatenate WAVs using ffmpeg concat demuxer.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not segment_paths:
        return

    # Create concat list file
    list_file = out_path.parent / (out_path.stem + "_concat_list.txt")
    lines = [f"file '{p.resolve()}'" for p in segment_paths]
    list_file.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Cleanup
    try:
        list_file.unlink()
    except OSError:
        pass

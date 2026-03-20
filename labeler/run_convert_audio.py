import subprocess
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("aufnahmen")
OUT_DIR_16K = Path("data/normalized_16kHz")
OUT_DIR_24K = Path("data/normalized_24kHz")

def main():
    # Create missing directories
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR.resolve()} does not exist.")
        return
        
    OUT_DIR_16K.mkdir(parents=True, exist_ok=True)
    OUT_DIR_24K.mkdir(parents=True, exist_ok=True)
    
    # Collect media files
    files_to_process = [
        f for f in INPUT_DIR.iterdir() 
        if f.is_file() and f.suffix.lower() == ".mp4"
    ]
    
    if not files_to_process:
        print(f"No valid audio/video files found in {INPUT_DIR.resolve()}.")
        return
        
    print(f"Found {len(files_to_process)} files to process.")
    
    # Process files with a progress bar
    for file_path in tqdm(files_to_process, desc="Converting Audio"):
        filename = file_path.stem + ".wav"
        
        out_path_16k = OUT_DIR_16K / filename
        out_path_24k = OUT_DIR_24K / filename
        
        # 16kHz Conversion
        if not out_path_16k.exists():
            cmd_16k = [
                "ffmpeg", "-y",
                "-i", str(file_path),
                "-vn",                 # Disable video output
                "-ac", "1",            # Force mono
                "-ar", "16000",        # 16kHz sample rate
                "-c:a", "pcm_s16le",   # PCM 16-bit little-endian
                str(out_path_16k)
            ]
            subprocess.run(cmd_16k, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        # 24kHz Conversion
        if not out_path_24k.exists():
            cmd_24k = [
                "ffmpeg", "-y",
                "-i", str(file_path),
                "-vn",                 # Disable video output
                "-ac", "1",            # Force mono
                "-ar", "24000",        # 24kHz sample rate
                "-c:a", "pcm_s16le",   # PCM 16-bit little-endian
                str(out_path_24k)
            ]
            subprocess.run(cmd_24k, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":
    main()

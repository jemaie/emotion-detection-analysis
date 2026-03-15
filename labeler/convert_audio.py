import subprocess
from pathlib import Path
from tqdm import tqdm

def main():
    # Define directories
    base_dir = Path(__file__).parent
    input_dir = base_dir.parent / "aufnahmen"
    out_dir_16k = base_dir / "data" / "normalized_16kHz"
    out_dir_24k = base_dir / "data" / "normalized_24kHz"
    
    # Create missing directories
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir.resolve()} does not exist.")
        return
        
    out_dir_16k.mkdir(parents=True, exist_ok=True)
    out_dir_24k.mkdir(parents=True, exist_ok=True)
    
    # Collect media files
    files_to_process = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() == ".mp4"
    ]
    
    if not files_to_process:
        print(f"No valid audio/video files found in {input_dir.resolve()}.")
        return
        
    print(f"Found {len(files_to_process)} files to process.")
    
    # Process files with a progress bar
    for file_path in tqdm(files_to_process, desc="Converting Audio"):
        filename = file_path.stem + ".wav"
        
        out_path_16k = out_dir_16k / filename
        out_path_24k = out_dir_24k / filename
        
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

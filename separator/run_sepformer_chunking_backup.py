from pathlib import Path
import torchaudio
import torch
import gc
from speechbrain.inference.separation import SepformerSeparation as separator
from tqdm import tqdm

from audio_io import normalize_to_wav16k_mono

def separate_long_audio(model, audio_tensor, sample_rate, chunk_len=30.0, overlap=15.0):
    chunk_samples = int(chunk_len * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step = chunk_samples - overlap_samples
    
    T = audio_tensor.shape[1]
    
    if T <= chunk_samples:
        with torch.no_grad():
            est = model.separate_batch(audio_tensor)
            est = est / est.abs().max(dim=1, keepdim=True)[0]
            return est
            
    out_tensor = torch.zeros(2, T, device='cpu')
    fade_in = torch.linspace(0, 1, overlap_samples, device='cpu').unsqueeze(-1)
    fade_out = torch.linspace(1, 0, overlap_samples, device='cpu').unsqueeze(-1)
    
    prev_raw = None
    
    for i in range(0, T, step):
        end = min(i + chunk_samples, T)
        chunk = audio_tensor[:, i:end]
        
        with torch.no_grad():
            est_sources = model.separate_batch(chunk).detach().cpu()[0] 
        
        curr_time = est_sources.shape[0]
        
        if prev_raw is not None:
            end_prev = i - step + prev_raw.shape[0]
            ov_size = min(end_prev - i, curr_time)
            
            if ov_size > 0:
                prev_overlap = prev_raw[-ov_size:]
                curr_overlap = est_sources[:ov_size]
                
                diff_0 = torch.abs(prev_overlap - curr_overlap).mean()
                diff_1 = torch.abs(prev_overlap - curr_overlap.flip(dims=[1])).mean()
                
                if diff_1 < diff_0:
                    est_sources = est_sources.flip(dims=[1])
                    
        prev_raw = est_sources.clone()
        
        weight = torch.ones(curr_time, 1, device='cpu')
        
        if i > 0:
            ov_size = min(overlap_samples, curr_time)
            weight[:ov_size] *= fade_in[:ov_size]
            
        if end < T:
            ov_size = min(overlap_samples, curr_time)
            weight[-ov_size:] *= fade_out[-ov_size:]
            
        est_sources *= weight
        out_tensor[:, i:end] += est_sources.transpose(0, 1)
        
        del chunk
        del est_sources
        del weight
        gc.collect()
        
    out_tensor = out_tensor.transpose(0, 1).unsqueeze(0)
    out_tensor = out_tensor / out_tensor.abs().max(dim=1, keepdim=True)[0]
    
    return out_tensor

RAW_DIR = Path("../aufnahmen25")
NORM_DIR = Path("data/normalized")
CHUNKS_DIR = Path("data/chunks")
OUT_CHUNKS_DIR = Path("out_separated_chunks")
OUT_DIR = Path("out_separated")
MODEL_SOURCE = "speechbrain/sepformer-whamr16k"
SAVE_DIR = "pretrained_models/sepformer-whamr16k"

def list_input_files() -> list[Path]:
    files = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            files.append(p)
    return sorted(files)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Sepformer model from {MODEL_SOURCE}...")
    # NOTE: savedir parameter is removed to avoid symlink issues on Windows
    model = separator.from_hparams(source=MODEL_SOURCE)
    print("Model loaded successfully.")

    audio_files = list_input_files()
    if not audio_files:
        print(f"No .mp4 files found in {RAW_DIR}")
        return

    print(f"Found {len(audio_files)} files to process.")

    for src_path in tqdm(audio_files, desc="Separating audio"):
        recording_id = src_path.stem
        norm_path = NORM_DIR / f"{recording_id}.wav"
        
        # 1. Normalize and resample to 16kHz mono
        if not norm_path.exists():
            try:
                normalize_to_wav16k_mono(src_path, norm_path)
            except Exception as e:
                print(f"\nError normalizing {src_path.name}: {e}")
                continue
                
        # 2. Setup output directory for this recording
        rec_out_dir = OUT_DIR / recording_id
        rec_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_spk1 = rec_out_dir / f"{recording_id}_spk1.wav"
        out_spk2 = rec_out_dir / f"{recording_id}_spk2.wav"
        
        # Skip if already processed
        if out_spk1.exists() and out_spk2.exists():
            continue

        # 3. Process the entire normalized file with Sepformer
        try:
            sample_rate = 16000
            
            # Load the normalized file into a tensor
            batch, fs_file = torchaudio.load(str(norm_path))
            if fs_file != sample_rate:
                tf = torchaudio.transforms.Resample(orig_freq=fs_file, new_freq=sample_rate)
                batch = tf(batch.mean(dim=0, keepdim=True))
            
            # Separate overlapping chunks
            est_sources = separate_long_audio(model, batch, sample_rate, chunk_len=30.0, overlap=15.0)
            
            # Save separated tracks
            torchaudio.save(str(out_spk1), est_sources[:, :, 0], sample_rate)
            torchaudio.save(str(out_spk2), est_sources[:, :, 1], sample_rate)
            
            del batch
            del est_sources
            gc.collect()
            
        except Exception as e:
            print(f"\nError separating full file {recording_id}: {e}")

if __name__ == "__main__":
    main()

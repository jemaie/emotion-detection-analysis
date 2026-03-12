from pathlib import Path
import torchaudio
import torch
import gc
from speechbrain.inference.separation import SepformerSeparation as separator

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm

def separate_long_audio(model, audio_tensor, sample_rate, chunk_len=30.0, overlap=15.0):
    """
    Separates long audio using overlapping chunks.
    Keeps the heavy 'out_tensor' in CPU RAM to prevent GPU OOM,
    while leveraging GPU for the individual forward passes on 'chunk'.
    """
    chunk_samples = int(chunk_len * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step = chunk_samples - overlap_samples
    
    T = audio_tensor.shape[1]
    
    if T <= chunk_samples:
        with torch.no_grad():
            est = model.separate_batch(audio_tensor).detach().cpu()
            est = est / est.abs().max(dim=1, keepdim=True)[0]
            return est
            
    out_tensor = torch.zeros(2, T, device='cpu')
    fade_in = torch.linspace(0, 1, overlap_samples, device='cpu').unsqueeze(-1)
    fade_out = torch.linspace(1, 0, overlap_samples, device='cpu').unsqueeze(-1)
    
    prev_raw = None
    
    for i in range(0, T, step):
        end = min(i + chunk_samples, T)
        chunk = audio_tensor[:, i:end]
        
        # Model efficiently processes the chunk on GPU (if configured), then moves result back to CPU RAM immediately.
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    out_tensor = out_tensor.transpose(0, 1).unsqueeze(0)
    out_tensor = out_tensor / out_tensor.abs().max(dim=1, keepdim=True)[0]
    
    return out_tensor

def main():
    # Example usage for Colab, similar to the notebook cells:
    NORM_DIR = Path("data/normalized") # Change to /content/normalized in colab
    OUT_DIR = Path("out_separated")    # Change to /content/out_separated in colab
    MODEL_SOURCE = "speechbrain/sepformer-whamr16k"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Sepformer model from {MODEL_SOURCE}...")
    run_opts = {"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
    model = separator.from_hparams(source=MODEL_SOURCE, run_opts=run_opts)
    print(f"Model loaded successfully on {run_opts['device']}.")

    audio_files = sorted([p for p in NORM_DIR.glob("*.wav") if p.is_file()])
    if not audio_files:
        print(f"No .wav files found in {NORM_DIR}")
        return
    print(f"Found {len(audio_files)} files to process.")

    pbar = tqdm(audio_files, desc="Separating audio")
    for norm_path in pbar:
        recording_id = norm_path.stem
        pbar.set_postfix(file=recording_id)
        
        rec_out_dir = OUT_DIR / recording_id
        rec_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_spk1 = rec_out_dir / f"{recording_id}_spk1.wav"
        out_spk2 = rec_out_dir / f"{recording_id}_spk2.wav"
        
        if out_spk1.exists() and out_spk2.exists():
            continue

        try:
            sample_rate = 16000
            
            # 1. Load the normalized file into a tensor
            batch, fs_file = torchaudio.load(str(norm_path))
            if fs_file != sample_rate:
                tf = torchaudio.transforms.Resample(orig_freq=fs_file, new_freq=sample_rate)
                batch = tf(batch.mean(dim=0, keepdim=True))
            
            # Ensure GPU memory is clean before processing a new file
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 2. Separate overlapping chunks (avoids OOM on long files)
            est_sources = separate_long_audio(model, batch, sample_rate, chunk_len=30.0, overlap=15.0)
            
            # 3. Save separated tracks
            torchaudio.save(str(out_spk1), est_sources[:, :, 0], sample_rate)
            torchaudio.save(str(out_spk2), est_sources[:, :, 1], sample_rate)
            
            del batch
            del est_sources
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError separating full file {recording_id}: {e}")

if __name__ == "__main__":
    main()

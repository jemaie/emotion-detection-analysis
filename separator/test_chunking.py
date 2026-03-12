import torch
import gc

def separate_long_audio(model, audio_tensor, sample_rate, chunk_len=30, overlap=15):
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

class MockModel:
    def separate_batch(self, mix):
        T = mix.shape[1]
        out = torch.zeros(1, T, 2)
        out[0, :, 0] = mix[0] * 1.0
        out[0, :, 1] = mix[0] * -1.0
        # random swap
        if torch.rand(1).item() > 0.5:
            out = out.flip(dims=[2])
        return out

if __name__ == "__main__":
    model = MockModel()
    sr = 16000
    # 70 seconds audio
    audio = torch.randn(1, 70 * sr)
    out = separate_long_audio(model, audio, sr, chunk_len=30, overlap=15)
    print("Output shape:", out.shape)
    
    # Verify reconstruction roughly
    # Both sources should be perfectly reconstructed if chunking worked
    # source 0 is +audio, source 1 is -audio (ignoring random swap which is righted by the code)
    # wait, due to max norm they might be scaled
    # Let's check max norm
    print("Max out:", out.abs().max())
    print("Sum of MSE against expected:")
    expected_0 = audio[0] / audio[0].abs().max()
    expected_1 = -audio[0] / audio[0].abs().max()
    
    # it might swap the final output whole track
    mse1 = torch.nn.functional.mse_loss(out[0, :, 0], expected_0) + torch.nn.functional.mse_loss(out[0, :, 1], expected_1)
    mse2 = torch.nn.functional.mse_loss(out[0, :, 0], expected_1) + torch.nn.functional.mse_loss(out[0, :, 1], expected_0)
    print("MSE:", min(mse1.item(), mse2.item()))

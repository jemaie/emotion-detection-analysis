import pandas as pd
import json
from pathlib import Path

INTENSITY_HIERARCHY = {
    'unusable': 0, 'other': 0, 'neutral': 0, 'calm': 0, 'happy': 0, 'uncertain': 0,
    'curious': 1, 'surprised': 1,
    'confused': 2, 'anxious': 2, 'fearful': 2, 'frustrated': 2,
    'angry': 3
}

def clean_emotion(e):
    if pd.isna(e): return None
    return str(e).lower().strip()

def resolve_consensus(emotions):
    emotions = [e for e in emotions if pd.notna(e)]
    if not emotions: return None
    counts = pd.Series(emotions).value_counts()
    candidates = counts[counts == counts.max()].index.tolist()
    if len(candidates) == 1: return candidates[0]
    best_candidate = candidates[0]
    max_intensity = -1
    for c in candidates:
        intensity = INTENSITY_HIERARCHY.get(c, 0)
        if intensity > max_intensity:
            max_intensity = intensity
            best_candidate = c
    return best_candidate

def parse_time(t_str):
    if t_str == "-" or pd.isna(t_str): return None
    parts = t_str.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return None

def parse_segment_midpoint(segment_name):
    # e.g., seg_0000_0.01_3.29
    parts = segment_name.split('_')
    if len(parts) >= 4:
        try:
            start = float(parts[2])
            end = float(parts[3])
            return (start + end) / 2.0
        except ValueError:
            pass
    return 0.0

def evaluate_phase(emotions):
    if not emotions: return None, {}, None
    
    # Approach A: Peak Emotion
    peak_emotion = emotions[0]
    max_intensity = -1
    for e in emotions:
        intensity = INTENSITY_HIERARCHY.get(e, 0)
        if intensity > max_intensity:
            max_intensity = intensity
            peak_emotion = e
            
    # Approach B: Distribution
    dist = pd.Series(emotions).value_counts(normalize=True).to_dict()
    
    # Approach C: Trajectory
    start = emotions[0]
    end = emotions[-1]
    traj = f"{start} -> {peak_emotion} -> {end}"
    
    return peak_emotion, dist, traj

def analyze_phases():
    print("=== Phase 3: Phase-Level Analysis ===\n")
    
    target_models = [
        "openai_realtime_1_5_ft", "openai_realtime_1_5_ft_2",
        "openai_realtime_1_5_ft_e", "openai_realtime_1_5_ft_e_2",
        "openai_realtime_1_5_ft_erp", "openai_realtime_1_5_ft_erp_2"
    ]
    
    phases_path = Path("output/phases_analysis.json")
    if not phases_path.exists():
        print("Phase file not found.")
        return
        
    with open(phases_path, 'r', encoding='utf-8') as f:
        phases_data = json.load(f)
        
    csv_path = Path("output/ratings_segments.csv")
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    df['emotion'] = df['emotion'].apply(clean_emotion)
    df = df.dropna(subset=['emotion'])
    
    consensus_df = df.groupby(['conv_id', 'segment'])['emotion'].apply(lambda x: resolve_consensus(x.tolist())).reset_index()
    consensus_df.rename(columns={'emotion': 'human_consensus'}, inplace=True)
    consensus_df['segment_stem'] = consensus_df['segment'].apply(lambda x: Path(x).stem)
    consensus_df['midpoint'] = consensus_df['segment_stem'].apply(parse_segment_midpoint)
    consensus_df.sort_values(by=['conv_id', 'midpoint'], inplace=True)

    segments_dir = Path("output/caller_segments")
    model_predictions = []
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment_stem = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            preds = data.get("predictions", {})
            row = {"conv_id": conv_id, "segment_stem": segment_stem}
            has_data = False
            for t_model in target_models:
                pred_obj = preds.get(t_model)
                if pred_obj:
                    val = pred_obj.get("emotion", "unknown") if isinstance(pred_obj, dict) else pred_obj
                    row[t_model] = str(val).lower().strip()
                    has_data = True
            if has_data:
                model_predictions.append(row)
                
    model_df = pd.DataFrame(model_predictions)
    if 'segment_stem' in model_df:
        model_df['midpoint'] = model_df['segment_stem'].apply(parse_segment_midpoint)
        model_df.sort_values(by=['conv_id', 'midpoint'], inplace=True)
    else:
        print("No predictions found to parse.")
        return
    
    merged_segments = pd.merge(consensus_df, model_df, on=['conv_id', 'segment_stem', 'midpoint'], how='inner')
    print(f"Matched {len(merged_segments)} segments with both Human Consensus and Model Predictions\n")
    
    import tabulate
    for t_model in target_models:
        if t_model not in merged_segments.columns:
            continue
            
        results = []
        for conv_file, conv_phases in phases_data.items():
            conv_id = conv_file.replace(".wav", "")
            
            conv_segments = merged_segments[merged_segments['conv_id'] == conv_id]
            if len(conv_segments) == 0:
                continue
            
            for phase in conv_phases["phases"]:
                start_s = parse_time(phase["start_time"])
                end_s = parse_time(phase["end_time"])
                if start_s is None or end_s is None:
                    continue
                    
                phase_segs = conv_segments[(conv_segments['midpoint'] >= start_s) & (conv_segments['midpoint'] <= end_s)]
                if len(phase_segs) == 0:
                    continue
                    
                human_emotions = phase_segs['human_consensus'].tolist()
                model_emotions = phase_segs[t_model].dropna().tolist()
                if not model_emotions:
                    continue
                
                h_peak, h_dist, h_traj = evaluate_phase(human_emotions)
                m_peak, m_dist, m_traj = evaluate_phase(model_emotions)
                
                results.append({
                    "Conv": conv_id[:20] + "...",
                    "Phase": phase["phase_name"],
                    "Segments": len(human_emotions),
                    "H-Peak": h_peak,
                    "M-Peak": m_peak,
                    "H-Trajectory": h_traj,
                    "M-Trajectory": m_traj
                })
                
        res_df = pd.DataFrame(results)
        if len(res_df) == 0:
            print(f"No matches between segments and phase timings for {t_model}.")
            continue
            
        print(f"\n--- Model ({t_model}) ---")
        print("Sample Phase-Level Interpretations:")
        print(res_df.head(20).to_markdown(index=False))
        
        matches = (res_df['H-Peak'] == res_df['M-Peak']).sum()
        print(f"\nPeak Emotion Mapping Match (Approach A): {matches} / {len(res_df)} phases ({matches/len(res_df)*100:.2f}%)")
        
        traj_matches = (res_df['H-Trajectory'] == res_df['M-Trajectory']).sum()
        print(f"Full Trajectory Match (Approach C): {traj_matches} / {len(res_df)} phases ({traj_matches/len(res_df)*100:.2f}%)")

if __name__ == "__main__":
    analyze_phases()

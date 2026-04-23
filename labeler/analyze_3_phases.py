import os
import json
import argparse
import pandas as pd
from pathlib import Path

def clean_emotion(e):
    if e is None or pd.isna(e): return "unknown"
    e = str(e).lower().strip()
    mapping = {
        'hap': 'happy', 'neu': 'neutral', 'ang': 'angry', 
        'sad': 'sad', 'dis': 'disgust', 'fea': 'fearful', 
        'sur': 'surprised', 'cal': 'calm'
    }
    return mapping.get(e, e)

def analyze_phases(args):
    print("=== Phase 3: Phase-Level Analysis (From Pre-Calculated JSONs) ===\n")
    
    target_models = [
        "openai_realtime", "openai_realtime_2",
        "openai_realtime_rp", "openai_realtime_rp_2",
        "openai_realtime_ft", "openai_realtime_ft_2",
        "openai_realtime_1_5_ft", "openai_realtime_1_5_ft_2",
        "openai_realtime_1_5_ft_e", "openai_realtime_1_5_ft_e_2",
        "openai_realtime_1_5_ft_erp", "openai_realtime_1_5_ft_erp_2",
        "openai_realtime_1_5_ft_rp", "openai_realtime_1_5_ft_rp_2",
        "openai_realtime_1_5_new_taxonomy", "openai_realtime_1_5_new_taxonomy_2",
        "openai_realtime_1_5_ft_ns", "openai_realtime_1_5_ft_ns_2",
        "openai_realtime_1_5_ft_rp_ns", "openai_realtime_1_5_ft_rp_ns_2",
        "ehcalabres/wav2vec2", "superb/hubert_large", "iic/emotion2vec_large"
    ]
    
    phases_dir = Path(args.phases_dir)
    if not phases_dir.exists():
        print(f"Directory {phases_dir} not found. Run aggregate_seg_ratings_to_phase_ratings.py first.")
        return

    all_results = []

    for model in target_models:
        results = []
        for json_file in phases_dir.rglob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preds = data.get("predictions", {})
            
            human = preds.get("human_consensus")
            if not human:
                continue
                
            model_pred = preds.get(model)
            if not model_pred:
                continue
                
            # Extract Human Metrics
            h_peak = human.get("seg_agg_peak", "unknown")
            h_maj = human.get("seg_agg_maj_tiebroken", "unknown")
            h_traj_start = human.get("seg_agg_traj_start", "unknown")
            h_traj_end = human.get("seg_agg_traj_end", "unknown")
            h_traj = f"{h_traj_start} -> {h_peak} -> {h_traj_end}"
            
            # Format human distribution as string
            h_dist_raw = human.get("seg_agg_dist", {})
            h_dist = ", ".join([f"{k}: {v:.0%}" for k, v in h_dist_raw.items()])
            
            # Extract Model Metrics
            m_peak = model_pred.get("seg_agg_peak", "unknown")
            m_maj = model_pred.get("seg_agg_maj_tiebroken", "unknown")
            m_traj_start = model_pred.get("seg_agg_traj_start", "unknown")
            m_traj_end = model_pred.get("seg_agg_traj_end", "unknown")
            m_traj = f"{m_traj_start} -> {m_peak} -> {m_traj_end}"
            
            m_dist_raw = model_pred.get("seg_agg_dist", {})
            m_dist = ", ".join([f"{k}: {v:.0%}" for k, v in m_dist_raw.items()])
            
            # Direct Model Rating
            m_direct_raw = model_pred.get("emotion") if isinstance(model_pred, dict) else model_pred
            m_direct = clean_emotion(m_direct_raw)
            
            # Approach B Metric: Histogram Intersection
            all_keys = set(h_dist_raw.keys()).union(set(m_dist_raw.keys()))
            hist_intersection = sum(min(h_dist_raw.get(k, 0), m_dist_raw.get(k, 0)) for k in all_keys)
            
            conv_id = json_file.parent.name.replace("conv__", "")
            phase_filename = json_file.name
            
            results.append({
                "Model": model,
                "Conv": conv_id,
                "Phase_File": phase_filename,
                "H-Peak": h_peak,
                "M-Peak": m_peak,
                "H-Majority": h_maj,
                "M-Majority": m_maj,
                "M-Direct": m_direct,
                "H-Trajectory": h_traj,
                "M-Trajectory": m_traj,
                "H-Distribution": h_dist,
                "M-Distribution": m_dist,
                "Dist_Overlap": hist_intersection
            })
            
        all_results.extend(results)
        res_df = pd.DataFrame(results)
        if len(res_df) == 0:
            print(f"\n--- Model ({model}) ---")
            print("No matches found.")
            continue
            
        print(f"\n--- Model ({model}) ---")
        
        dir_matches = (res_df['H-Peak'] == res_df['M-Direct']).sum()
        print(f"Direct Phase Rating Match: {dir_matches} / {len(res_df)} phases ({dir_matches/len(res_df)*100:.2f}%)")
        
        matches = (res_df['H-Peak'] == res_df['M-Peak']).sum()
        print(f"Peak Emotion Match (Approach A): {matches} / {len(res_df)} phases ({matches/len(res_df)*100:.2f}%)")
        
        maj_matches = (res_df['H-Majority'] == res_df['M-Majority']).sum()
        print(f"Phase Majority Match (Approach B): {maj_matches} / {len(res_df)} phases ({maj_matches/len(res_df)*100:.2f}%)")
        
        avg_overlap = res_df['Dist_Overlap'].mean() * 100
        print(f"Distribution Overlap (Approach C): {avg_overlap:.2f}%")
        
        traj_matches = (res_df['H-Trajectory'] == res_df['M-Trajectory']).sum()
        print(f"Full Trajectory Match (Approach D): {traj_matches} / {len(res_df)} phases ({traj_matches/len(res_df)*100:.2f}%)")

    # Save to CSV
    combined_df = pd.DataFrame(all_results)
    if not combined_df.empty:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "phase_level_metrics.csv"
        combined_df.to_csv(csv_path, index=False, sep=";")
        print(f"\nSaved combined phase metrics to {csv_path}")
    else:
        print("\nNo data generated to save.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output/analysis_results')
    parser.add_argument('--phases_dir', default='output/caller_phases')
    args = parser.parse_args()
    
    os.makedirs(Path(args.output_dir), exist_ok=True)

    analyze_phases(args)

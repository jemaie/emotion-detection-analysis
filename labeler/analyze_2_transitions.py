import argparse
import os
import json
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')



def parse_duration(segment_str):
    parts = str(segment_str).split('_')
    if len(parts) >= 4:
        try:
            return float(parts[3]) - float(parts[2])
        except ValueError:
            pass
    return 0.0

def extract_transition_data(matrix, name, tier, bucket):
    if not matrix:
        return pd.DataFrame()
        
    states = set(matrix.keys())
    for v in matrix.values(): states.update(v.keys())
    states = sorted(list(states))
    
    df = pd.DataFrame(index=states, columns=states).fillna(0).astype(int)
    for s1, targets in matrix.items():
        for s2, count in targets.items():
            df.loc[s1, s2] = count

    df_prob = df.div(df.sum(axis=1), axis=0).fillna(0)
    
    records = []
    for s1 in states:
        for s2 in states:
            records.append({
                'Data Tier': tier,
                'Duration Bucket': "All" if bucket == 0 else f">{bucket}s",
                'Model': name,
                'From_Emotion': s1,
                'To_Emotion': s2,
                'Count': df.loc[s1, s2],
                'Probability': df_prob.loc[s1, s2]
            })
    return pd.DataFrame(records)

def analyze_transitions_deep(args):
    print("=== Phase 2: Deep Trajectory and Transition Analysis ===\n")
    
    target_models = set(["human_consensus"])
    segments_dir = Path(args.segments_dir)
    model_predictions = []
    
    exclude_models = set()
    if hasattr(args, 'exclude_models') and args.exclude_models:
        exclude_models = set(args.exclude_models.split(','))
    
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment = json_file.stem
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            preds = data.get("predictions", {})
                    
            clean_preds = {
                'conv_id': conv_id, 
                'segment': segment,
                'duration': parse_duration(segment)
            }
            
            has_unusable = False
            for model_name, pred_obj in preds.items():
                if not model_name.startswith("human_") and model_name not in exclude_models:
                    target_models.add(model_name)
                    
                if isinstance(pred_obj, dict):
                    if model_name.startswith("human_") and model_name != "human_consensus":
                        human_emotion = pred_obj.get("extracted_normalized_emotion", pred_obj.get("emotion"))
                        if str(human_emotion).lower().strip() == "unusable":
                            has_unusable = True
                            
                    elif model_name == "human_consensus":
                        clean_preds["human_consensus"] = str(pred_obj.get("maj_tiebroken", "unknown")).lower().strip()
                        clean_preds["pairwise_agreement"] = float(pred_obj.get("pairwise_agreement", 0.0))
                    else:
                        clean_preds[model_name] = str(pred_obj.get("emotion", "unknown")).lower().strip()
                else:
                    clean_preds[model_name] = str(pred_obj).lower().strip()
                    
            clean_preds['has_unusable'] = has_unusable
            model_predictions.append(clean_preds)
            
    if not model_predictions:
        print("No model segments found.")
        return
        
    merged_df = pd.DataFrame(model_predictions)
    merged_df.sort_values(by=['conv_id', 'segment'], inplace=True)
    
    all_dfs = []
    
    tiers = [
        ("All Segments (Unfiltered)", lambda row: True),
        ("Valid Consensus", lambda row: (row['has_unusable'] == False) and (row['pairwise_agreement'] > 0.0))
    ]
    
    buckets = [0, 1, 2, 3]
    
    for tier_name, tier_condition in tiers:
        for bucket in buckets:
            
            def is_valid(r):
                return tier_condition(r) and r['duration'] > bucket
                
            model_matrices = {m: {} for m in target_models}
            
            for conv_id, group in merged_df.groupby('conv_id'):
                n_segs = len(group)
                for i in range(n_segs - 1):
                    row_curr = group.iloc[i]
                    row_next = group.iloc[i+1]
                    
                    if is_valid(row_curr) and is_valid(row_next):
                        for m in target_models:
                            if m in row_curr and m in row_next:
                                state_curr = row_curr[m]
                                state_next = row_next[m]
                                if pd.isna(state_curr) or pd.isna(state_next) or state_curr == 'unknown' or state_next == 'unknown':
                                    continue
                                
                                if state_curr not in model_matrices[m]:
                                    model_matrices[m][state_curr] = {}
                                model_matrices[m][state_curr][state_next] = model_matrices[m][state_curr].get(state_next, 0) + 1
                                
            for m in target_models:
                m_display = "Human Consensus" if m == "human_consensus" else f"Model ({m})"
                df_res = extract_transition_data(model_matrices[m], m_display, tier_name, bucket)
                if not df_res.empty:
                    all_dfs.append(df_res)
                    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        results_dir = Path(args.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_df = combined_df[(combined_df['Data Tier'] == 'All Segments (Unfiltered)') & (combined_df['Duration Bucket'] == 'All')]
        baseline_csv = results_dir / "transitions_baseline.csv"
        baseline_df.drop(columns=['Data Tier', 'Duration Bucket'], errors='ignore').to_csv(baseline_csv, index=False)
        
        deep_csv = results_dir / "transitions_deep.csv"
        combined_df.to_csv(deep_csv, index=False)
        
        print(f"\nExported baseline transition metrics to {baseline_csv}")
        print(f"Exported deep transition metrics to {deep_csv}")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--segments_dir', default='output/caller_segments')
    parser.add_argument('--output_dir', default='output/analysis_results')
    parser.add_argument('--exclude_models', default=None, help='Comma-separated list of model config names to exclude from analysis')
    args = parser.parse_args()
    
    os.makedirs(Path(args.output_dir), exist_ok=True)

    analyze_transitions_deep(args)

import pandas as pd
import json
from pathlib import Path
import numpy as np
import krippendorff
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

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
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Tie-breaker: pick the one with highest intensity
    best_candidate = candidates[0]
    max_intensity = -1
    for c in candidates:
        intensity = INTENSITY_HIERARCHY.get(c, 0)
        if intensity > max_intensity:
            max_intensity = intensity
            best_candidate = c
    return best_candidate

def calculate_krippendorff(df):
    unique_emotions = df['emotion'].unique().tolist()
    emotion_to_id = {e: i for i, e in enumerate(unique_emotions)}
    
    pivot = df.pivot_table(index=['conv_id', 'segment'], columns='rater', values='emotion', aggfunc='first')
    
    data = pivot.replace(emotion_to_id).T.values
    return krippendorff.alpha(reliability_data=data, level_of_measurement='nominal')

def analyze_segment_metrics():
    print("=== Phase 1: Segment-Level Analysis ===\n")
    
    csv_path = Path("output/ratings_segments.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    df['emotion'] = df['emotion'].apply(clean_emotion)
    df = df.dropna(subset=['emotion'])
    
    # 1. Krippendorff's alpha (IRR)
    alpha = calculate_krippendorff(df)
    print(f"Human Raters Inter-rater Reliability (Krippendorff's Alpha): {alpha:.3f}\n")
    
    # Group to find consensus
    consensus_df = df.groupby(['conv_id', 'segment'])['emotion'].apply(lambda x: resolve_consensus(x.tolist())).reset_index()
    consensus_df.rename(columns={'emotion': 'human_consensus'}, inplace=True)
    # Remove extension from segment for merging
    consensus_df['segment'] = consensus_df['segment'].apply(lambda x: Path(x).stem)
    
    print(f"Processed {len(consensus_df)} segments to establish Human Consensus.\n")
    
    # 2. Match with model JSON files
    segments_dir = Path("output/caller_segments")
    model_predictions = []
    
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment = json_file.stem
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            preds = data.get("predictions", {})
            
            clean_preds = {}
            for model_name, pred_obj in preds.items():
                if isinstance(pred_obj, dict):
                    clean_preds[model_name] = str(pred_obj.get("emotion", "unknown"))
                else:
                    clean_preds[model_name] = str(pred_obj)
                    
            clean_preds['conv_id'] = conv_id
            clean_preds['segment'] = segment
            model_predictions.append(clean_preds)
            
    if not model_predictions:
        print("No model segments found.")
        return
        
    models_df = pd.DataFrame(model_predictions)
    merged_df = pd.merge(consensus_df, models_df, on=['conv_id', 'segment'], how='inner')
    
    # Evaluate each model configuration against human_consensus
    metrics = []
    models_to_test = [c for c in merged_df.columns if c not in ['conv_id', 'segment', 'human_consensus']]
    
    for model in models_to_test:
        valid_data = merged_df.dropna(subset=['human_consensus', model])
        if len(valid_data) == 0:
            continue
            
        y_true = valid_data['human_consensus']
        y_pred = valid_data[model].str.lower().str.strip()
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        metrics.append({
            'Model': model,
            'Segments Evaluated': len(valid_data),
            'Accuracy': acc,
            'F1 (Macro)': f1_macro,
            'F1 (Weighted)': f1_weighted
        })
        
    res_df = pd.DataFrame(metrics).sort_values('F1 (Weighted)', ascending=False)
    
    import tabulate
    print("Performance vs. Human Consensus:")
    print(res_df.to_markdown(index=False, floatfmt=".3f"))

if __name__ == "__main__":
    analyze_segment_metrics()

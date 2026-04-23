import argparse
import os
import json
import warnings
import pandas as pd
import krippendorff
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

warnings.filterwarnings('ignore')

def clean_emotion(e):
    if pd.isna(e): return None
    return str(e).lower().strip()

def parse_duration(segment_str):
    parts = str(segment_str).split('_')
    if len(parts) >= 4:
        try:
            return float(parts[3]) - float(parts[2])
        except ValueError:
            pass
    return 0.0

def calculate_krippendorff(df):
    unique_emotions = df['emotion'].unique().tolist()
    emotion_to_id = {e: i for i, e in enumerate(unique_emotions)}
    
    pivot = df.pivot_table(index=['conv_id', 'segment'], columns='rater', values='emotion', aggfunc='first')
    
    data = pivot.replace(emotion_to_id).T.values
    return krippendorff.alpha(reliability_data=data, level_of_measurement='nominal')

def analyze_segment_metrics(args):
    print("=== Phase 1: Segment-Level Analysis ===\n")
    
    segments_dir = Path(args.segments_dir)
    if not segments_dir.exists():
        print(f"Error: {segments_dir} not found.")
        return
        
    model_predictions = []
    human_irr_records = []
    
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment = json_file.stem
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            preds = data.get("predictions", {})
            
            clean_preds = {}
            has_unusable = False
            for model_name, pred_obj in preds.items():
                if isinstance(pred_obj, dict):
                    # For humans, capture IRR data using extracted_normalized_emotion if available
                    if model_name.startswith("human_") and model_name != "human_consensus":
                        human_emotion = pred_obj.get("extracted_normalized_emotion", pred_obj.get("emotion"))
                        human_emotion = str(human_emotion).lower().strip()
                        
                        if human_emotion == "unusable":
                            has_unusable = True
                            
                        # Append to IRR dataset
                        if human_emotion and human_emotion != "none":
                            human_irr_records.append({
                                "conv_id": conv_id,
                                "segment": segment,
                                "rater": model_name.replace("human_", ""),
                                "emotion": clean_emotion(human_emotion)
                            })
                            
                    if model_name == "human_consensus":
                        clean_preds["human_maj_tiebroken"] = str(pred_obj.get("maj_tiebroken", "unknown")).lower().strip()
                        clean_preds["pairwise_agreement"] = float(pred_obj.get("pairwise_agreement", 0.0))
                    else:
                        clean_preds[model_name] = str(pred_obj.get("emotion", "unknown")).lower().strip()
                else:
                    clean_preds[model_name] = str(pred_obj).lower().strip()
                    
            clean_preds['conv_id'] = conv_id
            clean_preds['segment'] = segment
            clean_preds['duration'] = parse_duration(segment)
            clean_preds['has_unusable'] = has_unusable
            model_predictions.append(clean_preds)
            
    if not model_predictions:
        print("No model segments found.")
        return
        
    # 1. Krippendorff's alpha (IRR) from JSON data
    irr_df = pd.DataFrame(human_irr_records)
    irr_df = irr_df.dropna(subset=['emotion'])
    if not irr_df.empty:
        alpha = calculate_krippendorff(irr_df)
        print(f"Human Raters Inter-rater Reliability (Krippendorff's Alpha): {alpha:.3f}\n")
    else:
        print("Human Raters Inter-rater Reliability (Krippendorff's Alpha): N/A (No human data)\n")
        
    merged_df = pd.DataFrame(model_predictions)
    
    # Filter for strict validity
    # 1. Exclude if ANY human flagged as unusable
    # 2. Exclude if pairwise_agreement is 0.0 (meaning all 4 raters completely disagreed)
    valid_df = merged_df[(merged_df['has_unusable'] == False) & (merged_df['pairwise_agreement'] > 0.0)]
    
    def evaluate_models(df_subset, ground_truth_col):
        metrics_list = []
        models_to_test = [c for c in df_subset.columns if not c.startswith("human_") and c not in ['conv_id', 'segment', 'duration', 'pairwise_agreement', 'has_unusable']]
        for model in models_to_test:
            valid_data = df_subset.dropna(subset=[ground_truth_col, model])
                
            if len(valid_data) == 0: continue
            y_true = valid_data[ground_truth_col]
            y_pred = valid_data[model].str.lower().str.strip()
            
            acc = accuracy_score(y_true, y_pred)
            # handle warnings gracefully if model predicted only 1 class or missing classes
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1_macro = f1_score(y_true, y_pred, average='macro')
                f1_weighted = f1_score(y_true, y_pred, average='weighted')
            
            metrics_list.append({
                'Model': model,
                'Segments Evaluated': len(valid_data),
                'Accuracy': acc,
                'F1 (Macro)': f1_macro,
                'F1 (Weighted)': f1_weighted
            })
        if metrics_list:
            return pd.DataFrame(metrics_list).sort_values('F1 (Weighted)', ascending=False)
        return pd.DataFrame()

    all_results = []
    
    # Loop over duration buckets
    for bucket in [0, 1, 2, 3]:
        # Unfiltered Metric (All Segments)
        df_bucket_all = merged_df[merged_df['duration'] > bucket]
        res_all = evaluate_models(df_bucket_all, "human_maj_tiebroken")
        if not res_all.empty:
            res_all.insert(0, "Duration Bucket", "All" if bucket == 0 else f">{bucket}s")
            res_all.insert(0, "Data Tier", "All Segments (Unfiltered)")
            all_results.append(res_all)
            
        # Tiebroken Valid Metric
        df_bucket_valid = valid_df[valid_df['duration'] > bucket]
        res_valid = evaluate_models(df_bucket_valid, "human_maj_tiebroken")
        if not res_valid.empty:
            res_valid.insert(0, "Duration Bucket", "All" if bucket == 0 else f">{bucket}s")
            res_valid.insert(0, "Data Tier", "Valid Consensus")
            all_results.append(res_valid)
            
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Print the baseline (All segments) for console readout
    baseline_df = combined_df[combined_df['Duration Bucket'] == 'All']

    import tabulate
    print("--- Performance vs. Valid Human Consensus (All Segments) ---")
    print(f"(Filtered out {len(merged_df) - len(valid_df)} invalid/noisy/totally fractured segments)")
    print(baseline_df.drop(columns=['Data Tier', 'Duration Bucket']).to_markdown(index=False, floatfmt=".3f"))
    
    # Export deep results
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline
    baseline_csv = results_dir / "model_performance_segments_baseline.csv"
    baseline_df.to_csv(baseline_csv, index=False, float_format="%.3f")
    
    # Deep run
    deep_csv = results_dir / "model_performance_segments_deep.csv"
    combined_df.to_csv(deep_csv, index=False, float_format="%.3f")
    print(f"\nExported baseline metrics to {baseline_csv}")
    print(f"Exported deep metrics (duration bucketing) to {deep_csv}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--segments_dir', default='output/caller_segments')
    parser.add_argument('--output_dir', default='output/analysis_results')
    args = parser.parse_args()
    
    os.makedirs(Path(args.output_dir), exist_ok=True)

    analyze_segment_metrics(args)

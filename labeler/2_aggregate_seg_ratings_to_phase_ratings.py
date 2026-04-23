import pandas as pd
import json
from pathlib import Path
import argparse

from scripts_core.segmentation_utils import parse_time, parse_segment_midpoint

INTENSITY_HIERARCHY = {
    # Original 16 labels
    "angry": 4, "frustrated": 3, "annoyed": 2,
    "sad": 3, "concerned": 2, "anxious": 3,
    "happy": 3, "grateful": 2, "relieved": 2,
    "fearful": 4, "surprised": 3, "confused": 2,
    "disgusted": 4, "disgust": 4,
    "curious": 1, "calm": 1, "neutral": 0,
    "unusable": -1, "uncertain": -1,
    
    # Ekman & Plutchik additions
    "anger": 4, "fear": 4, "happiness": 3, "sadness": 3, "surprise": 3, 
    "joy": 3, "trust": 2, "anticipation": 2, 
    
    # Willcox additions
    "mad": 4, "scared": 4, "joyful": 3, "powerful": 3, "peaceful": 1,
    
    # Russell additions (High/Low Arousal Positive/Negative Valence)
    "hanv": 4, "lanv": 2, "hapv": 3, "lapv": 1,
    
    # Catch-all
    "unknown": -1
}

def clean_emotion(e):
    if pd.isna(e): return None
    return str(e).lower().strip()

def resolve_consensus(emotions):
    emotions = [e for e in emotions if pd.notna(e) and e != 'tie']
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

def get_majority(emotions):
    emotions = [e for e in emotions if pd.notna(e) and e != 'tie']
    if not emotions: return None
    counts = pd.Series(emotions).value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    if len(candidates) == 1: return candidates[0]
    return 'tie'

def get_peak_emotion(emotions):
    emotions = [e for e in emotions if pd.notna(e) and e != 'tie']
    if not emotions: return None
    peak_emotion = emotions[0]
    max_intensity = -1
    for e in emotions:
        intensity = INTENSITY_HIERARCHY.get(e, 0)
        if intensity > max_intensity:
            max_intensity = intensity
            peak_emotion = e
    return peak_emotion

def reorder_predictions(phase_eval):
    if "predictions" not in phase_eval: return
    preds = phase_eval["predictions"]
    # We want keys starting with 'human' to be first.
    human_keys = sorted([k for k in preds.keys() if k.startswith("human")])
    other_keys = sorted([k for k in preds.keys() if not k.startswith("human")])
    
    new_preds = {}
    for k in human_keys:
        new_preds[k] = preds[k]
    for k in other_keys:
        new_preds[k] = preds[k]
        
    phase_eval["predictions"] = new_preds

def process_phases(phases_data, segments_dir, output_phases_dir):
    if not segments_dir.exists():
        print("Caller segments directory not found at", segments_dir)
        return
        
    for conv_file, conv_phases in phases_data.items():
        conv_id = conv_file.replace(".wav", "")
        seg_dir = segments_dir / conv_id
        if not seg_dir.exists():
            continue
            
        seg_records = []
        for json_file in seg_dir.rglob("*.json"):
            midpoint = parse_segment_midpoint(json_file.stem)
            with open(json_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
                
            for predictor_name, pred in eval_data.get("predictions", {}).items():
                if predictor_name == "human_consensus":
                    continue # Skip segment-level human consensus, we calculate phase-level later
                
                if isinstance(pred, dict):
                    # Prioritize extracted_normalized_emotion over raw emotion!
                    emotion = pred.get("extracted_normalized_emotion", pred.get("emotion"))
                else:
                    emotion = pred
                    
                emotion = clean_emotion(emotion)
                if emotion and "error" not in emotion:
                    seg_records.append({"midpoint": midpoint, "predictor": predictor_name, "emotion": emotion})
                    
        if not seg_records:
            continue
            
        seg_df = pd.DataFrame(seg_records)
        
        for idx, phase in enumerate(conv_phases.get("phases", [])):
            start_s = parse_time(phase.get("start_time"))
            end_s = parse_time(phase.get("end_time"))
            
            if start_s is None or end_s is None:
                continue
                
            phase_segs = seg_df[(seg_df['midpoint'] >= start_s) & (seg_df['midpoint'] <= end_s)]
            if phase_segs.empty:
                continue
                
            phase_name_raw = phase.get("phase_name", f"phase_{idx}")
            numeric_id = phase_name_raw.split(".")[0].strip() if "." in phase_name_raw else "0"
            phase_json_name = f"phase_{idx+1}_{numeric_id}_{start_s:.2f}_{end_s:.2f}.json"
            
            orig_json_path = output_phases_dir / conv_id / phase_json_name
            if not orig_json_path.exists():
                orig_json_path = Path("output/caller_phases") / conv_id / phase_json_name
            
            if orig_json_path.exists():
                with open(orig_json_path, 'r', encoding='utf-8') as f:
                    phase_eval = json.load(f)
            else:
                phase_eval = {"filename": phase_json_name, "predictions": {}}
                
            if "predictions" not in phase_eval:
                phase_eval["predictions"] = {}
                
            predictors = phase_segs['predictor'].unique()
            updates = 0
            
            phase_rater_results = []
            
            for predictor in predictors:
                pred_df = phase_segs[phase_segs['predictor'] == predictor].sort_values(by='midpoint')
                emotions = pred_df['emotion'].tolist()
                if not emotions:
                    continue
                
                emotion_peak = get_peak_emotion(emotions)
                emotion_majority = get_majority(emotions)
                emotion_majority_tiebroken = resolve_consensus(emotions)
                traj_start = emotions[0]
                traj_end = emotions[-1]
                
                if predictor not in phase_eval["predictions"]:
                    phase_eval["predictions"][predictor] = {}
                elif not isinstance(phase_eval["predictions"][predictor], dict):
                    phase_eval["predictions"][predictor] = {"emotion": phase_eval["predictions"][predictor]}
                
                phase_eval["predictions"][predictor]["seg_agg_peak"] = emotion_peak
                phase_eval["predictions"][predictor]["seg_agg_maj"] = emotion_majority
                phase_eval["predictions"][predictor]["seg_agg_maj_tiebroken"] = emotion_majority_tiebroken
                phase_eval["predictions"][predictor]["seg_agg_traj_start"] = traj_start
                phase_eval["predictions"][predictor]["seg_agg_traj_end"] = traj_end
                
                if predictor.startswith("human_"):
                    phase_rater_results.append({
                        "emotion_peak": emotion_peak,
                        "emotion_majority": emotion_majority,
                        "emotion_majority_tiebroken": emotion_majority_tiebroken,
                        "traj_start": traj_start,
                        "traj_peak": emotion_peak,
                        "traj_end": traj_end
                    })
                    
                updates += 1
                
            if phase_rater_results:
                cons_peak = resolve_consensus([r['emotion_peak'] for r in phase_rater_results])
                cons_maj = resolve_consensus([r['emotion_majority'] for r in phase_rater_results])
                cons_maj_tiebroke = resolve_consensus([r['emotion_majority_tiebroken'] for r in phase_rater_results])
                
                cons_traj_start = resolve_consensus([r['traj_start'] for r in phase_rater_results])
                cons_traj_peak = resolve_consensus([r['traj_peak'] for r in phase_rater_results])
                cons_traj_end = resolve_consensus([r['traj_end'] for r in phase_rater_results])
                
                if "human_consensus" not in phase_eval["predictions"]:
                    phase_eval["predictions"]["human_consensus"] = {}
                    
                phase_eval["predictions"]["human_consensus"]["seg_agg_peak"] = cons_peak
                phase_eval["predictions"]["human_consensus"]["seg_agg_maj"] = cons_maj
                phase_eval["predictions"]["human_consensus"]["seg_agg_maj_tiebroken"] = cons_maj_tiebroke
                phase_eval["predictions"]["human_consensus"]["seg_agg_traj_start"] = cons_traj_start
                phase_eval["predictions"]["human_consensus"]["seg_agg_traj_end"] = cons_traj_end
                updates += 1
                
            if updates > 0:
                reorder_predictions(phase_eval)
                out_path = output_phases_dir / conv_id / phase_json_name
                import os
                os.makedirs(out_path.parent, exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(phase_eval, f, indent=2, ensure_ascii=False)
                    
    print("Successfully mapped human raters and model predictions into their respective phase JSONs.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments_dir', default='output/caller_segments')
    parser.add_argument('--output_dir', default='output/caller_phases')
    args = parser.parse_args()
    
    print("=== Phase Ratings Aggregation ===")
    
    phases_path = Path("output/phases_analysis.json")
    if not phases_path.exists():
        print("Phases analysis file not found at", phases_path)
        return
        
    with open(phases_path, 'r', encoding='utf-8') as f:
        phases_data = json.load(f)
        
    print("-- Processing Humans & Models into Phases --")
    process_phases(phases_data, Path(args.segments_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

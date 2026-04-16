import pandas as pd
import json
from pathlib import Path

# Need to import shared timing and storage routines based on our finalized logic!
from scripts_core.segmentation_utils import parse_time, parse_segment_midpoint
from storage import read_evaluation, write_evaluation

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

def process_humans(phases_data):
    csv_path = Path("output/ratings_segments.csv")
    if not csv_path.exists():
        print("Segment ratings file not found at", csv_path)
        return
        
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    df['emotion'] = df['emotion'].apply(clean_emotion)
    df = df.dropna(subset=['emotion'])
    
    df['midpoint'] = df['segment'].apply(parse_segment_midpoint)
    
    results = []
    
    for conv_file, conv_phases in phases_data.items():
        conv_id = conv_file.replace(".wav", "")
        # Get segments for this conversation specifically
        conv_segs = df[df['conv_id'] == conv_id]
        
        for idx, phase in enumerate(conv_phases.get("phases", [])):
            start_s = parse_time(phase.get("start_time"))
            end_s = parse_time(phase.get("end_time"))
            
            if start_s is None or end_s is None:
                continue
                
            phase_segs = conv_segs[(conv_segs['midpoint'] >= start_s) & (conv_segs['midpoint'] <= end_s)]
            if phase_segs.empty:
                continue
                
            phase_name_raw = phase.get("phase_name", f"phase_{idx}")
            numeric_id = phase_name_raw.split(".")[0].strip() if "." in phase_name_raw else "0"
            phase_wav_name = f"phase_{idx+1}_{numeric_id}_{start_s:.2f}_{end_s:.2f}.wav"
            
            phase_json_path = Path(f"data/caller_phases_24kHz/{conv_id}/{phase_wav_name}")
            phase_eval = read_evaluation(phase_json_path) 
            if "predictions" not in phase_eval:
                phase_eval["predictions"] = {}
                
            raters = phase_segs['rater'].unique()
            phase_rater_results = []
            
            updates = 0
            for rater in raters:
                rater_df = phase_segs[phase_segs['rater'] == rater].sort_values(by='midpoint')
                emotions = rater_df['emotion'].tolist()
                if not emotions:
                    continue
                
                emotion_peak = get_peak_emotion(emotions)
                emotion_majority = get_majority(emotions)
                emotion_majority_tiebroken = resolve_consensus(emotions)
                
                traj_start = emotions[0]
                traj_end = emotions[-1]
                traj_peak = emotion_peak
                
                rater_key = f"human_{rater}"
                if rater_key not in phase_eval["predictions"]:
                    phase_eval["predictions"][rater_key] = {}
                    
                phase_eval["predictions"][rater_key]["seg_agg_peak"] = emotion_peak
                phase_eval["predictions"][rater_key]["seg_agg_maj"] = emotion_majority
                phase_eval["predictions"][rater_key]["seg_agg_maj_tiebroken"] = emotion_majority_tiebroken
                phase_eval["predictions"][rater_key]["seg_agg_traj_start"] = traj_start
                phase_eval["predictions"][rater_key]["seg_agg_traj_end"] = traj_end
                
                phase_rater_results.append({
                    "emotion_peak": emotion_peak,
                    "emotion_majority": emotion_majority,
                    "emotion_majority_tiebroken": emotion_majority_tiebroken,
                    "traj_start": traj_start,
                    "traj_peak": traj_peak,
                    "traj_end": traj_end
                })
                updates += 1
                
            if phase_rater_results:
                cons_peak = resolve_consensus([r['emotion_peak'] for r in phase_rater_results])
                cons_maj = resolve_consensus([r['emotion_majority'] for r in phase_rater_results])
                cons_maj_tiebroke = resolve_consensus([r['emotion_majority_tiebroken'] for r in phase_rater_results])
                
                # Trajectory consensus is calculated point-by-point
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
                write_evaluation(phase_json_path, phase_eval)
                
    print("Successfully mapped human raters directly into their respective phase JSONs.")


def process_models(phases_data):
    segments_dir = Path("output/caller_segments")
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
            eval_data = read_evaluation(json_file)
            for model_name, pred in eval_data.get("predictions", {}).items():
                if isinstance(pred, dict):
                    emotion = pred.get("emotion")
                else:
                    emotion = pred
                    
                emotion = clean_emotion(emotion)
                if emotion and "error" not in emotion:
                    seg_records.append({"midpoint": midpoint, "model": model_name, "emotion": emotion})
                    
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
            phase_wav_name = f"phase_{idx+1}_{numeric_id}_{start_s:.2f}_{end_s:.2f}.wav"
            
            phase_json_path = Path(f"data/caller_phases_24kHz/{conv_id}/{phase_wav_name}")
            
            # Using storage.py to resolve routing safely. 
            # We map against the audio name in caller_phases like the workers do.
            phase_eval = read_evaluation(phase_json_path) 
            if "predictions" not in phase_eval:
                phase_eval["predictions"] = {}
                
            models = phase_segs['model'].unique()
            updates = 0
            for model in models:
                model_df = phase_segs[phase_segs['model'] == model].sort_values(by='midpoint')
                emotions = model_df['emotion'].tolist()
                if not emotions:
                    continue
                
                emotion_peak = get_peak_emotion(emotions)
                emotion_majority = get_majority(emotions)
                emotion_majority_tiebroken = resolve_consensus(emotions)
                traj_start = emotions[0]
                traj_end = emotions[-1]
                
                if model not in phase_eval["predictions"]:
                    phase_eval["predictions"][model] = {}
                elif not isinstance(phase_eval["predictions"][model], dict):
                    phase_eval["predictions"][model] = {"emotion": phase_eval["predictions"][model]}
                
                # Append to phase JSON directly nested inside the model!
                phase_eval["predictions"][model]["seg_agg_peak"] = emotion_peak
                phase_eval["predictions"][model]["seg_agg_maj"] = emotion_majority
                phase_eval["predictions"][model]["seg_agg_maj_tiebroken"] = emotion_majority_tiebroken
                phase_eval["predictions"][model]["seg_agg_traj_start"] = traj_start
                phase_eval["predictions"][model]["seg_agg_traj_end"] = traj_end
                updates += 1
                
            if updates > 0:
                reorder_predictions(phase_eval)
                write_evaluation(phase_json_path, phase_eval)
    
    print("Successfully mapped segment model ratings into their respective phase JSONs.")


def main():
    print("=== Phase Ratings Aggregation ===")
    
    phases_path = Path("output/phases_analysis.json")
    if not phases_path.exists():
        print("Phases analysis file not found at", phases_path)
        return
        
    with open(phases_path, 'r', encoding='utf-8') as f:
        phases_data = json.load(f)
        
    print("-- Processing Humans --")
    process_humans(phases_data)
    
    print("\n-- Processing Models --")
    process_models(phases_data)


if __name__ == "__main__":
    main()

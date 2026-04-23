import pandas as pd
import json
from pathlib import Path
from storage import read_evaluation, write_evaluation
from scripts_core.gpt_emotion_extractor import extract_normalized_emotion

INTENSITY_HIERARCHY = {
    # Original 16 human UI labels + status labels
    "angry": 4, "frustrated": 3, "annoyed": 2,
    "sad": 3, "concerned": 2, "anxious": 3,
    "happy": 3, "grateful": 2, "relieved": 2,
    "fearful": 4, "surprised": 3, "confused": 2,
    "disgusted": 4, "disgust": 4,
    "curious": 1, "calm": 1, "neutral": 0,
    "unusable": -1, "uncertain": -1, "other": -1, "unknown": -1
}

def clean_emotion(e):
    if pd.isna(e): return None
    return str(e).lower().strip()

def resolve_consensus(emotions):
    emotions = [e for e in emotions if pd.notna(e) and e != 'tie']
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

def get_majority(emotions):
    emotions = [e for e in emotions if pd.notna(e) and e != 'tie']
    if not emotions: return None
    counts = pd.Series(emotions).value_counts()
    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    if len(candidates) == 1: return candidates[0]
    return 'tie'

def get_agreement_ratio(emotions):
    emotions = [e for e in emotions if pd.notna(e)]
    if not emotions: return 0.0
    counts = pd.Series(emotions).value_counts()
    return counts.max() / len(emotions)

def get_pairwise_agreement(emotions):
    emotions = [e for e in emotions if pd.notna(e)]
    n = len(emotions)
    if n < 2: return 1.0 # Trivial agreement if 0 or 1 rater
    from itertools import combinations
    pairs = list(combinations(emotions, 2))
    matches = sum(1 for a, b in pairs if a == b)
    return matches / len(pairs)

def reorder_predictions(eval_data):
    if "predictions" not in eval_data: return
    preds = eval_data["predictions"]
    # We want keys starting with 'human' to be first, but human_consensus last in that group.
    human_rater_keys = sorted([k for k in preds.keys() if k.startswith("human") and k != "human_consensus"])
    human_consensus_keys = [k for k in preds.keys() if k == "human_consensus"]
    other_keys = sorted([k for k in preds.keys() if not k.startswith("human")])
    
    new_preds = {}
    for k in human_rater_keys:
        new_preds[k] = preds[k]
    for k in human_consensus_keys:
        new_preds[k] = preds[k]
    for k in other_keys:
        new_preds[k] = preds[k]
        
    eval_data["predictions"] = new_preds

def main():
    print("=== Injecting Human Ratings into Segment JSONs ===")
    csv_path = Path("output/ratings_segments.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    df['emotion'] = df['emotion'].apply(clean_emotion)
    df = df.dropna(subset=['emotion'])
    
    # We need to map back to JSONs. The JSONs are in output/caller_segments/{conv_id}/{segment}.json
    # In ratings_segments.csv, segment is e.g. "seg_0000_0.01_3.29.wav"
    df['segment_stem'] = df['segment'].apply(lambda x: Path(x).stem)
    
    segments_dir = Path("output/caller_segments")
    updates = 0
    
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment_stem = json_file.stem
        
        # Get ratings for this specific segment
        seg_df = df[(df['conv_id'] == conv_id) & (df['segment_stem'] == segment_stem)]
        if seg_df.empty:
            continue
            
        data = read_evaluation(json_file)
            
        if "predictions" not in data:
            data["predictions"] = {}
            
        # Get individual raters
        emotions = []
        for _, row in seg_df.iterrows():
            rater = row['rater']
            emotion = row['emotion']
            note_str = str(row.get('notes', '')).strip()
            
            normalized_emotion = emotion
            comment_text = None
            
            if emotion == "other":
                # We have notes attached to the "other"
                if note_str and note_str.lower() != "nan":
                    comment_text = note_str.replace('\n', ' ').replace('\r', '')
                    rater_key_temp = f"human_{rater}"
                    existing_pred = data.get("predictions", {}).get(rater_key_temp, {})
                    
                    if isinstance(existing_pred, dict) and existing_pred.get("comment") == comment_text and existing_pred.get("extracted_normalized_emotion"):
                        normalized_emotion = existing_pred.get("extracted_normalized_emotion")
                    else:
                        # Call LLM normalization
                        print(f"Normalizing human comment: '{comment_text}'...")
                        normalized_emotion = extract_normalized_emotion(comment_text)
                        print(f" -> Mapped to: {normalized_emotion}")
            
            emotions.append(normalized_emotion)
            
            rater_key = f"human_{rater}"
            
            if rater_key not in data["predictions"]:
                data["predictions"][rater_key] = {}
            elif not isinstance(data["predictions"][rater_key], dict):
                data["predictions"][rater_key] = {"emotion": data["predictions"][rater_key]}
            
            data["predictions"][rater_key]["emotion"] = emotion
            
            if comment_text:
                data["predictions"][rater_key]["comment"] = comment_text
                data["predictions"][rater_key]["extracted_normalized_emotion"] = normalized_emotion
            else:
                # To keep it clean, if it wasn't 'other', remove old fields if they previously existed
                data["predictions"][rater_key].pop("comment", None)
                data["predictions"][rater_key].pop("extracted_normalized_emotion", None)

        # Consensus (now uses the list of 'normalized_emotion' directly)
        majority = get_majority(emotions)
        majority_tiebroken = resolve_consensus(emotions)
        agreement_ratio = get_agreement_ratio(emotions)
        pairwise_agreement = get_pairwise_agreement(emotions)
        
        if "human_consensus" not in data["predictions"]:
            data["predictions"]["human_consensus"] = {}
            
        data["predictions"]["human_consensus"]["maj"] = majority
        data["predictions"]["human_consensus"]["maj_tiebroken"] = majority_tiebroken
        data["predictions"]["human_consensus"]["agreement_ratio"] = agreement_ratio
        data["predictions"]["human_consensus"]["pairwise_agreement"] = pairwise_agreement
        
        reorder_predictions(data)
        
        write_evaluation(json_file, data)
            
        updates += 1
        
    print(f"Updated {updates} segment JSON files with component human ratings and consensus.")

if __name__ == "__main__":
    main()

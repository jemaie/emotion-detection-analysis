import json
import shutil
from pathlib import Path
import os

INPUT_DIR = Path("output/caller_segments")
BASE_OUT_DIR = Path("output/emotion_frameworks")

# Define the theoretical frameworks and their mappings
MAPPINGS = {
    "russell": {
        "angry": "hanv", "frustrated": "hanv", "anxious": "hanv", "disgusted": "hanv", "disgust": "hanv",
        "sad": "lanv", "annoyed": "lanv", "concerned": "lanv", "confused": "lanv",
        "happy": "hapv", "surprised": "hapv",
        "calm": "lapv", "relieved": "lapv", "grateful": "lapv",
        "neutral": "neutral", "other": "neutral", "curious": "neutral", "unknown": "unknown", "unusable": "unusable"
    },
    "gew": {
        "angry": "hostile", "frustrated": "hostile", "annoyed": "hostile", "disgusted": "hostile", "disgust": "hostile",
        "anxious": "vulnerable", "concerned": "vulnerable", "sad": "vulnerable", "confused": "vulnerable",
        "happy": "satisfied",
        "grateful": "receptive", "relieved": "receptive", "surprised": "receptive", "curious": "receptive",
        "neutral": "neutral", "calm": "neutral", "other": "neutral", "unknown": "unknown", "unusable": "unusable"
    },
    "plutchik": {
        "angry": "anger", "frustrated": "anger", "annoyed": "anger",
        "anxious": "fear", "concerned": "fear",
        "happy": "joy", "grateful": "joy", "relieved": "joy",
        "sad": "sadness",
        "surprised": "surprise", "confused": "surprise",
        "curious": "anticipation",
        "disgusted": "disgust", "disgust": "disgust",
        "calm": "trust", "neutral": "trust", "other": "trust", "unknown": "unknown", "unusable": "unusable"
    },
    "willcox": {
        "angry": "mad", "frustrated": "mad", "annoyed": "mad", "disgusted": "mad", "disgust": "mad",
        "anxious": "scared", "concerned": "scared", "confused": "scared",
        "happy": "joyful", "surprised": "joyful", "grateful": "joyful", "relieved": "joyful",
        "calm": "peaceful", "neutral": "peaceful", "other": "peaceful",
        "curious": "powerful",
        "sad": "sad", "unknown": "unknown", "unusable": "unusable"
    },
    "ekman": {
        "angry": "anger", "frustrated": "anger", "annoyed": "anger",
        "anxious": "fear", "concerned": "fear", "confused": "fear",
        "happy": "happiness", "grateful": "happiness", "relieved": "happiness", "calm": "happiness", "curious": "happiness",
        "sad": "sadness",
        "surprised": "surprise",
        "disgusted": "disgust", "disgust": "disgust",
        "neutral": "neutral", "other": "neutral", "unknown": "unknown", "unusable": "unusable"
    }
}

def map_emotion(e_str, mapping):
    if e_str is None: return None
    e_str = str(e_str).lower().strip()
    return mapping.get(e_str, e_str)

def process_file(file_path, output_path, mapping):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    preds = data.get("predictions", {})
    
    # We want to map human_consensus
    if "human_consensus" in preds:
        hc = preds["human_consensus"]
        if isinstance(hc, dict) and "maj_tiebroken" in hc:
            hc["maj_tiebroken"] = map_emotion(hc["maj_tiebroken"], mapping)
            
    # We want to map individual human raters
    for k, v in preds.items():
        if k.startswith("human_") and k != "human_consensus":
            if isinstance(v, dict) and "emotion" in v:
                v["emotion"] = map_emotion(v["emotion"], mapping)
                
    # We want to map all model predictions
    for k, v in preds.items():
        if not k.startswith("human_"):
            if isinstance(v, dict) and "emotion" in v:
                v["emotion"] = map_emotion(v["emotion"], mapping)
            elif isinstance(v, str):
                preds[k] = map_emotion(v, mapping)

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} not found.")
        return
        
    for fw_name, mapping in MAPPINGS.items():
        print(f"Generating dataset for {fw_name}...")
        fw_dir = BASE_OUT_DIR / fw_name / "caller_segments"
        
        import pandas as pd
        csv_path = Path("output/ratings_segments.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
            df['emotion'] = df['emotion'].apply(lambda x: map_emotion(x, mapping))
            df = df.dropna(subset=['emotion'])
            csv_out = BASE_OUT_DIR / fw_name / "ratings_segments.csv"
            import os
            os.makedirs(csv_out.parent, exist_ok=True)
            df.to_csv(csv_out, sep=';', index=False)
        
        count = 0
        for json_file in INPUT_DIR.rglob("*.json"):
            rel_path = json_file.relative_to(INPUT_DIR)
            out_file = fw_dir / rel_path
            process_file(json_file, out_file, mapping)
            count += 1
            
        print(f"  Processed {count} files for {fw_name}.")

if __name__ == "__main__":
    main()

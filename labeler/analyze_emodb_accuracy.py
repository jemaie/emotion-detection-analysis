import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("output/emodb")

# EmoDB Ground Truth mappings
# W=Wut(Anger), L=Langeweile(Boredom), E=Ekel(Disgust), A=Angst(Fear), F=Freude(Happiness), T=Trauer(Sadness), N=Neutral
EMODB_GT = {
    "W": "anger",
    "L": "other",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral"
}

# Normalize predictions to a standard comparable set
PRED_TO_STANDARD = {
    "angry": "anger", "frustrated": "anger", "annoyed": "anger", "anger": "anger", "ang": "anger",
    "anxious": "fear", "concerned": "fear", "confused": "fear", "fearful": "fear", "fear": "fear",
    "happy": "happiness", "grateful": "happiness", "relieved": "happiness", "curious": "happiness", "happiness": "happiness", "joy": "happiness", "hap": "happiness",
    "sad": "sadness", "sadness": "sadness",
    "surprised": "surprise", "surprise": "surprise",
    "disgusted": "disgust", "disgust": "disgust",
    "neutral": "neutral", "other": "other", "unknown": "unknown", "calm": "neutral", "neu": "neutral"
}

# For Russell, we map the GT to Russell
EMODB_TO_RUSSELL = {
    "W": "high_arousal_negative",
    "L": "low_arousal_negative",
    "E": "high_arousal_negative",
    "A": "high_arousal_negative",
    "F": "high_arousal_positive",
    "T": "low_arousal_negative",
    "N": "neutral"
}

def normalize_pred(model_name, pred_val, gt_initial):
    if not pred_val: return None, None
    
    if isinstance(pred_val, dict):
        pred_str = pred_val.get("emotion", "")
    else:
        pred_str = str(pred_val)
        
    pred_str = pred_str.lower().strip()
    
    if "russell" in model_name:
        return pred_str == EMODB_TO_RUSSELL.get(gt_initial), pred_str
        
    # Standard mapping
    standard_pred = PRED_TO_STANDARD.get(pred_str, pred_str)
    standard_gt = EMODB_GT.get(gt_initial)
    
    return standard_pred == standard_gt, standard_pred

def main():
    if not OUTPUT_DIR.exists():
        print(f"Output directory {OUTPUT_DIR} does not exist.")
        return

    results = defaultdict(lambda: {"correct": 0, "total": 0, "distribution": defaultdict(int)})
    all_preds = defaultdict(dict)  # file -> model -> pred
    
    for json_file in OUTPUT_DIR.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        filename = data.get("filename", "")
        if len(filename) < 6: continue
        
        # Position 6 is index 5
        gt_initial = filename[5]
        if gt_initial not in EMODB_GT:
            continue
            
        preds = data.get("predictions", {})
        for model_name, pred_val in preds.items():
            if "error" in str(pred_val).lower():
                continue
                
            is_correct, standard_pred = normalize_pred(model_name, pred_val, gt_initial)
            if is_correct is not None:
                all_preds[filename][model_name] = standard_pred
                results[model_name]["total"] += 1
                results[model_name]["distribution"][standard_pred] += 1
                if is_correct:
                    results[model_name]["correct"] += 1
                    
    print(f"{'Model':<42} | {'Accuracy':<10} | {'Correct/Total'} | {'% Neutral'}")
    print("-" * 85)
    for model_name, stats in sorted(results.items()):
        total = stats["total"]
        if total == 0: continue
        acc = (stats["correct"] / total) * 100
        neutral_pct = (stats["distribution"].get("neutral", 0) / total) * 100
        print(f"{model_name:<42} | {acc:>8.2f}% | {stats['correct']:>3}/{total:<5} | {neutral_pct:>7.2f}%")

    # Consistency check
    print("\nConsistency Check (Model vs. _2 Variant):")
    print(f"{'Model Pair':<42} | {'Overall Agree'} | {'Non-Neutral Agree'}")
    print("-" * 85)
    
    # Get all base models that have a _2 variant
    base_models = sorted([m for m in results.keys() if f"{m}_2" in results])
    
    for base in base_models:
        variant = f"{base}_2"
        agreements = 0
        total_common = 0
        
        nn_agreements = 0
        nn_total = 0
        
        for file_preds in all_preds.values():
            if base in file_preds and variant in file_preds:
                p1, p2 = file_preds[base], file_preds[variant]
                total_common += 1
                if p1 == p2:
                    agreements += 1
                
                # Non-neutral consistency: both must be non-neutral
                if p1 != "neutral" and p2 != "neutral":
                    nn_total += 1
                    if p1 == p2:
                        nn_agreements += 1
        
        if total_common > 0:
            agreement_pct = (agreements / total_common) * 100
            nn_pct = (nn_agreements / nn_total * 100) if nn_total > 0 else 0.0
            nn_str = f"{nn_pct:>6.2f}% ({nn_agreements}/{nn_total})" if nn_total > 0 else "N/A"
            print(f"{base:<42} | {agreement_pct:>7.2f}% ({agreements}/{total_common}) | {nn_str}")

if __name__ == "__main__":
    main()

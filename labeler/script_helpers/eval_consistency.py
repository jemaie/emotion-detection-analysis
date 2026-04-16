import logging
import sys
from pathlib import Path
sys.path.insert(0, "..")

from storage import get_all_evaluations

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def is_valid_pair(predictions, base_model, model_2):
    if base_model in predictions and model_2 in predictions:
        pred_base = predictions[base_model]
        pred_2 = predictions[model_2]
        if (isinstance(pred_base, dict) and "emotion" in pred_base and "error" not in pred_base and
            isinstance(pred_2, dict) and "emotion" in pred_2 and "error" not in pred_2):
            return True
    return False

def get_emotion(predictions, model_name):
    return predictions[model_name]["emotion"].strip().lower()

def evaluate_consistency():
    all_evals = get_all_evaluations()
    eval_list = all_evals.get("concat", []) + all_evals.get("segments", [])
    
    # Store valid files for each comparison pair
    # Using the tuple (base, 2nd) as key
    valid_files_per_pair = {}
    
    # Pre-defined pairs we care about
    pairs = [
        ("openai_realtime", "openai_realtime_2"),
        ("openai_realtime_rp", "openai_realtime_rp_2"),
        ("openai_realtime_ft", "openai_realtime_ft_2"),
        ("openai_realtime_1_5_ft", "openai_realtime_1_5_ft_2"),
        ("openai_realtime_1_5_ft_e", "openai_realtime_1_5_ft_e_2")
    ]
    
    # Find valid files for each pair
    for pair in pairs:
        valid_files_per_pair[pair] = set()
        for idx, eval_data in enumerate(eval_list):
            predictions = eval_data.get("predictions", {})
            if is_valid_pair(predictions, pair[0], pair[1]):
                valid_files_per_pair[pair].add(idx)

    # Intersection of files valid for both pair 1 and pair 2
    comp1_and_2_files = valid_files_per_pair[pairs[0]].intersection(valid_files_per_pair[pairs[1]])
    
    def print_stats(pair, file_indices_to_use, subset_name="All Valid Files"):
        total = 0
        matches = 0
        mismatched_files = []
        
        for idx in file_indices_to_use:
            eval_data = eval_list[idx]
            predictions = eval_data.get("predictions", {})
            if is_valid_pair(predictions, pair[0], pair[1]):
                total += 1
                if get_emotion(predictions, pair[0]) == get_emotion(predictions, pair[1]):
                    matches += 1
                else:
                    filename = eval_data.get("filename", f"index_{idx}")
                    emotion_1 = get_emotion(predictions, pair[0])
                    emotion_2 = get_emotion(predictions, pair[1])
                    mismatched_files.append(f"{filename} ({emotion_1} vs {emotion_2})")
                    
        print(f"\nComparisons between '{pair[0]}' and '{pair[1]}' ({subset_name}):")
        if total > 0:
            match_rate = (matches / total) * 100
            print(f"  Total Valid Comparisons: {total}")
            print(f"  Matches:                 {matches}")
            print(f"  Mismatches:              {len(mismatched_files)}")
            print(f"  Consistency Percentage:  {match_rate:.2f}%")
            if mismatched_files:
                print(f"  Mismatched Files:")
                for mf in mismatched_files:
                    print(f"    - {mf}")
        else:
            print(f"  No valid comparisons found in this subset.")

    print("--- Baseline Comparisons ---")
    print_stats(pairs[0], valid_files_per_pair[pairs[0]])
    print_stats(pairs[1], valid_files_per_pair[pairs[1]])
    
    print("\n--- Newer Models (All Files) ---")
    print_stats(pairs[2], valid_files_per_pair[pairs[2]])
    print_stats(pairs[3], valid_files_per_pair[pairs[3]])

    print("\n--- Newer Models (ONLY using files valid for BOTH comparison 1 and 2) ---")
    # Using comp1_and_2_files subset
    print(f"Intersection subset currently has {len(comp1_and_2_files)} files.")
    print_stats(pairs[2], comp1_and_2_files, subset_name="Subset valid in 1 & 2")
    print_stats(pairs[3], comp1_and_2_files, subset_name="Subset valid in 1 & 2")

if __name__ == "__main__":
    evaluate_consistency()


import pandas as pd
import glob
import json
import os
from collections import Counter

from pathlib import Path

def main():
    print("Starting emotion extraction...")
    
    human_emotions = Counter()
    total_human = 0
    model_emotions = Counter()
    total_model = 0

    base_dir = Path(__file__).parent.parent / "output"
    
    json_files = glob.glob(str(base_dir / "caller_segments/*/*.json")) + \
                 glob.glob(str(base_dir / "caller_phases/*/*.json")) + \
                 glob.glob(str(base_dir / "caller_concat/*.json"))

    model_configs_to_check = [
        "openai_realtime_1_5_ft_e",
        "openai_realtime_1_5_ft_e_2",
        "openai_realtime_1_5_ft_erp",
        "openai_realtime_1_5_ft_erp_2"
    ]

    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as f_in:
                data = json.load(f_in)
                preds = data.get("predictions", {})
                
                for config, pred_data in preds.items():
                    if not isinstance(pred_data, dict):
                        continue
                        
                    # Handle Human Data
                    if config.startswith("human_") and config != "human_consensus":
                        # Prefer extracted_normalized_emotion over generic emotion
                        emotion = pred_data.get("extracted_normalized_emotion", pred_data.get("emotion"))
                        if emotion:
                            emotion_str = str(emotion).strip().lower()
                            if emotion_str and emotion_str != "nan":
                                human_emotions[emotion_str] += 1
                                total_human += 1
                                
                    # Handle Targeted Model Data
                    elif config in model_configs_to_check:
                        emotion = pred_data.get("emotion")
                        if emotion:
                            emotion_str = str(emotion).strip().lower()
                            if emotion_str and emotion_str != "nan":
                                model_emotions[emotion_str] += 1
                                total_model += 1

        except Exception as e:
            pass

    # 3. Create combined CSV
    rows = []
    for em, count in human_emotions.items():
        pct = (count / total_human * 100) if total_human > 0 else 0
        rows.append({"Source": "Human", "Emotion": em, "Count": count, "Percentage": round(pct, 2)})

    for em, count in model_emotions.items():
        pct = (count / total_model * 100) if total_model > 0 else 0
        rows.append({"Source": "Model", "Emotion": em, "Count": count, "Percentage": round(pct, 2)})

    out_df = pd.DataFrame(rows)
    # Sort with Human first, then by count descending
    out_df["SortOrder"] = out_df["Source"].apply(lambda x: 0 if x == "Human" else 1)
    out_df = out_df.sort_values(by=["SortOrder", "Count"], ascending=[True, False]).drop(columns=["SortOrder"])
    
    out_csv = base_dir / "used_emotions_summary.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Done! Saved to {out_csv}")

if __name__ == "__main__":
    main()

import json
from pathlib import Path

def main():
    audio_dir_concat = Path("data/caller_concat_24kHz")
    audio_dir_segments = Path("data/caller_segments_24kHz")
    eval_dir = Path("output")
    
    missing_concat = 0
    incomplete_concat = 0
    missing_segments = 0
    incomplete_segments = 0
    
    # Concat
    if audio_dir_concat.exists():
        for wav_file in audio_dir_concat.glob("*.wav"):
            json_file = eval_dir / "caller_concat" / f"{wav_file.stem}.json"
            if not json_file.exists():
                missing_concat += 1
                print(f"Missing concat JSON for: {wav_file.name}")
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    preds = data.get("predictions", {})
                    # Check if any prediction has an error
                    has_error = False
                    for model, pred in preds.items():
                        if isinstance(pred, dict) and "error" in pred:
                            has_error = True
                            print(f"Incomplete concat JSON (error in {model}): {wav_file.name}")
                    if has_error:
                        incomplete_concat += 1
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                
    # Segments
    if audio_dir_segments.exists():
        for wav_file in audio_dir_segments.rglob("*.wav"):
            conv_id = wav_file.parent.name
            json_file = eval_dir / "caller_segments" / conv_id / f"{wav_file.stem}.json"
            if not json_file.exists():
                missing_segments += 1
                print(f"Missing segment JSON for: {conv_id}/{wav_file.name}")
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    preds = data.get("predictions", {})
                    has_error = False
                    for model, pred in preds.items():
                        if isinstance(pred, dict) and "error" in pred:
                            has_error = True
                            print(f"Incomplete segment JSON (error in {model}): {conv_id}/{wav_file.name}")
                    if has_error:
                        incomplete_segments += 1
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

    print(f"\nSummary:")
    print(f"Missing concat files: {missing_concat}")
    print(f"Incomplete concat files: {incomplete_concat}")
    print(f"Missing segment files: {missing_segments}")
    print(f"Incomplete segment files: {incomplete_segments}")

if __name__ == '__main__':
    main()

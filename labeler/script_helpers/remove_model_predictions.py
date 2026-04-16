"""
remove_model_predictions.py

Usage:
    python remove_model_predictions.py <model_key>

Example:
    python remove_model_predictions.py openai_realtime_rp
    python remove_model_predictions.py "iic/emotion2vec_base"

Removes the given model key from the "predictions" object in every output JSON file.
"""

import json
import sys
from pathlib import Path

OUTPUT_DIR = Path("../output")

def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_model_predictions.py <model_key>")
        sys.exit(1)

    model_key = sys.argv[1]
    changed = 0
    skipped = 0

    for f in sorted(OUTPUT_DIR.rglob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            preds = data.get("predictions", {})
            if model_key in preds:
                del preds[model_key]
                f.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
                print(f"  Cleaned: {f}")
                changed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR {f}: {e}")

    print(f"\nDone. {changed} file(s) updated, {skipped} had no '{model_key}' entry.")

if __name__ == "__main__":
    main()

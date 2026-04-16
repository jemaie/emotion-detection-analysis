"""
Collect all unique emotion labels predicted by specific models across
the labeler output (both caller_concat and caller_segments).

Target models:
  - openai_realtime_1_5_ft_e
  - openai_realtime_1_5_ft_e_2
  - openai_realtime_1_5_ft_erp
"""

import json
from collections import Counter
from pathlib import Path

OUTPUT_DIR = Path("../output")

TARGET_MODELS = [
    "openai_realtime_1_5_ft_e",
    "openai_realtime_1_5_ft_e_2",
    "openai_realtime_1_5_ft_erp",
]


def collect_labels(output_dir: Path) -> dict[str, Counter]:
    """Return a dict mapping each target model to a Counter of emotion labels."""
    label_counts: dict[str, Counter] = {model: Counter() for model in TARGET_MODELS}

    json_files: list[Path] = []

    # caller_concat: flat directory of JSON files
    concat_dir = output_dir / "caller_concat"
    if concat_dir.exists():
        json_files.extend(concat_dir.glob("*.json"))

    # caller_segments: subdirectories, each containing segment JSON files
    segments_dir = output_dir / "caller_segments"
    if segments_dir.exists():
        json_files.extend(segments_dir.rglob("*.json"))

    for json_file in sorted(json_files):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        predictions = data.get("predictions", {})
        for model in TARGET_MODELS:
            if model in predictions:
                emotion = predictions[model].get("emotion")
                if emotion is not None:
                    label_counts[model][emotion] += 1

    return label_counts


def main() -> None:
    label_counts = collect_labels(OUTPUT_DIR)

    # -- Per-model breakdown ------------------------------------------------
    for model in TARGET_MODELS:
        counts = label_counts[model]
        total = sum(counts.values())
        print(f"\n{'-' * 60}")
        print(f"Model: {model}  ({total} predictions)")
        print(f"{'-' * 60}")
        for emotion, count in counts.most_common():
            pct = count / total * 100 if total else 0
            print(f"  {emotion:<20s}  {count:>5d}  ({pct:5.1f}%)")

    # -- Combined unique labels across all three models --------------------
    all_labels = set()
    for counts in label_counts.values():
        all_labels.update(counts.keys())

    print(f"\n{'=' * 60}")
    print(f"All unique emotion labels across target models ({len(all_labels)}):")
    print(f"{'=' * 60}")
    for label in sorted(all_labels):
        print(f"  * {label}")


if __name__ == "__main__":
    main()

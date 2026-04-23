import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def parse_duration(segment_str):
    # expect format: seg_0000_1.50_3.00
    parts = segment_str.split('_')
    if len(parts) >= 4:
        try:
            return float(parts[3]) - float(parts[2])
        except ValueError:
            pass
    return 0.0

def load_human_ratings(csv_path):
    if not csv_path.exists():
        return set(), set()
        
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    
    # Clean emotions
    df['emotion'] = df['emotion'].astype(str).str.lower().str.strip()
    
    # Stem segments so "seg_0000.wav" matches JSON stems
    if 'segment' in df.columns:
        df['segment_stem'] = df['segment'].apply(lambda x: Path(str(x)).stem)
    else:
        return set(), set()
    
    # Find segments with at least one "unusable"
    unusable_mask = df['emotion'] == 'unusable'
    unusable_segments = set(zip(df[unusable_mask]['conv_id'], df[unusable_mask]['segment_stem']))
    
    # Find segments with at least one "uncertain"
    uncertain_mask = df['emotion'] == 'uncertain'
    uncertain_segments = set(zip(df[uncertain_mask]['conv_id'], df[uncertain_mask]['segment_stem']))
    
    return unusable_segments, uncertain_segments

def analyze_consistency(args):
    segments_dir = Path(args.segments_dir)
    if not segments_dir.exists():
        print("No segments found.")
        return

    csv_path = segments_dir.parent / "ratings_segments.csv"
    if not csv_path.exists():
        csv_path = Path("output/ratings_segments.csv")
    unusable_set, uncertain_set = load_human_ratings(csv_path)

    # results format: results[tier][duration_bucket][model] = {"match": 0, "total": 0}
    # Tiers: "Baseline", "Usable-Only", "Pristine"
    # Buckets: 0 (All), 1 (>1s), 2 (>2s), 3 (>3s)
    
    tiers = ["Baseline", "Usable-Only", "Pristine"]
    buckets = [0, 1, 2, 3] 
    
    # nested init
    results = {
        t: {b: defaultdict(lambda: {"match": 0, "total": 0}) for b in buckets}
        for t in tiers
    }
    
    found_models_ordered = []
    
    phase_results = defaultdict(lambda: {"direct_match": 0, "maj_match": 0, "peak_match": 0, "total": 0})
    phases_dir = Path(args.phases_dir)
    
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment_stem = json_file.stem
        segment_id = (conv_id, segment_stem)
        
        duration = parse_duration(segment_stem)
        
        # Determine tiers for this segment
        segment_tiers = ["Baseline"]
        if segment_id not in unusable_set:
            segment_tiers.append("Usable-Only")
            if segment_id not in uncertain_set:
                segment_tiers.append("Pristine")
                
        # Determine buckets for this segment
        segment_buckets = [0]
        if duration > 1.0: segment_buckets.append(1)
        if duration > 2.0: segment_buckets.append(2)
        if duration > 3.0: segment_buckets.append(3)
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        preds = data.get("predictions", {})
        
        # Maintain chronologic/exact order from the JSON keys
        for key in preds.keys():
            if not key.endswith("_2"):
                continue
            base_model = key[:-2]
            model_2 = key
            
            if base_model in preds and model_2 in preds:
                if base_model not in found_models_ordered:
                    found_models_ordered.append(base_model)
                
                # Extract emotions carefully
                pred_1 = preds.get(base_model)
                pred_2 = preds.get(model_2)
                
                e1 = str(pred_1.get("emotion") if isinstance(pred_1, dict) else pred_1).lower().strip()
                e2 = str(pred_2.get("emotion") if isinstance(pred_2, dict) else pred_2).lower().strip()
                
                match = 1 if e1 == e2 else 0
                
                for t in segment_tiers:
                    for b in segment_buckets:
                        results[t][b][base_model]["total"] += 1
                        results[t][b][base_model]["match"] += match

    if phases_dir.exists():
        for json_file in phases_dir.rglob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            preds = data.get("predictions", {})
            for key in preds.keys():
                if not key.endswith("_2"):
                    continue
                base_model = key[:-2]
                model_2 = key
                if base_model in preds and model_2 in preds:
                    if base_model not in found_models_ordered:
                        found_models_ordered.append(base_model)
                    pred_1 = preds.get(base_model)
                    pred_2 = preds.get(model_2)
                    
                    e1 = str(pred_1.get("emotion") if isinstance(pred_1, dict) else pred_1).lower().strip()
                    e2 = str(pred_2.get("emotion") if isinstance(pred_2, dict) else pred_2).lower().strip()
                    
                    if isinstance(pred_1, dict) and isinstance(pred_2, dict):
                        maj1 = str(pred_1.get("seg_agg_maj_tiebroken", "none")).lower().strip()
                        maj2 = str(pred_2.get("seg_agg_maj_tiebroken", "none")).lower().strip()
                        peak1 = str(pred_1.get("seg_agg_peak", "none")).lower().strip()
                        peak2 = str(pred_2.get("seg_agg_peak", "none")).lower().strip()
                    else:
                        maj1, maj2, peak1, peak2 = "none", "none", "none", "none"
                    
                    phase_results[base_model]["total"] += 1
                    if e1 == e2 and e1 != "none":
                        phase_results[base_model]["direct_match"] += 1
                    if maj1 == maj2 and maj1 != "none":
                        phase_results[base_model]["maj_match"] += 1
                    if peak1 == peak2 and peak1 != "none":
                        phase_results[base_model]["peak_match"] += 1

    if not found_models_ordered:
        print("No paired runs (base and _2) found.")
        return

    print("=== Phase 0: Deep Model Consistency Analysis ===\n")
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. 'Usable-Only' and 'Pristine' tiers will mirror 'Baseline'.\n")
    
    tier_dfs = {}
    
    for tier in tiers:
        print(f"--- Tier: {tier} ---")
        if tier == "Usable-Only":
            print("(Excludes segments where ANY coder marked it 'unusable')")
        elif tier == "Pristine":
            print("(Excludes segments where ANY coder marked it 'unusable' OR 'uncertain')")
            
        rows = []
        for model in found_models_ordered:
            row = {"Model Config": model}
            for b in buckets:
                stats = results[tier][b][model]
                total = stats["total"]
                matches = stats["match"]
                pct = (matches / total * 100) if total > 0 else 0.0
                
                bucket_name = "All Segments" if b == 0 else f">{b}s"
                row[f"{bucket_name} (%)"] = f"{pct:.1f}% ({total} segs)"
            rows.append(row)
            
        df = pd.DataFrame(rows)
        import tabulate
        print(df.to_markdown(index=False))
        print("\n")
        
        # Save tier dataframe to a dictionary for CSV exporting
        tier_dfs[tier] = df

    print("\n--- Phase-Level Consistency ---")
    if sum(stats["total"] for stats in phase_results.values()) == 0:
        print("No phase data found.")
    else:
        phase_rows = []
        for model in found_models_ordered:
            stats = phase_results[model]
            total = stats["total"]
            if total > 0:
                direct_pct = (stats["direct_match"] / total) * 100
                maj_pct = (stats["maj_match"] / total) * 100
                peak_pct = (stats["peak_match"] / total) * 100
                phase_rows.append({
                    "Model Config": model,
                    "Total Phases": total,
                    "Direct Match (%)": f"{direct_pct:.1f}%",
                    "Majority Match (%)": f"{maj_pct:.1f}%",
                    "Peak Match (%)": f"{peak_pct:.1f}%"
                })
        if phase_rows:
            phase_df = pd.DataFrame(phase_rows)
            import tabulate
            print(phase_df.to_markdown(index=False))
            print("\n")

    # Prepare analysis_results directory
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Export 1: Basic Baseline (Baseline Tier, All Segments only)
    if "Baseline" in tier_dfs:
        base_df = tier_dfs["Baseline"][["Model Config", "All Segments (%)"]].copy()
        base_df.columns = ["Model Config", "Consistency"] # Simplify column name
        base_csv = results_dir / "consistency_baseline.csv"
        base_df.to_csv(base_csv, index=False)
        print(f"Exported Baseline CSV to {base_csv}")

    # Export 2: Deep Analysis (All Tiers, All Buckets)
    deep_rows = []
    for tier_name, tier_df in tier_dfs.items():
        tier_df_copy = tier_df.copy()
        tier_df_copy.insert(0, "Data Tier", tier_name)
        deep_rows.append(tier_df_copy)
        
    if deep_rows:
        deep_csv = results_dir / "consistency_deep.csv"
        pd.concat(deep_rows, ignore_index=True).to_csv(deep_csv, index=False)
        print(f"Exported Deep Analysis CSV to {deep_csv}")

    if 'phase_df' in locals() and not phase_df.empty:
        phase_csv = results_dir / "consistency_phase.csv"
        phase_df.to_csv(phase_csv, index=False)
        print(f"Exported Phase Consistency CSV to {phase_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments_dir', default='output/caller_segments')
    parser.add_argument('--output_dir', default='output/analysis_results')
    parser.add_argument('--phases_dir', default='output/caller_phases')
    args = parser.parse_args()
    
    analyze_consistency(args)

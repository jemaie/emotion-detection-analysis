import os
import json
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    "out_mapped/pyannote",
    "out_pyannote/1_refs",
    "out_mapped/openai",
    "out_openai/1_refs",
    "out_openai/2_refs",
    "out_openai/3_refs",
    "out_openai/4_refs",
]

# Find all call IDs
calls = set()
for src in sources:
    index_dir = os.path.join(base_dir, src.replace('/', os.sep), "index")
    if os.path.exists(index_dir):
        for f in os.listdir(index_dir):
            if f.endswith(".summary.json"):
                calls.add(f.replace(".summary.json", ""))

calls = sorted(list(calls))

# ── CSV: eval_num_speakers.csv & eval_agent_matched_by_reference.csv ─────────

flags_config = {
    "num_speakers": {"target": "dynamic"},
    "agent_matched_by_reference": {"target": True},
    # "fallback_used": {"target": True}  # always inverse of agent_matched_by_reference
}

# Override defaults for calls we know have not 2 speakers
expected_speakers_per_call = {
    "conv__+4915203230182_22-08-2024_8_06_41": 3,
    "conv__+491779108125_26-08-2024_10_23_16": 3,
    "conv__+491729920245_24-07-2024_10_15_48": 3
}

for flag, config in flags_config.items():
    csv_file = os.path.join(base_dir, f"eval_{flag}.csv")
    with open(csv_file, 'w', newline='') as f:
        f.write("sep=,\n")
        writer = csv.writer(f)
        
        header = ["Call ID"]
        if flag == "num_speakers":
            header.append("Expected")
        header.extend(sources)
        writer.writerow(header)
        
        totals = {src: 0 for src in sources}
        
        for call in calls:
            row = [call]
            if flag == "num_speakers":
                row.append(expected_speakers_per_call.get(call, 2))
                
            for src in sources:
                json_path = os.path.join(base_dir, src.replace('/', os.sep), "index", f"{call}.summary.json")
                val = "N/A"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                            val = data.get("flags", {}).get(flag, "N/A")
                            
                            expected_target = config["target"]
                            if flag == "num_speakers":
                                expected_target = expected_speakers_per_call.get(call, 2)
                                
                            if val == expected_target:
                                totals[src] += 1
                    except Exception as e:
                        val = "Error"
                row.append(val)
            writer.writerow(row)
            
        writer.writerow([])
        
        total_label = "Total (== Expected)" if flag == "num_speakers" else f"Total (== {config['target']})"
        total_row = [total_label]
        if flag == "num_speakers":
            total_row.append("") # skip expected column
        for src in sources:
            total_row.append(totals[src])
        writer.writerow(total_row)
        
        pct_row = ["Percentage"]
        if flag == "num_speakers":
            pct_row.append("") # skip expected column
        for src in sources:
            pct = (totals[src] / len(calls)) * 100 if len(calls) > 0 else 0
            pct_row.append(f"{pct:.1f}%")
        writer.writerow(pct_row)
        
    print(f"Created {csv_file}")


# ── CSV: eval_caller_duration.csv & eval_caller_segments.csv ─────────────────

def load_summary(src, call):
    path = os.path.join(base_dir, src.replace("/", os.sep), "index", f"{call}.summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_caller_duration(data):
    """Sum durations of all speakers mapped to 'caller' role."""
    if not data:
        return None
    role_map = data.get("speaker_to_role", {})
    durations = data.get("speaker_durations_sec", {})
    total = 0.0
    for speaker, role in role_map.items():
        if role == "caller":
            total += durations.get(speaker, 0.0)
    return round(total, 2)


# --- CSV: Caller Duration ---
csv_file = os.path.join(base_dir, "eval_caller_duration.csv")
with open(csv_file, "w", newline="") as f:
    f.write("sep=,\n")
    writer = csv.writer(f)
    writer.writerow(["Call ID"] + sources)

    sums = {src: 0.0 for src in sources}
    counts = {src: 0 for src in sources}
    zero_counts = {src: 0 for src in sources}
    non_zero_sums = {src: 0.0 for src in sources}
    non_zero_counts = {src: 0 for src in sources}

    for call in calls:
        row = [call]
        for src in sources:
            data = load_summary(src, call)
            dur = get_caller_duration(data)
            if dur is not None:
                sums[src] += dur
                counts[src] += 1
                if dur == 0.0:
                    zero_counts[src] += 1
                else:
                    non_zero_sums[src] += dur
                    non_zero_counts[src] += 1
                row.append(f"{dur:.2f}")
            else:
                row.append("N/A")
        writer.writerow(row)

    writer.writerow([])

    avg_row = ["Average (sec)"]
    for src in sources:
        avg = sums[src] / counts[src] if counts[src] > 0 else 0
        avg_row.append(f"{avg:.2f}")
    writer.writerow(avg_row)

    avg_nz_row = ["Average (>0s) (sec)"]
    for src in sources:
        avg_nz = non_zero_sums[src] / non_zero_counts[src] if non_zero_counts[src] > 0 else 0
        avg_nz_row.append(f"{avg_nz:.2f}")
    writer.writerow(avg_nz_row)

    zero_row = ["0-Duration Calls"]
    for src in sources:
        zero_row.append(str(zero_counts[src]))
    writer.writerow(zero_row)

    total_row = ["Total (sec)"]
    for src in sources:
        total_row.append(f"{sums[src]:.2f}")
    writer.writerow(total_row)

print(f"Created {csv_file}")


# --- CSV: Caller Segments (raw and final) ---
csv_file = os.path.join(base_dir, "eval_caller_segments.csv")
with open(csv_file, "w", newline="") as f:
    f.write("sep=,\n")
    writer = csv.writer(f)

    header = ["Call ID"]
    for src in sources:
        header.append(f"{src} (raw)")
        header.append(f"{src} (final)")
    writer.writerow(header)

    sums_raw = {src: 0 for src in sources}
    sums_final = {src: 0 for src in sources}
    sums_dropped_total = {src: 0 for src in sources}
    sums_dropped_trim_dur = {src: 0 for src in sources}
    sums_dropped_overlap = {src: 0 for src in sources}
    sums_merged = {src: 0 for src in sources}
    counts = {src: 0 for src in sources}

    for call in calls:
        row = [call]
        for src in sources:
            data = load_summary(src, call)
            if data:
                raw = data.get("num_segments_caller_raw", "N/A")
                final = data.get("num_segments_caller_final", "N/A")
                dropped_total = data.get("num_segments_caller_dropped", 0)
                dropped_trim_dur = data.get("num_segments_caller_dropped_trim_dur", 0)
                dropped_overlap = data.get("num_segments_caller_dropped_overlap", 0)
                merged = data.get("num_segments_caller_merged", 0)
                
                # If these new detailed metrics don't exist yet, try to default them
                if "num_segments_caller_dropped_trim_dur" not in data:
                     # Old format fallback
                     dropped_trim_dur = dropped_total
                     
                
                if isinstance(raw, int) and isinstance(final, int):
                    sums_raw[src] += raw
                    sums_final[src] += final
                    sums_dropped_total[src] += dropped_total
                    sums_dropped_trim_dur[src] += dropped_trim_dur
                    sums_dropped_overlap[src] += dropped_overlap
                    sums_merged[src] += merged
                
                counts[src] += 1
                row.append(raw)
                row.append(final)
            else:
                row.append("N/A")
                row.append("N/A")
        writer.writerow(row)

    writer.writerow([])

    avg_row = ["Average"]
    for src in sources:
        avg_r = sums_raw[src] / counts[src] if counts[src] > 0 else 0
        avg_f = sums_final[src] / counts[src] if counts[src] > 0 else 0
        avg_row.append(f"{avg_r:.1f}")
        avg_row.append(f"{avg_f:.1f}")
    writer.writerow(avg_row)

    total_row = ["Total"]
    for src in sources:
        total_row.append(sums_raw[src])
        total_row.append(sums_final[src])
    writer.writerow(total_row)

    drop_row = ["Total Drop rate (filtered out)"]
    for src in sources:
        if sums_raw[src] > 0:
            drop_pct = (sums_dropped_total[src] / sums_raw[src]) * 100
            drop_row.append(f"{drop_pct:.1f}%")
        else:
            drop_row.append("N/A")
        drop_row.append("")
    writer.writerow(drop_row)

    drop_dur_row = [" - Trim / Duration Drop rate"]
    for src in sources:
        if sums_raw[src] > 0:
            drop_pct = (sums_dropped_trim_dur[src] / sums_raw[src]) * 100
            drop_dur_row.append(f"{drop_pct:.1f}%")
        else:
            drop_dur_row.append("N/A")
        drop_dur_row.append("")
    writer.writerow(drop_dur_row)

    drop_overlap_row = [" - Overlap Drop rate"]
    for src in sources:
        if sums_raw[src] > 0:
            drop_pct = (sums_dropped_overlap[src] / sums_raw[src]) * 100
            drop_overlap_row.append(f"{drop_pct:.1f}%")
        else:
            drop_overlap_row.append("N/A")
        drop_overlap_row.append("")
    writer.writerow(drop_overlap_row)

    merge_row = ["Merge rate (combined)"]
    for src in sources:
        if sums_raw[src] > 0:
            merge_pct = (sums_merged[src] / sums_raw[src]) * 100
            merge_row.append(f"{merge_pct:.1f}%")
        else:
            merge_row.append("N/A")
        merge_row.append("")
    writer.writerow(merge_row)

    old_calc_row = ["Total reduction rate (merged+dropped)"]
    for src in sources:
        if sums_raw[src] > 0:
            red_pct = ((sums_raw[src] - sums_final[src]) / sums_raw[src]) * 100
            old_calc_row.append(f"{red_pct:.1f}%")
        else:
            old_calc_row.append("N/A")
        old_calc_row.append("")
    writer.writerow(old_calc_row)

print(f"Created {csv_file}")

import os
import json
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))

sources = ["output"]

# Find all call IDs
calls = set()
for src in sources:
    index_dir = os.path.join(base_dir, src.replace('/', os.sep), "index")
    if os.path.exists(index_dir):
        for f in os.listdir(index_dir):
            if f.endswith(".summary.json"):
                calls.add(f.replace(".summary.json", ""))

calls = sorted(list(calls))

expected_speakers_per_call = {
    "conv__+4915203230182_22-08-2024_8_06_41": 3,
    "conv__+491779108125_26-08-2024_10_23_16": 3,
    "conv__+491729920245_24-07-2024_10_15_48": 3
}

def load_summary(src, call):
    path = os.path.join(base_dir, src.replace("/", os.sep), "index", f"{call}.summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def get_caller_duration(data):
    if not data:
        return None
    role_map = data.get("speaker_to_role", {})
    durations = data.get("speaker_durations_sec", {})
    total = 0.0
    for speaker, role in role_map.items():
        if role == "caller":
            total += durations.get(speaker, 0.0)
    return round(total, 2)

csv_file = os.path.join(base_dir, "eval_output_summary.csv")
with open(csv_file, 'w', newline='') as f:
    f.write("sep=,\n")
    writer = csv.writer(f)
    
    header = [
        "Call ID", 
        "Expected Speakers", 
        "Agent Matched", 
        "Caller Duration (s)", 
        "Caller Segments (raw)", 
        "Caller Segments (final)"
    ]
    writer.writerow(header)
    
    # Track stats
    total_matched = 0
    sums_dur = 0.0
    counts_dur = 0
    zero_counts = 0
    non_zero_sums = 0.0
    non_zero_counts = 0
    
    sums_raw = 0
    sums_final = 0
    sums_dropped_total = 0
    sums_dropped_trim_dur = 0
    sums_dropped_overlap = 0
    sums_merged = 0
    counts_seg = 0
    
    for call in calls:
        row = [call, expected_speakers_per_call.get(call, 2)]
        data = load_summary("output", call)
        
        # Agent Matched
        agent_matched = "N/A"
        if data:
            agent_matched = data.get("flags", {}).get("agent_matched_by_reference", "N/A")
            if agent_matched is True:
                total_matched += 1
        row.append(agent_matched)
        
        # Caller Duration
        dur = get_caller_duration(data)
        if dur is not None:
            sums_dur += dur
            counts_dur += 1
            if dur == 0.0:
                zero_counts += 1
            else:
                non_zero_sums += dur
                non_zero_counts += 1
            row.append(f"{dur:.2f}")
        else:
            row.append("N/A")
            
        # Caller Segments
        if data:
            raw = data.get("num_segments_caller_raw", "N/A")
            final = data.get("num_segments_caller_final", "N/A")
            dropped_total = data.get("num_segments_caller_dropped", 0)
            dropped_trim_dur = data.get("num_segments_caller_dropped_trim_dur", 0)
            if "num_segments_caller_dropped_trim_dur" not in data:
                 dropped_trim_dur = dropped_total
            dropped_overlap = data.get("num_segments_caller_dropped_overlap", 0)
            merged = data.get("num_segments_caller_merged", 0)
            
            if isinstance(raw, int) and isinstance(final, int):
                sums_raw += raw
                sums_final += final
                sums_dropped_total += dropped_total
                sums_dropped_trim_dur += dropped_trim_dur
                sums_dropped_overlap += dropped_overlap
                sums_merged += merged
            
            counts_seg += 1
            row.append(raw)
            row.append(final)
        else:
            row.append("N/A")
            row.append("N/A")
            
        writer.writerow(row)
        
    writer.writerow([])
    
    # Summary rows
    # 1. Matched
    writer.writerow(["SUMMARY: Agent Matched"])
    pct_matched = (total_matched / len(calls)) * 100 if len(calls) > 0 else 0
    writer.writerow(["", f"Total Matched: {total_matched}"])
    writer.writerow(["", f"Percentage: {pct_matched:.1f}%"])
    writer.writerow([])
    
    # 2. Duration
    writer.writerow(["SUMMARY: Caller Duration"])
    avg_dur = sums_dur / counts_dur if counts_dur > 0 else 0
    avg_nz_dur = non_zero_sums / non_zero_counts if non_zero_counts > 0 else 0
    writer.writerow(["", f"Average (s): {avg_dur:.2f}"])
    writer.writerow(["", f"Average (>0s) (s): {avg_nz_dur:.2f}"])
    writer.writerow(["", f"0-Duration Calls: {zero_counts}"])
    writer.writerow(["", f"Total (s): {sums_dur:.2f}"])
    writer.writerow([])
    
    # 3. Segments
    writer.writerow(["SUMMARY: Caller Segments"])
    avg_r = sums_raw / counts_seg if counts_seg > 0 else 0
    avg_f = sums_final / counts_seg if counts_seg > 0 else 0
    drop_pct = (sums_dropped_total / sums_raw) * 100 if sums_raw > 0 else 0
    drop_dur_pct = (sums_dropped_trim_dur / sums_raw) * 100 if sums_raw > 0 else 0
    drop_ov_pct = (sums_dropped_overlap / sums_raw) * 100 if sums_raw > 0 else 0
    merge_pct = (sums_merged / sums_raw) * 100 if sums_raw > 0 else 0
    red_pct = ((sums_raw - sums_final) / sums_raw) * 100 if sums_raw > 0 else 0
    
    writer.writerow(["", "Average Raw:", f"{avg_r:.1f}"])
    writer.writerow(["", "Average Final:", f"{avg_f:.1f}"])
    writer.writerow(["", "Total Raw:", sums_raw])
    writer.writerow(["", "Total Final:", sums_final])
    writer.writerow(["", "Total Drop Rate:", f"{drop_pct:.1f}%"])
    writer.writerow(["", " - Trim/Duration:", f"{drop_dur_pct:.1f}%"])
    writer.writerow(["", " - Overlap:", f"{drop_ov_pct:.1f}%"])
    writer.writerow(["", "Merge Rate:", f"{merge_pct:.1f}%"])
    writer.writerow(["", "Total Red. Rate:", f"{red_pct:.1f}%"])

print(f"Created {csv_file}")

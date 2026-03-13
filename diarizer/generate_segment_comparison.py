import os
import csv
import re

METHODS = [
    "out_mapped/openai",
    "out_mapped/pyannote",
    "out_openai/1_refs",
    "out_openai/2_refs",
    "out_pyannote/1_refs"
]

# Ignore discrepancies smaller than this threshold (in seconds)
# This removes minor boundary differences to highlight real missing speech/bleed-through
MIN_DISCREPANCY_DURATION = 0.5 

def parse_segments(directory):
    segments = []
    if not os.path.exists(directory):
        return segments
    
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            m = re.match(r"seg_\d+_(.+?)_(.+?)\.wav", filename)
            if m:
                start = float(m.group(1))
                end = float(m.group(2))
                segments.append((start, end))
    return sorted(segments)

def get_discrepancies(call_id, base_dir):
    method_segments = {}
    all_time_points = set()
    
    for method in METHODS:
        path = os.path.join(base_dir, method, "caller_segments", call_id)
        segs = parse_segments(path)
        method_segments[method] = segs
        for s, e in segs:
            all_time_points.add(s)
            all_time_points.add(e)
            
    time_points = sorted(list(all_time_points))
    intervals = []
    for i in range(len(time_points)-1):
        s = time_points[i]
        e = time_points[i+1]
        if abs(e - s) < 1e-5:
            continue
        mid = (s + e) / 2
        
        coverage = {}
        for method in METHODS:
            cov = False
            for ms, me in method_segments[method]:
                if ms <= mid <= me:
                    cov = True
                    break
            coverage[method] = cov
        
        has_any = any(coverage.values())
        has_all = all(coverage.values())
        
        if has_any and not has_all:
            intervals.append((s, e, coverage))
            
    # merge adjacent intervals with the same coverage pattern
    merged = []
    if not intervals:
        return merged
        
    curr_s, curr_e, curr_cov = intervals[0]
    for i in range(1, len(intervals)):
        next_s, next_e, next_cov = intervals[i]
        if abs(next_s - curr_e) < 1e-5 and next_cov == curr_cov:
            curr_e = next_e
        else:
            merged.append((curr_s, curr_e, curr_cov))
            curr_s, curr_e, curr_cov = next_s, next_e, next_cov
    merged.append((curr_s, curr_e, curr_cov))
    
    return merged

def main():
    base_dir = "c:/Users/jemai/projects/emotion-detection-analysis/diarizer"
    eval_file = os.path.join(base_dir, "eval_manual_listening.csv")
    out_csv = os.path.join(base_dir, "eval_segment_discrepancies.csv")
    
    call_ids = set()
    with open(eval_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_idx = 0
    delimiter = ";" if ";" in lines[0] else ","
    if lines and lines[0].startswith("sep="):
        start_idx = 1
        delimiter = lines[0].strip().split("=")[1]
        
    reader = csv.DictReader(lines[start_idx:], delimiter=delimiter)
    for row in reader:
        call_id = row.get("Call ID")
        if call_id and call_id.strip():
            call_ids.add(call_id.strip())
            
    call_ids = sorted(list(call_ids))
    
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        header = ["Call ID", "Start", "End", "Duration"] + METHODS
        writer.writerow(header)
        
        # We also might want to add sep=, as in the other file
        # Wait, the other file added sep=, manually, we can omit it if Excel reads standard CSV
        # Let's add it to be consistent with Excel in regions with comma decimal
        
        for call_id in call_ids:
            discrepancies = get_discrepancies(call_id, base_dir)
            for s, e, cov in discrepancies:
                dur = round(e - s, 2)
                if dur < MIN_DISCREPANCY_DURATION: 
                    continue
                row = [call_id, f"{s:.2f}", f"{e:.2f}", f"{dur:.2f}"]
                for m in METHODS:
                    row.append("YES" if cov[m] else "")
                writer.writerow(row)
                
    print(f"Generated {out_csv} with {len(call_ids)} calls analyzed.")
    print(f"Ignored minor discrepancies < {MIN_DISCREPANCY_DURATION}s.")

if __name__ == "__main__":
    main()

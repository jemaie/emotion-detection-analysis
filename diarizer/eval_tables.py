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

flags_config = {
    "num_speakers": {"target": 2},
    "agent_matched_by_reference": {"target": True},
    "fallback_used": {"target": True} 
}

for flag, config in flags_config.items():
    csv_file = os.path.join(base_dir, f"eval_{flag}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ["Call ID"] + sources
        writer.writerow(header)
        
        totals = {src: 0 for src in sources}
        
        for call in calls:
            row = [call]
            for src in sources:
                json_path = os.path.join(base_dir, src.replace('/', os.sep), "index", f"{call}.summary.json")
                val = "N/A"
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                            val = data.get("flags", {}).get(flag, "N/A")
                            if val == config["target"]:
                                totals[src] += 1
                    except Exception as e:
                        val = "Error"
                row.append(val)
            writer.writerow(row)
            
        writer.writerow([])
        
        total_row = [f"Total (== {config['target']})"]
        for src in sources:
            total_row.append(totals[src])
        writer.writerow(total_row)
        
        pct_row = ["Percentage"]
        for src in sources:
            pct = (totals[src] / len(calls)) * 100 if len(calls) > 0 else 0
            pct_row.append(f"{pct:.1f}%")
        writer.writerow(pct_row)
        
    print(f"Created {csv_file}")

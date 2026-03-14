import os
import json

dir1 = r"c:\Users\jemai\projects\emotion-detection-analysis\diarizer\out_pyannote\1_refs\index"
dir2 = r"c:\Users\jemai\projects\emotion-detection-analysis\diarizer\output\index"

files1 = set(os.listdir(dir1))
files2 = set(os.listdir(dir2))

common_files = files1.intersection(files2)
print(f"Total files in dir1 (1_refs): {len(files1)}")
print(f"Number of common files: {len(common_files)}")

identical = 0
differing = 0
diff_examples = []

for f in common_files:
    path1 = os.path.join(dir1, f)
    path2 = os.path.join(dir2, f)
    try:
        with open(path1, "r") as f1, open(path2, "r") as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
            
            if data1 == data2:
                identical += 1
            else:
                differing += 1
                if len(diff_examples) < 2:
                    diffs = []
                    keys1 = set(data1.keys())
                    keys2 = set(data2.keys())
                    if keys1 != keys2:
                        diffs.append(f"Keys diff: 1_refs has {keys1 - keys2}, output has {keys2 - keys1}")
                    for k in keys1.intersection(keys2):
                        if data1[k] != data2[k]:
                            v1 = data1[k]
                            v2 = data2[k]
                            if isinstance(v1, list) and isinstance(v2, list):
                                diffs.append(f"Key '{k}' differs: 1_refs list len {len(v1)}, output list len {len(v2)}")
                            elif isinstance(v1, dict) and isinstance(v2, dict):
                                diffs.append(f"Key '{k}' differs in dict keys: {set(v1.keys()) ^ set(v2.keys())} or values")
                            else:
                                diffs.append(f"Key '{k}' differs: {v1} vs {v2}")
                    diff_examples.append({"file": f, "diffs": diffs})
    except Exception as e:
        print(f"Error reading {f}: {e}")

print(f"\nIdentical common files: {identical}")
print(f"Differing common files: {differing}")
if diff_examples:
    print("\nExamples of differences:")
    for ex in diff_examples:
        print(f"File: {ex['file']}")
        for d in ex['diffs']:
            print(f"  - {d}")

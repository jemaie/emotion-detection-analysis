import os
import shutil
import pandas as pd
from orchestrator import EmotionOrchestrator

# Setup dummy environment
if not os.path.exists("aufnahmen_test"):
    os.makedirs("aufnahmen_test")
    # Create dummy mp4 files
    open("aufnahmen_test/test1.mp4", "w").close()
    open("aufnahmen_test/test2.mp4", "w").close()
    open("aufnahmen_test/test3.mp4", "w").close()

# Mock ResultsLogger to use a test csv
test_csv = "results_test.csv"
if os.path.exists(test_csv):
    os.remove(test_csv)

orc = EmotionOrchestrator()
orc.logger.filename = test_csv # Override for test

# Check next file
print("--- Test 1: Get first file ---")
next_file = orc.get_next_file("aufnahmen_test")
print(f"Next file: {next_file}")
assert next_file.endswith("test1.mp4")

# Simulate logging
print("--- Test 2: Log result ---")
orc.save_feedback({"file": next_file, "openai": {}, "local": []}, "openai")

# Check next file again
print("--- Test 3: Get second file ---")
next_file_2 = orc.get_next_file("aufnahmen_test")
print(f"Next file: {next_file_2}")
assert next_file_2.endswith("test2.mp4")

print("Verification Successful!")

# Cleanup
# os.remove(test_csv)
# shutil.rmtree("aufnahmen_test")

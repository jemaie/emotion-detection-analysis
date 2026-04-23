import subprocess
import os
from pathlib import Path

FRAMEWORKS = ["russell", "gew", "plutchik", "willcox", "ekman"]

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def run_pipeline_for_dirs(segments_dir, phases_dir, analysis_dir):
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 0. Consistency Analysis
    run_cmd(f".venv\\Scripts\\python.exe analyze_0_consistency.py --segments_dir {segments_dir} --output_dir {analysis_dir} --phases_dir {phases_dir}")
    
    # 1. Aggregate Segments to Phases
    run_cmd(f".venv\\Scripts\\python.exe 2_aggregate_seg_ratings_to_phase_ratings.py --segments_dir {segments_dir} --output_dir {phases_dir}")
    
    # 2. Analyze Segments
    run_cmd(f".venv\\Scripts\\python.exe analyze_1_segments.py --segments_dir {segments_dir} --output_dir {analysis_dir}")
    
    # 3. Analyze Transitions
    run_cmd(f".venv\\Scripts\\python.exe analyze_2_transitions.py --segments_dir {segments_dir} --output_dir {analysis_dir}")
    
    # 4. Analyze Phases
    run_cmd(f".venv\\Scripts\\python.exe analyze_3_phases.py --output_dir {analysis_dir} --phases_dir {phases_dir}")

def main():
    base_out = Path("output")
    
    print(f"\n{'='*50}")
    print(f"RUNNING PIPELINE FOR BASE DATA")
    print(f"{'='*50}\n")
    
    base_segments_dir = base_out / "caller_segments"
    base_phases_dir = base_out / "caller_phases"
    base_analysis_dir = base_out / "analysis_results"
    
    run_pipeline_for_dirs(base_segments_dir, base_phases_dir, base_analysis_dir)
    
    fw_base_out = Path("output/emotion_frameworks")
    
    for fw in FRAMEWORKS:
        print(f"\n{'='*50}")
        print(f"RUNNING PIPELINE FOR FRAMEWORK: {fw.upper()}")
        print(f"{'='*50}\n")
        
        segments_dir = fw_base_out / fw / "caller_segments"
        phases_dir = fw_base_out / fw / "caller_phases"
        analysis_dir = fw_base_out / fw / "analysis_results"
        
        run_pipeline_for_dirs(segments_dir, phases_dir, analysis_dir)
        
    print("\nAll categorizations and analysis complete!")

if __name__ == "__main__":
    main()

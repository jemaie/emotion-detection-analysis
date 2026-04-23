import pandas as pd
from pathlib import Path

def generate_report():
    print("Generating Comprehensive V2 Taxonomy Report...")
    
    output_md = Path("output/analysis_I1_new_set.md")
    results_dir = Path("output/analysis_results")
    
    target_models = ['openai_realtime_1_5_ft_ns', 'openai_realtime_1_5_ft_ns_2', 'openai_realtime_1_5_ft_rp_ns', 'openai_realtime_1_5_ft_rp_ns_2']
    
    sections = []
    sections.append(f"# Comprehensive V2 Output Report: New Emotion Set")
    sections.append(f"This report filters the entire analysis pipeline exclusively for the `_ns` model configurations, measuring how well the new bounded taxonomy performs against fully normalized human consensus.")
    
    # ---------------- 0. Consistency ----------------
    cons_file = results_dir / "consistency_deep.csv"
    if cons_file.exists():
        df0 = pd.read_csv(cons_file)
        if 'Model Config' in df0.columns:
            m_df0 = df0[df0['Model Config'].isin(target_models)]
            if not m_df0.empty:
                sections.append("\n## 0. Inter-Run Consistency")
                sections.append("Measures how reliably the model produces the exact same emotion label across repeated runs.")
                sections.append("\n" + m_df0.to_markdown(index=False))
            else:
                sections.append(f"\n## 0. Inter-Run Consistency\nNo data available yet for the specified models.")
    
    # ---------------- 1. Segments ----------------
    seg_file = results_dir / "model_performance_segments_deep.csv"
    if seg_file.exists():
        df1 = pd.read_csv(seg_file)
        if 'Model' in df1.columns:
            m_df1 = df1[df1['Model'].isin(target_models)]
            if not m_df1.empty:
                sections.append("\n## 1. Segment-Level Ground Truth Accuracy")
                sections.append("Evaluates the model against the aggregated human consensus `maj_tiebroken` tag strictly per audio segment.")
                sections.append("\n" + m_df1.to_markdown(index=False, floatfmt=".3f"))
            else:
                sections.append(f"\n## 1. Segment-Level Ground Truth Accuracy\nNo data available yet for the specified models.")
                
    # ---------------- 2. Transitions ----------------
    trans_file = results_dir / "transitions_deep.csv"
    if trans_file.exists():
        df2 = pd.read_csv(trans_file)
        if 'Model' in df2.columns:
            # Transitions outputs model names as 'Model (openai_realtime_1_5_ft_ns)' so we need a substring search
            m_df2 = df2[df2['Model'].apply(lambda x: any(m in str(x) for m in target_models))]
            if not m_df2.empty:
                sections.append("\n## 2. Dynamic Trajectory Tracking")
                sections.append("Evaluates the model's ability to accurately capture the emotional dynamics of transitions compared to human trajectories.")
                sections.append("\n" + m_df2.to_markdown(index=False, floatfmt=".3f"))
            else:
                sections.append(f"\n## 2. Dynamic Trajectory Tracking\nNo data available yet for the specified models.")

    # ---------------- 3. Phases ----------------
    phase_file = results_dir / "phase_level_metrics.csv"
    if phase_file.exists():
        df3 = pd.read_csv(phase_file, sep=';')
        if 'Model' in df3.columns:
            m_df3 = df3[df3['Model'].isin(target_models)]
            if not m_df3.empty:
                sections.append("\n## 3. Top-Down Phase Emotion Capture")
                sections.append("Analyzes how well the model categorizes entire conversational phases across four analytical evaluation approaches (A: Peak, B: Phase Majority, C: Dist Overlap, D: Full Trajectory match).")
                sections.append("\n" + m_df3.to_markdown(index=False, floatfmt=".3f"))
            else:
                sections.append(f"\n## 3. Top-Down Phase Emotion Capture\nNo data available yet for the specified models.")

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(sections))
        
    print(f"Report successfully saved to {output_md}")

if __name__ == "__main__":
    generate_report()

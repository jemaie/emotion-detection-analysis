import pandas as pd
from pathlib import Path

def generate_report():
    print("Generating Comprehensive Emotion Frameworks Report...")
    
    frameworks = ['russell', 'gew', 'plutchik', 'willcox', 'ekman']
    models_of_interest = [
        'openai_realtime_1_5_ft',
        'openai_realtime_1_5_ft_rp',
        'openai_realtime_1_5_ft_ns', 
        'openai_realtime_1_5_ft_rp_ns', 
        'openai_realtime_1_5_ft_e', 
        'openai_realtime_1_5_ft_erp',
        'openai_realtime_1_5_ft_ns_2', 
        'openai_realtime_1_5_ft_rp_ns_2', 
        'openai_realtime_1_5_ft_e_2', 
        'openai_realtime_1_5_ft_erp_2'
    ]

    sections = []
    sections.append("# Analysis I2: Emotion Frameworks\n")
    sections.append("This report evaluates the performance of the models when mapped onto established psychological frameworks.\n")
    
    for fw in frameworks:
        sections.append(f"\n## Framework: {fw.upper()}\n")
        
        base = Path('output/emotion_frameworks') / fw / 'analysis_results'
        
        # ---------------- 0. Consistency ----------------
        cons_file = base / "consistency_deep.csv"
        if cons_file.exists():
            df0 = pd.read_csv(cons_file)
            if 'Model Config' in df0.columns:
                m_df0 = df0[df0['Model Config'].isin(models_of_interest)]
                if not m_df0.empty:
                    sections.append("\n### 0. Inter-Run Consistency")
                    sections.append("Measures how reliably the model produces the exact same emotion label across repeated runs.")
                    sections.append("\n" + m_df0.to_markdown(index=False))
                else:
                    sections.append(f"\n### 0. Inter-Run Consistency\nNo data available yet for the specified models.")
        
        # ---------------- 1. Segments ----------------
        seg_file = base / "model_performance_segments_deep.csv"
        if seg_file.exists():
            df1 = pd.read_csv(seg_file)
            if 'Model' in df1.columns:
                m_df1 = df1[df1['Model'].isin(models_of_interest)]
                if not m_df1.empty:
                    sections.append("\n### 1. Segment-Level Ground Truth Accuracy")
                    sections.append("Evaluates the model against the aggregated human consensus `maj_tiebroken` tag strictly per audio segment.")
                    sections.append("\n" + m_df1.to_markdown(index=False, floatfmt=".3f"))
                else:
                    sections.append(f"\n### 1. Segment-Level Ground Truth Accuracy\nNo data available yet for the specified models.")
                    
        # ---------------- 2. Transitions ----------------
        trans_file = base / "transitions_deep.csv"
        if trans_file.exists():
            df2 = pd.read_csv(trans_file)
            if 'Model' in df2.columns and 'From_Emotion' in df2.columns and 'To_Emotion' in df2.columns:
                m_df2 = df2[df2['Model'].apply(lambda x: any(m in str(x) for m in models_of_interest))]
                # Filter to only show stasis (From_Emotion == To_Emotion) to avoid 19MB file
                m_df2_stasis = m_df2[m_df2['From_Emotion'] == m_df2['To_Emotion']]
                
                if not m_df2_stasis.empty:
                    sections.append("\n### 2. Dynamic Trajectory Tracking (Stasis Only)")
                    sections.append("Evaluates the model's ability to accurately capture the emotional dynamics of transitions compared to human trajectories. Shows only stasis probabilities.")
                    sections.append("\n" + m_df2_stasis.to_markdown(index=False, floatfmt=".3f"))
                else:
                    sections.append(f"\n### 2. Dynamic Trajectory Tracking\nNo data available yet for the specified models.")

        # ---------------- 3. Phases ----------------
        phase_file = base / "phase_level_metrics.csv"
        if phase_file.exists():
            df3 = pd.read_csv(phase_file, sep=';', on_bad_lines='skip')
            if 'Model' in df3.columns:
                m_df3 = df3[df3['Model'].isin(models_of_interest)]
                if not m_df3.empty:
                    sections.append("\n### 3. Top-Down Phase Emotion Capture")
                    sections.append("Analyzes how well the model categorizes entire conversational phases across four analytical evaluation approaches.")
                    sections.append("\n" + m_df3.to_markdown(index=False, floatfmt=".3f"))
                else:
                    sections.append(f"\n### 3. Top-Down Phase Emotion Capture\nNo data available yet for the specified models.")
        
        sections.append("\n---\n")

    output_md = Path("output/analysis_I2_emotion_frameworks.md")
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(sections))
        
    print(f"Report successfully saved to {output_md}")

if __name__ == "__main__":
    generate_report()
 
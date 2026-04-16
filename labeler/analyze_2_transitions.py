import pandas as pd
import json
from pathlib import Path

INTENSITY_HIERARCHY = {
    'unusable': 0, 'other': 0, 'neutral': 0, 'calm': 0, 'happy': 0, 'uncertain': 0,
    'curious': 1, 'surprised': 1,
    'confused': 2, 'anxious': 2, 'fearful': 2, 'frustrated': 2,
    'angry': 3
}

def clean_emotion(e):
    if pd.isna(e): return None
    return str(e).lower().strip()

def resolve_consensus(emotions):
    emotions = [e for e in emotions if pd.notna(e)]
    if not emotions: return None
    counts = pd.Series(emotions).value_counts()
    candidates = counts[counts == counts.max()].index.tolist()
    if len(candidates) == 1: return candidates[0]
    best_candidate = candidates[0]
    max_intensity = -1
    for c in candidates:
        intensity = INTENSITY_HIERARCHY.get(c, 0)
        if intensity > max_intensity:
            max_intensity = intensity
            best_candidate = c
    return best_candidate

def calculate_transition_matrix(sequences):
    transitions = {}
    for seq in sequences:
        for i in range(len(seq) - 1):
            curr_state = seq[i]
            next_state = seq[i+1]
            if curr_state not in transitions:
                transitions[curr_state] = {}
            transitions[curr_state][next_state] = transitions[curr_state].get(next_state, 0) + 1
    return transitions

def print_transition_matrix(matrix, name):
    print(f"\n--- Transition Matrix: {name} ---")
    if not matrix:
        print("No transitions found.")
        return
        
    states = set(matrix.keys())
    for v in matrix.values(): states.update(v.keys())
    states = sorted(list(states))
    
    df = pd.DataFrame(index=states, columns=states).fillna(0).astype(int)
    for s1, targets in matrix.items():
        for s2, count in targets.items():
            df.loc[s1, s2] = count

    df_prob = df.div(df.sum(axis=1), axis=0).fillna(0)
    
    import tabulate
    print("Counts:")
    print(df.to_markdown())
    print("\nProbabilities:")
    print(df_prob.to_markdown(floatfmt=".2f"))

def analyze_transitions():
    print("=== Phase 2: Trajectory and Transition Analysis ===\n")
    
    target_models = [
        "openai_realtime_1_5_ft", "openai_realtime_1_5_ft_2",
        "openai_realtime_1_5_ft_e", "openai_realtime_1_5_ft_e_2",
        "openai_realtime_1_5_ft_erp", "openai_realtime_1_5_ft_erp_2"
    ]
    
    csv_path = Path("output/ratings_segments.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip')
    df['emotion'] = df['emotion'].apply(clean_emotion)
    df = df.dropna(subset=['emotion'])
    
    consensus_df = df.groupby(['conv_id', 'segment'])['emotion'].apply(lambda x: resolve_consensus(x.tolist())).reset_index()
    consensus_df.rename(columns={'emotion': 'human_consensus'}, inplace=True)
    consensus_df['segment'] = consensus_df['segment'].apply(lambda x: Path(x).stem)
    
    consensus_df.sort_values(by=['conv_id', 'segment'], inplace=True)
    
    human_seqs = []
    for conv_id, group in consensus_df.groupby('conv_id'):
        human_seqs.append(group['human_consensus'].tolist())
        
    human_matrix = calculate_transition_matrix(human_seqs)
    print_transition_matrix(human_matrix, "Human Consensus")
    
    segments_dir = Path("output/caller_segments")
    model_predictions = []
    for json_file in segments_dir.rglob("*.json"):
        conv_id = json_file.parent.name
        segment = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            preds = data.get("predictions", {})
            row = {"conv_id": conv_id, "segment": segment}
            has_data = False
            for t_model in target_models:
                pred_obj = preds.get(t_model)
                if pred_obj:
                    if isinstance(pred_obj, dict):
                        emotion = str(pred_obj.get("emotion", "unknown")).lower().strip()
                    else:
                        emotion = str(pred_obj).lower().strip()
                    row[t_model] = emotion
                    has_data = True
            if has_data:
                model_predictions.append(row)
                
    if not model_predictions:
        print("No predictions found for target models")
        return
        
    model_df = pd.DataFrame(model_predictions)
    
    # Methodological Fix: Merge with consensus_df to ensure identical sequence length and segments
    merged_df = pd.merge(consensus_df[['conv_id', 'segment']], model_df, on=['conv_id', 'segment'], how='inner')
    merged_df.sort_values(by=['conv_id', 'segment'], inplace=True)
    
    for t_model in target_models:
        if t_model not in merged_df.columns:
            continue
            
        model_seqs = []
        for conv_id, group in merged_df.groupby('conv_id'):
            # Filter NaNs for missing runs if any
            seq = group[t_model].dropna().tolist()
            if seq:
                model_seqs.append(seq)
                
        model_matrix = calculate_transition_matrix(model_seqs)
        print_transition_matrix(model_matrix, f"Model ({t_model})")

if __name__ == "__main__":
    analyze_transitions()

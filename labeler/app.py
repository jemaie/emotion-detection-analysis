import os
import pandas as pd
import streamlit as st
from tqdm import tqdm
from storage import get_all_evaluations, write_evaluation, read_runner_state

# Import runner_state to trick Streamlit's file watcher into monitoring it
try:
    import runner_state
except ImportError:
    print("Failed to import runner_state.py")

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diarizer", "output", "caller_concat")

st.set_page_config(page_title="Emotion Labeler", layout="wide")

def main():
    # st.title("🗣️ Emotion Labeler")
    
    evaluations = get_all_evaluations()
    state = read_runner_state()
    
    local_state = state.get("local", {})
    openai_state = state.get("openai", {})
    
    # --- Top Section: Dashboard and Status ---
    total_files = state.get("total_files", max(local_state.get("total_files", 0), openai_state.get("total_files", 0)))
    rated = len([e for e in evaluations if e.get("user_rating")])
    
    model_names = ["ehcalabres", "speechbrain", "hubert_german", "emotion2vec", "openai_realtime"]
    
    # Create columns: 1 for each model + 2 for (Rated, VRAM)
    cols = st.columns(len(model_names) + 2)
    
    cols[0].metric("Rated", f"{rated} / {total_files}")

    for i, m_name in enumerate(model_names):
        m_processed = len([e for e in evaluations if m_name in e.get("predictions", {})])
        cols[i + 1].metric(f"Processed by {m_name}", f"{m_processed} / {total_files}")

    cols[-1].metric("VRAM (alloc/res) in MB", f"{state.get('vram_mb', 0)} / {state.get('vram_reserved_mb', 0)}")

    @st.fragment(run_every="1s")
    def render_status():
        # Read the latest state periodically
        current_state = read_runner_state()
        active_workers = 0
        
        for worker_key in ["local", "openai"]:
            worker_state = current_state.get(worker_key, {})
            if worker_state.get("status") == "processing":
                active_workers += 1
                tqdm_dict = worker_state.get("tqdm_dict", {})
                completed = tqdm_dict.get("n", 0)
                total = tqdm_dict.get("total", 0)
                
                if tqdm_dict:
                    tqdm_dict["bar_format"] = "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                    tqdm_str = tqdm.format_meter(**tqdm_dict) 
                else:
                    tqdm_str = "..."
                    
                col1, col2 = st.columns([3, 1])
                with col1:
                    file_name = worker_state.get('file')
                    st.markdown(f"Running model **{worker_state.get('model')}** on **{file_name}**", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div style='text-align: right; font-family: monospace;'>{tqdm_str}</div>", unsafe_allow_html=True)
                
                pct = completed / total if total > 0 else 0.0
                st.progress(min(max(pct, 0.0), 1.0))
                
        if active_workers == 0:
            st.warning("**Pipeline Idle:** No model currently running.")

    render_status()
            
    st.markdown("---")
            
    # --- Bottom Section: Rating Interface ---
    st.header("🗣️ Emotion Labeler")
    
    # Filter for processed but unrated
    pending = [e for e in evaluations if e.get("status") == "processed" and not e.get("user_rating")]
    
    if not pending:
        st.info("All caught up! No new files ready for rating.")
    else:
        current_eval = pending[0]
        filename = current_eval["filename"]
        audio_path = os.path.join(AUDIO_DIR, filename)
        
        st.subheader(f"Current File: {filename}")
        
        if os.path.exists(audio_path):
            st.audio(audio_path)
        else:
            st.error("Audio file missing from the directory.")
        
        st.markdown("### Model Predictions")
        preds = current_eval.get("predictions", {})
        
        if preds:
            df = pd.DataFrame([
                {"Model": model, "Predicted Emotion": data.get("emotion"), "Confidence": data.get("confidence")}
                for model, data in preds.items()
            ])
            st.dataframe(df, hide_index=True)
        else:
            st.info("No predictions found for this file.")
            
        st.markdown("### User Rating")
        
        with st.form(key="rating_form"):
            true_emotion = st.selectbox("True Emotion", ["angry", "sad", "happy", "fear", "neutral", "surprise", "disgust", "other"])
            notes = st.text_area("Notes (optional)")
            submit = st.form_submit_button("Save Rating")
            
            if submit:
                # Update the json
                current_eval["user_rating"] = {
                    "true_emotion": true_emotion,
                    "notes": notes
                }
                write_evaluation(audio_path, current_eval)
                st.success("Rating saved!")
                st.rerun()

    # --- Optional: Rating History ---
    if rated > 0:
        with st.expander("View Rating History"):
            history_data = []
            for e in evaluations:
                if e.get("user_rating"):
                    history_data.append({
                        "File": e["filename"],
                        "True Emotion": e["user_rating"]["true_emotion"],
                        "Notes": e["user_rating"]["notes"]
                    })
            st.dataframe(pd.DataFrame(history_data), hide_index=True)

if __name__ == "__main__":
    main()

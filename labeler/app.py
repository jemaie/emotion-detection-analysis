from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from tqdm import tqdm
from storage import get_all_evaluations, write_evaluation, read_runner_state, read_evaluation, get_evaluation_lock

# Import runner_state to trick Streamlit's file watcher into monitoring it
try:
    import runner_state
except ImportError:
    print("Failed to import runner_state.py")

CONCAT_DIR = Path("data/caller_concat_mixed")
SEGMENTS_DIR = Path("data/caller_segments_24kHz")

st.set_page_config(page_title="Emotion Labeler", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    [data-testid="stHeader"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {
        padding-top: 1.5rem;
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1.5rem;
    }
    </style>""",
    unsafe_allow_html=True
)

def main():
    # st.title("🗣️ Emotion Labeler")
    
    evaluations = get_all_evaluations()
    state = read_runner_state()
    
    # --- Top Section: Dashboard and Status ---
    total_concat = state.get("total_concat_files", 0)
    total_segments = state.get("total_segments", 0)
    
    evals_concat = evaluations.get("concat", [])
    evals_segments = evaluations.get("segments", [])
    
    model_names = ["openai_realtime_1_5_ft", "openai_realtime_1_5_ft_2", "openai_realtime_1_5_ft_e", "openai_realtime_1_5_ft_e_2", "openai_realtime_1_5_ft_erp", "openai_realtime_1_5_ft_erp_2",
                    "ehcalabres/wav2vec2", "speechbrain/wav2vec2", "superb/wav2vec2_base", "superb/wav2vec2_large",
                    "superb/hubert_base", "superb/hubert_large", "iic/emotion2vec_base", "iic/emotion2vec_large"]
    
    # Create columns: 1 for each model + 2 for (Rated, VRAM)
    cols1 = st.columns([1] * (len(model_names) + 2))
    rated_concat = len([e for e in evals_concat if e.get("user_rating")])
    cols1[0].metric("Rated", f"{rated_concat} / {total_concat}")
    for i, m_name in enumerate(model_names):
        m_processed = len([e for e in evals_concat if m_name in e.get("predictions", {})])
        cols1[i + 1].metric(f"{m_name}", f"{m_processed} / {total_concat}")
    cols1[-1].metric("VRAM (alloc/res)", f"{state.get('vram_mb', 0)}")
    
    # Second row for segment metrics
    cols2 = st.columns([1] * (len(model_names) + 2))
    rated_segments = len([e for e in evals_segments if e.get("user_rating")])
    cols2[0].metric(label="-", value=f"{rated_segments} / {total_segments}", label_visibility="collapsed")
    for i, m_name in enumerate(model_names):
        m_processed_seg = len([e for e in evals_segments if m_name in e.get("predictions", {})])
        cols2[i + 1].metric(label="-", value=f"{m_processed_seg} / {total_segments}", label_visibility="collapsed")
    cols2[-1].metric(label="-", value=f"{state.get('vram_reserved_mb', 0)}", label_visibility="collapsed")

    @st.fragment(run_every="1s")
    def render_status():
        # Read the latest state periodically
        current_state = read_runner_state()
        
        active_workers = 0
        for worker_key in ["local", "openai"]:
            worker_state = current_state.get(worker_key, {})
            if worker_state.get("status") in ["idle", "processing"]:
                active_workers += 1

        if active_workers > 0:
            for worker_key in ["local", "openai"]:
                worker_state = current_state.get(worker_key, {})
                col1, col2 = st.columns([3, 1])
                
                status = worker_state.get("status", "stopped")
                if status == "processing":
                    tqdm_dict = worker_state.get("tqdm_dict", {})
                    completed = tqdm_dict.get("n", 0)
                    total = tqdm_dict.get("total", 0)
                    
                    if tqdm_dict:
                        tqdm_dict["bar_format"] = "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                        tqdm_str = tqdm.format_meter(**tqdm_dict) 
                    else:
                        tqdm_str = "..."
                        
                    with col1:
                        file_name = worker_state.get('file')
                        is_segment = worker_state.get('is_segment', False)
                        if is_segment:
                            st.markdown(f"Running model **{worker_state.get('model')}** on **{file_name}** [running on segments]", unsafe_allow_html=True)
                        else:
                            st.markdown(f"Running model **{worker_state.get('model')}** on **{file_name}**", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div style='text-align: right; font-family: monospace;'>{tqdm_str}</div>", unsafe_allow_html=True)
                    
                    pct = completed / total if total > 0 else 0.0
                    st.progress(min(max(pct, 0.0), 1.0))
                else:
                    worker_display = "Local Models" if worker_key == "local" else "OpenAI Realtime"
                    display_status = "Idle / Loading" if status == "idle" else "Stopped"
                    with col1:
                        st.markdown(f"**{worker_display}**: {display_status}", unsafe_allow_html=True)
                    with col2:
                        st.markdown("<div style='text-align: right; font-family: monospace;'>-</div>", unsafe_allow_html=True)
                    st.progress(0.0)

    render_status()
    
    # Group concat evaluations by conv_id
    conv_dict = {e.get("conv_id"): e for e in evals_concat if e.get("conv_id")}
    
    if not conv_dict:
        st.warning("No conversations available yet.")
    else:
        # Determine if a conversation is fully rated (concat + all its segments)
        def is_conv_fully_rated(c_id):
            c_eval = conv_dict[c_id]
            if not c_eval.get("user_rating"): return False
            c_segs = [s for s in evals_segments if s.get("conv_id") == c_id]
            for s in c_segs:
                if not s.get("user_rating"): return False
            return True

        conv_options = list(conv_dict.keys())
        # Sort so pending comes first, then sort alphabetically
        conv_options.sort(key=lambda x: (is_conv_fully_rated(x), x))
        
        if "selected_conv_id" not in st.session_state or st.session_state.selected_conv_id not in conv_options:
            st.session_state.selected_conv_id = conv_options[0]
            
        curr_idx = conv_options.index(st.session_state.selected_conv_id)
        
        def prev_conv():
            idx = conv_options.index(st.session_state.selected_conv_id)
            if idx > 0:
                st.session_state.selected_conv_id = conv_options[idx - 1]
                st.session_state.pop("selected_seg_filename", None)
                
        def next_conv():
            idx = conv_options.index(st.session_state.selected_conv_id)
            if idx < len(conv_options) - 1:
                st.session_state.selected_conv_id = conv_options[idx + 1]
                st.session_state.pop("selected_seg_filename", None)

        def on_conv_change():
            st.session_state.pop("selected_seg_filename", None)
            
        # Inject CSS to force Streamlit's native alerts to be compact and match button height
        st.markdown(
            """<style>
            [data-testid="stAlertContainer"] {
                height: 40px !important;
                padding-top: 0px !important;
                padding-bottom: 0px !important;
                display: flex !important;
                align-items: center !important;
            }
            /* The inner wrapper that contains the icon and text */
            [data-testid="stAlertContainer"] div[role="alert"] {
                padding: 0px !important;
                display: flex !important;
                align-items: center !important;
                height: 100% !important;
            }
            /* Fix the Markdown container misaligning text downwards */
            [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] {
                display: flex !important;
                align-items: center !important;
                height: auto !important;
                min-height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            /* Remove the default 1rem bottom margin added by Streamlit markdown */
            [data-testid="stAlertContainer"] p {
                margin: 0px !important;
                line-height: normal !important;
            }
            </style>""", 
            unsafe_allow_html=True
        )

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        # --- LEFT: Concat Column ---
        with col1:
            st.markdown("### Conversation (Full)")
            
            c_eval = conv_dict[st.session_state.selected_conv_id]
            ur_conv = c_eval.get("user_rating") or {}
            
            # Selector with arrows and status
            cc1, cc2, cc3, cc4 = st.columns([1, 1, 4, 2])
            with cc1: st.button("⏪", key="p_conv", on_click=prev_conv, disabled=curr_idx == 0, width='stretch')
            with cc2: st.button("⏩", key="n_conv", on_click=next_conv, disabled=curr_idx == len(conv_options) - 1, width='stretch')
            with cc3:
                st.selectbox(
                    "Select Conversation", 
                    conv_options, 
                    key="selected_conv_id",
                    on_change=on_conv_change,
                    label_visibility="collapsed"
                )
            with cc4:
                if not ur_conv:
                    st.info("Pending", icon="⏳")
                elif ur_conv.get("true_emotion") == "unusable":
                    st.error("Unusable", icon="❌")
                elif ur_conv.get("true_emotion") == "uncertain":
                    st.warning("Uncertain", icon="⚠️")
                else:
                    st.success(f"{ur_conv.get('true_emotion').title()}", icon="✅")
            c_filename = c_eval["filename"]
            c_audio_path = CONCAT_DIR / c_filename
            
            if c_audio_path.exists():
                st.audio(str(c_audio_path))
            else:
                st.error("Audio file missing.")
                
            # st.markdown("#### Model Predictions")
            if not c_eval.get("user_rating"):
                st.info("💡 Please submit your rating first before viewing the models' predictions.")
            else:
                c_preds = c_eval.get("predictions", {})
                if c_preds:
                    df_rows = []
                    for model_n, data in c_preds.items():
                        row = {"Model": model_n}
                        scores = list(data.get("scores", {}).items())
                        scores.sort(key=lambda x: x[1], reverse=True)
                        if scores:
                            row["1st Em"] = scores[0][0] if len(scores) > 0 else None
                            row["1st Conf"] = str(round(scores[0][1], 2)) if len(scores) > 0 else None
                            row["2nd Em"] = scores[1][0] if len(scores) > 1 else None
                            row["2nd Conf"] = str(round(scores[1][1], 2)) if len(scores) > 1 else None
                            row["3rd Em"] = scores[2][0] if len(scores) > 2 else None
                            row["3rd Conf"] = str(round(scores[2][1], 2)) if len(scores) > 2 else None
                        else:
                            row["1st Em"] = data.get("emotion")
                            conf = data.get("confidence")
                            conf_val = round(conf, 2) if conf is not None else data.get("reason") if data.get("emotion") else None
                            row["1st Conf"] = str(conf_val) if conf_val is not None else None
                            row["2nd Em"], row["2nd Conf"], row["3rd Em"], row["3rd Conf"] = None, None, None, None
                            
                        df_rows.append(row)
                    _pred_col_cfg = {
                        "1st Conf": st.column_config.TextColumn(width=300),
                    }
                    st.dataframe(pd.DataFrame(df_rows), hide_index=True, column_config=_pred_col_cfg, width="stretch")
                else:
                    st.warning("No predictions found for this file.")
                
            # st.markdown("#### User Rating")
            with st.form(key="concat_rating_form"):
                ur = c_eval.get("user_rating") or {}
                emotions = ["neutral", "frustrated", "angry", "sad", "happy", "fear", "surprise", "disgust", "other"]
                def_idx = emotions.index(ur.get("true_emotion")) if ur.get("true_emotion") in emotions else 0
                
                fc1, fc2, fc3, fc4 = st.columns([2, 8, 2, 2])
                with fc1: submit = st.form_submit_button("Save Rating", width='stretch')
                with fc2: true_emotion = st.selectbox("True Emotion", emotions, index=def_idx, label_visibility="collapsed")
                with fc3: btn_uncertain_conv = st.form_submit_button("Uncertain", width='stretch')
                with fc4: btn_unusable_conv = st.form_submit_button("Unusable", width='stretch')
                
                notes = st.text_area("Notes (optional)", value=ur.get("notes", ""))
                
                if btn_unusable_conv:
                    with get_evaluation_lock(c_audio_path):
                        locked_eval = read_evaluation(c_audio_path)
                        locked_eval["user_rating"] = {"true_emotion": "unusable", "notes": notes}
                        write_evaluation(c_audio_path, locked_eval)
                    st.session_state["_conv_flash"] = "Conversation marked as unusable!"
                    st.rerun()
                elif btn_uncertain_conv:
                    with get_evaluation_lock(c_audio_path):
                        locked_eval = read_evaluation(c_audio_path)
                        locked_eval["user_rating"] = {"true_emotion": "uncertain", "notes": notes}
                        write_evaluation(c_audio_path, locked_eval)
                    st.session_state["_conv_flash"] = "Conversation marked as uncertain!"
                    st.rerun()
                elif submit:
                    with get_evaluation_lock(c_audio_path):
                        locked_eval = read_evaluation(c_audio_path)
                        locked_eval["user_rating"] = {"true_emotion": true_emotion, "notes": notes}
                        write_evaluation(c_audio_path, locked_eval)
                    st.session_state["_conv_flash"] = "Rating saved!"
                    st.rerun()
            if flash := st.session_state.pop("_conv_flash", None):
                st.success(flash)

        # --- RIGHT: Segment Column ---
        with col2:
            st.markdown("### Segment")
            c_segs = [s for s in evals_segments if s.get("conv_id") == st.session_state.selected_conv_id]
            
            if not c_segs:
                st.warning("No segments found for this conversation.")
            else:
                c_segs.sort(key=lambda x: x.get("filename", ""))
                seg_options = [s.get("filename") for s in c_segs]
                
                if "selected_seg_filename" not in st.session_state or st.session_state.selected_seg_filename not in seg_options:
                    st.session_state.selected_seg_filename = seg_options[0]
                    
                curr_seg_idx = seg_options.index(st.session_state.selected_seg_filename)
                
                def prev_seg():
                    idx = seg_options.index(st.session_state.selected_seg_filename)
                    if idx > 0: st.session_state.selected_seg_filename = seg_options[idx - 1]
                    
                def next_seg():
                    idx = seg_options.index(st.session_state.selected_seg_filename)
                    if idx < len(seg_options) - 1: st.session_state.selected_seg_filename = seg_options[idx + 1]

                s_eval = next(s for s in c_segs if s.get("filename") == st.session_state.selected_seg_filename)
                s_ur = s_eval.get("user_rating") or {}

                sc1, sc2, sc3, sc4 = st.columns([1, 1, 4, 2])
                with sc1: st.button("◀️", key="p_seg", on_click=prev_seg, disabled=curr_seg_idx == 0, width='stretch')
                with sc2: st.button("▶️", key="n_seg", on_click=next_seg, disabled=curr_seg_idx == len(seg_options) - 1, width='stretch')
                with sc3:
                    st.selectbox(
                        "Select Segment",
                        seg_options,
                        key="selected_seg_filename",
                        label_visibility="collapsed"
                    )
                with sc4:
                    if not s_ur:
                        st.info("Pending", icon="⏳")
                    elif s_ur.get("true_emotion") == "unusable":
                        st.error("Unusable", icon="❌")
                    elif s_ur.get("true_emotion") == "uncertain":
                        p_str = f" ({s_ur.get('phase')})" if s_ur.get("phase") else ""
                        st.warning(f"Uncertain{p_str}", icon="⚠️")
                    else:
                        p_str = f" ({s_ur.get('phase')})" if s_ur.get("phase") else ""
                        st.success(f"{s_ur.get('true_emotion').title()}{p_str}", icon="✅")
                
                s_audio_path = SEGMENTS_DIR / st.session_state.selected_conv_id / st.session_state.selected_seg_filename
                
                if s_audio_path.exists():
                    st.audio(str(s_audio_path))
                else:
                    st.error("Audio file missing.")
                    
                # st.markdown("#### Model Predictions")
                if not s_eval.get("user_rating"):
                    st.info("💡 Please submit your rating first before viewing the models' predictions.")
                else:
                    s_preds = s_eval.get("predictions", {})
                    if s_preds:
                        df_rows = []
                        for model_n, data in s_preds.items():
                            row = {"Model": model_n}
                            scores = list(data.get("scores", {}).items())
                            scores.sort(key=lambda x: x[1], reverse=True)
                            if scores:
                                row["1st Em"] = scores[0][0] if len(scores) > 0 else None
                                row["1st Conf"] = str(round(scores[0][1], 2)) if len(scores) > 0 else None
                                row["2nd Em"] = scores[1][0] if len(scores) > 1 else None
                                row["2nd Conf"] = str(round(scores[1][1], 2)) if len(scores) > 1 else None
                                row["3rd Em"] = scores[2][0] if len(scores) > 2 else None
                                row["3rd Conf"] = str(round(scores[2][1], 2)) if len(scores) > 2 else None
                            else:
                                row["1st Em"] = data.get("emotion")
                                conf = data.get("confidence")
                                conf_val = round(conf, 2) if conf is not None else data.get("reason") if data.get("emotion") else None
                                row["1st Conf"] = str(conf_val) if conf_val is not None else None
                                row["2nd Em"], row["2nd Conf"], row["3rd Em"], row["3rd Conf"] = None, None, None, None
                                
                            df_rows.append(row)
                        _pred_col_cfg = {
                            "1st Conf": st.column_config.TextColumn(width=300),
                        }
                        st.dataframe(pd.DataFrame(df_rows), hide_index=True, column_config=_pred_col_cfg, width="stretch")
                    else:
                        st.warning("No predictions found.")
                    
                # st.markdown("#### User Rating")
                with st.form(key="segment_rating_form"):
                    s_ur = s_eval.get("user_rating") or {}
                    emotions = ["neutral", "frustrated", "angry", "sad", "happy", "fear", "surprise", "disgust", "other"]
                    def_idx = emotions.index(s_ur.get("true_emotion")) if s_ur.get("true_emotion") in emotions else 0
                    phases = ["Phase 1", "Phase 2", "Phase 3", "N/A"]
                    p_idx = phases.index(s_ur.get("phase")) if s_ur.get("phase") in phases else 3
                    
                    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 4, 4, 2, 2])
                    with fc1: submit_seg = st.form_submit_button("Save Rating", width='stretch')
                    with fc2: s_emotion = st.selectbox("True Emotion", emotions, index=def_idx, label_visibility="collapsed")
                    with fc3: s_phase = st.selectbox("Conversation Phase", phases, index=p_idx, label_visibility="collapsed")
                    with fc4: btn_uncertain = st.form_submit_button("Uncertain", width='stretch')
                    with fc5: btn_unusable = st.form_submit_button("Unusable", width='stretch')
                    
                    s_notes = st.text_area("Notes (optional)", value=s_ur.get("notes", ""))
                    
                    if btn_unusable:
                        with get_evaluation_lock(s_audio_path):
                            locked_eval = read_evaluation(s_audio_path)
                            locked_eval["user_rating"] = {
                                "true_emotion": "unusable", 
                                "phase": "N/A",
                                "notes": s_notes
                            }
                            write_evaluation(s_audio_path, locked_eval)
                        st.session_state["_seg_flash"] = "Segment marked as unusable."
                        st.rerun()
                    elif btn_uncertain:
                        with get_evaluation_lock(s_audio_path):
                            locked_eval = read_evaluation(s_audio_path)
                            locked_eval["user_rating"] = {
                                "true_emotion": "uncertain", 
                                "phase": s_phase,
                                "notes": s_notes
                            }
                            write_evaluation(s_audio_path, locked_eval)
                        st.session_state["_seg_flash"] = "Segment marked as uncertain."
                        st.rerun()
                    elif submit_seg:
                        with get_evaluation_lock(s_audio_path):
                            locked_eval = read_evaluation(s_audio_path)
                            locked_eval["user_rating"] = {
                                "true_emotion": s_emotion, 
                                "phase": s_phase,
                                "notes": s_notes
                            }
                            write_evaluation(s_audio_path, locked_eval)
                        st.session_state["_seg_flash"] = "Rating saved!"
                        st.rerun()
                if flash := st.session_state.pop("_seg_flash", None):
                    st.success(flash)

        # --- Aggregate Table ---
        if c_segs:
            st.markdown("---")
            st.markdown("### Aggregated Segment Predictions")
            
            agg_rows = []
            for s in c_segs:
                row = {"Segment": s.get("filename")}
                
                u_rating = s.get("user_rating") or {}
                row["[Phase]"] = u_rating.get("phase", "-")
                row["[Notes]"] = u_rating.get("notes", "-")
                row["[User]"] = u_rating.get("true_emotion", "-")
                
                s_preds = s.get("predictions", {})
                
                for model_name in model_names:
                    if not u_rating:
                        row[model_name] = "Waiting for rating..."
                    else:
                        pred_data = s_preds.get(model_name, {})
                        scores = list(pred_data.get("scores", {}).items())
                        scores.sort(key=lambda x: x[1], reverse=True)
                        
                        if scores:
                            row[model_name] = f"{scores[0][0]} ({scores[0][1]:.2f})" if len(scores) > 0 else "-"
                        else:
                            if pred_data.get('emotion'):
                                conf = pred_data.get('confidence')
                                if conf is not None:
                                    base_pred = f"{pred_data.get('emotion')} ({conf:.2f})"
                                else:
                                    base_pred = f"{pred_data.get('emotion')} | {pred_data.get('reason', '-')}"
                            else:
                                base_pred = "-"
                            row[model_name] = base_pred
                        
                agg_rows.append(row)
                
            agg_df = pd.DataFrame(agg_rows)
            agg_df = agg_df.set_index("Segment").T
            agg_df.index.name = "Metric"
            
            def highlight_selected(s):
                selected = st.session_state.get("selected_seg_filename")
                if s.name == selected:
                    return ['background-color: rgba(255, 255, 0, 0.2)'] * len(s)
                return [''] * len(s)
                
            styled_df = agg_df.style.apply(highlight_selected, axis=0)
            st.dataframe(styled_df, width='stretch', height=(len(agg_df) + 1) * 35 + 3)
            
    all_evaluations_list = evals_concat + evals_segments

    if rated_concat + rated_segments > 0:
        with st.expander("View Rating History"):
            history_data = []
            for e in all_evaluations_list:
                if e.get("user_rating"):
                    history_data.append({
                        "File": e["filename"],
                        "True Emotion": e["user_rating"]["true_emotion"],
                        "Notes": e["user_rating"]["notes"]
                    })
            st.dataframe(pd.DataFrame(history_data), hide_index=True)
            
    components.html(
        """
        <script>
        const doc = window.parent.document;
        
        const oldScript = doc.getElementById('key-listener');
        if (oldScript) oldScript.remove();
        
        const script = doc.createElement('script');
        script.id = 'key-listener';
        script.innerHTML = `
            if (window.myKeyListener) {
                window.removeEventListener('keydown', window.myKeyListener, {capture: true});
            }
            window.myKeyListener = function(e) {
                if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
                
                if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    let target = '';
                    if (e.key === 'ArrowLeft') target = '◀️';
                    if (e.key === 'ArrowRight') target = '▶️';
                    if (e.key === 'ArrowUp') target = '⏪';
                    if (e.key === 'ArrowDown') target = '⏩';
                    
                    const btn = Array.from(document.querySelectorAll('button p')).find(el => el.textContent === target);
                    if (btn && !btn.parentElement.disabled) {
                        btn.parentElement.click();
                    }
                }
                
                if (e.key === ' ') {
                    e.preventDefault();
                    e.stopPropagation();
                    const audios = document.querySelectorAll('audio');
                    if (audios.length > 0) {
                        let targetAudio = null;
                        if (e.shiftKey) {
                            // Shift+Space for Conversation (First audio)
                            targetAudio = audios[0];
                        } else {
                            // Space for Segment (Last audio)
                            targetAudio = audios[audios.length - 1];
                        }
                        
                        if (targetAudio) {
                            if (targetAudio.paused) targetAudio.play();
                            else targetAudio.pause();
                        }
                    }
                }
            };
            window.addEventListener('keydown', window.myKeyListener, {capture: true});
            
            setTimeout(function() {
                function styleBtn(btn, bg, border, color) {
                    btn.style.backgroundColor = bg;
                    btn.style.borderColor = border;
                    btn.style.color = color;
                    btn.style.fontWeight = '400';
                    btn.addEventListener('mouseenter', () => btn.style.filter = 'brightness(1.4)');
                    btn.addEventListener('mouseleave', () => btn.style.filter = 'brightness(1.0)');
                }
                
                // Match Uncertain button to Warning Banner
                Array.from(document.querySelectorAll('button p'))
                    .filter(el => el.textContent.trim() === 'Uncertain')
                    .forEach(p => { const btn = p.closest('button'); if (btn) styleBtn(btn, 'rgba(255, 189, 69, 0.2)', 'rgba(255, 189, 69, 0.5)', 'rgb(255, 189, 69)'); });
                
                // Match Unusable button to Error Banner
                Array.from(document.querySelectorAll('button p'))
                    .filter(el => el.textContent.trim() === 'Unusable')
                    .forEach(p => { const btn = p.closest('button'); if (btn) styleBtn(btn, 'rgba(255, 75, 75, 0.2)', 'rgba(255, 75, 75, 0.5)', 'rgb(255, 75, 75)'); });
            }, 50);
        `;
        doc.head.appendChild(script);
        </script>
        """,
        height=0,
        width=0,
    )

if __name__ == "__main__":
    main()

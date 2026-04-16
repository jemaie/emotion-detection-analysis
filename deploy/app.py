"""
Emotion Crowd-Rating App

A Streamlit app that lets multiple raters listen to conversation audio
and rate the perceived emotion. Designed to be deployed on Coolify.

Each rater enters their name, then works through conversations and segments,
rating each one. Model predictions are intentionally hidden to prevent bias.
"""

import csv
import io
import streamlit as st
import streamlit.components.v1 as components
from storage import (
    load_manifest,
    read_rating,
    write_rating,
    get_rater_progress,
    get_all_rater_names,
    delete_rater_ratings,
    ADMIN_NAME,
    CONCAT_DIR,
    SEGMENTS_DIR
)

st.set_page_config(page_title="Emotion Rater", layout="wide")
st.markdown(
    """<style>
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    [data-testid="stHeader"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container { padding-top: 1.5rem; }
    [data-testid="stAppViewBlockContainer"] { padding-top: 1.5rem; }

    /* Compact alert banners */
    [data-testid="stAlertContainer"] {
        height: 40px !important;
        padding-top: 0 !important; padding-bottom: 0 !important;
        display: flex !important; align-items: center !important;
    }
    [data-testid="stAlertContainer"] div[role="alert"] {
        padding: 0 !important; display: flex !important;
        align-items: center !important; height: 100% !important;
    }
    [data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] {
        display: flex !important; align-items: center !important;
        height: auto !important; min-height: 0 !important;
        margin: 0 !important; padding: 0 !important;
    }
    [data-testid="stAlertContainer"] p {
        margin: 0 !important; line-height: normal !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-size: clamp(0.55rem, 0.9vw, 0.875rem) !important;
    }
    /* Prevent line breaks in form buttons and scale font dynamically */
    [data-testid="stFormSubmitButton"] button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-size: clamp(0.55rem, 0.9vw, 0.875rem) !important;
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# ── Rater identification ─────────────────────────────────────────────

def get_rater_name() -> str | None:
    """Prompt the user for their name (stored in session state)."""
    if "rater_name" not in st.session_state or not st.session_state.rater_name:
        st.markdown("## 🗣️ Emotion Rating Tool")
        st.markdown(
            "Welcome! Please enter your name below to start rating conversations. "
            "Your task is to listen to each audio clip and select the emotion you perceive."
        )
        with st.form("rater_login"):
            name = st.text_input("Your Name", placeholder="e.g. Max Mustermann")
            submitted = st.form_submit_button("Start Rating", type="primary")
            if submitted and name.strip():
                st.session_state.rater_name = name.strip()
                st.rerun()
            elif submitted:
                st.error("Please enter a name.")
        return None
    return st.session_state.rater_name


def main():
    rater = get_rater_name()
    if not rater:
        return

    manifest = load_manifest()
    if not manifest:
        st.error("No conversations found. Make sure `data/conversations.json` exists.")
        return

    progress = get_rater_progress(rater, manifest)

    # ── Top bar: progress + rater info ────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])
    c1.metric("Conversations Rated", f"{progress['concat_rated']} / {progress['concat_total']}")
    c2.metric("Segments Rated", f"{progress['segments_rated']} / {progress['segments_total']}")
    with c3:
        st.markdown(f"**Rater:** {rater}")
        if st.button("Switch Rater", width='stretch'):
            st.session_state.rater_name = ""
            st.session_state.pop("selected_conv_id", None)
            st.session_state.pop("selected_seg_filename", None)
            st.rerun()

    st.markdown("---")

    # ── Build conv dict ───────────────────────────────────────────
    # Key by anon_id for display; keep conv_id for ratings storage
    conv_dict = {c["anon_id"]: c for c in manifest}
    conv_options = list(conv_dict.keys())

    # Stable alphabetical order (no re-sorting on rating)
    conv_options.sort()

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

    # ── Two-column layout ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    # === LEFT: Full Conversation ===
    with col1:
        st.markdown("### Conversation (Full)")
        anon_id = st.session_state.selected_conv_id
        conv_data = conv_dict[anon_id]
        conv_id = conv_data["conv_id"]  # real id for ratings storage
        ur_conv = read_rating(rater, conv_id) or {}

        cc1, cc2, cc3, cc4 = st.columns([1, 1, 4, 2])
        with cc1:
            st.button("⏪", key="p_conv", on_click=prev_conv, disabled=curr_idx == 0, width='stretch')
        with cc2:
            st.button("⏩", key="n_conv", on_click=next_conv, disabled=curr_idx == len(conv_options) - 1, width='stretch')
        with cc3:
            st.selectbox(
                "Select Conversation",
                conv_options,
                key="selected_conv_id",
                on_change=on_conv_change,
                label_visibility="collapsed",
            )
        with cc4:
            if not ur_conv:
                st.info("Pending", icon="⏳")
            elif ur_conv.get("true_emotion") == "unusable":
                st.error("Unusable", icon="❌")
            elif ur_conv.get("true_emotion") == "uncertain":
                st.warning("Uncertain", icon="⚠️")
            else:
                st.success(f"{ur_conv.get('true_emotion', '').title()}", icon="✅")

        c_audio_path = CONCAT_DIR / f"{conv_id}.wav"
        if c_audio_path.exists():
            st.audio(str(c_audio_path))
        else:
            st.error("Audio file missing.")

        # Rating form
        with st.form(key="concat_rating_form"):
            emotions = ["neutral", "frustrated", "calm", "anxious", "curious", "confused", "sad", "angry", "happy", "fearful", "surprised", "disgusted", "other"]
            def_idx = emotions.index(ur_conv.get("true_emotion")) if ur_conv.get("true_emotion") in emotions else 0

            fc1, fc2, fc3, fc4 = st.columns([2, 4, 2, 2])
            with fc1:
                submit = st.form_submit_button("Save Rating", width='stretch')
            with fc2:
                true_emotion = st.selectbox("True Emotion", emotions, index=def_idx, label_visibility="collapsed")
            with fc3:
                btn_uncertain_conv = st.form_submit_button("Uncertain", width='stretch')
            with fc4:
                btn_unusable_conv = st.form_submit_button("Unusable", width='stretch')

            notes = st.text_area("Notes (optional)", value=ur_conv.get("notes", ""))

            if btn_unusable_conv:
                write_rating(rater, conv_id, {"true_emotion": "unusable", "notes": notes})
                st.session_state["_conv_flash"] = "Conversation marked as unusable!"
                st.rerun()
            elif btn_uncertain_conv:
                write_rating(rater, conv_id, {"true_emotion": "uncertain", "notes": notes})
                st.session_state["_conv_flash"] = "Conversation marked as uncertain!"
                st.rerun()
            elif submit:
                write_rating(rater, conv_id, {"true_emotion": true_emotion, "notes": notes})
                st.session_state["_conv_flash"] = "Rating saved!"
                st.rerun()

        if flash := st.session_state.pop("_conv_flash", None):
            st.success(flash)

    # === RIGHT: Segments ===
    with col2:
        st.markdown("### Segment")
        seg_files = conv_data.get("segments", [])

        if not seg_files:
            st.warning("No segments found for this conversation.")
        else:
            if "selected_seg_filename" not in st.session_state or st.session_state.selected_seg_filename not in seg_files:
                st.session_state.selected_seg_filename = seg_files[0]

            curr_seg_idx = seg_files.index(st.session_state.selected_seg_filename)

            def prev_seg():
                idx = seg_files.index(st.session_state.selected_seg_filename)
                if idx > 0:
                    st.session_state.selected_seg_filename = seg_files[idx - 1]

            def next_seg():
                idx = seg_files.index(st.session_state.selected_seg_filename)
                if idx < len(seg_files) - 1:
                    st.session_state.selected_seg_filename = seg_files[idx + 1]

            s_ur = read_rating(rater, conv_id, st.session_state.selected_seg_filename) or {}

            sc1, sc2, sc3, sc4 = st.columns([1, 1, 4, 2])
            with sc1:
                st.button("◀️", key="p_seg", on_click=prev_seg, disabled=curr_seg_idx == 0, width='stretch')
            with sc2:
                st.button("▶️", key="n_seg", on_click=next_seg, disabled=curr_seg_idx == len(seg_files) - 1, width='stretch')
            with sc3:
                st.selectbox(
                    "Select Segment",
                    seg_files,
                    key="selected_seg_filename",
                    label_visibility="collapsed",
                )
            with sc4:
                if not s_ur:
                    st.info("Pending", icon="⏳")
                elif s_ur.get("true_emotion") == "unusable":
                    st.error("Unusable", icon="❌")
                elif s_ur.get("true_emotion") == "uncertain":
                    st.warning("Uncertain", icon="⚠️")
                else:
                    st.success(f"{s_ur.get('true_emotion', '').title()}", icon="✅")

            s_audio_path = SEGMENTS_DIR / conv_id / st.session_state.selected_seg_filename
            if s_audio_path.exists():
                st.audio(str(s_audio_path))
            else:
                st.error("Audio file missing.")

            # Segment rating form
            with st.form(key="segment_rating_form"):
                emotions = ["neutral", "frustrated", "calm", "anxious", "curious", "confused", "sad", "angry", "happy", "fearful", "surprised", "disgusted", "other"]
                def_idx = emotions.index(s_ur.get("true_emotion")) if s_ur.get("true_emotion") in emotions else 0

                fc1, fc2, fc3, fc4 = st.columns([2, 4, 2, 2])
                with fc1:
                    submit_seg = st.form_submit_button("Save Rating", width='stretch')
                with fc2:
                    s_emotion = st.selectbox("True Emotion", emotions, index=def_idx, label_visibility="collapsed")
                with fc3:
                    btn_uncertain = st.form_submit_button("Uncertain", width='stretch')
                with fc4:
                    btn_unusable = st.form_submit_button("Unusable", width='stretch')

                s_notes = st.text_area("Notes (optional)", value=s_ur.get("notes", ""))

                if btn_unusable:
                    write_rating(rater, conv_id, {"true_emotion": "unusable", "notes": s_notes}, segment=st.session_state.selected_seg_filename)
                    st.session_state["_seg_flash"] = "Segment marked as unusable."
                    st.rerun()
                elif btn_uncertain:
                    write_rating(rater, conv_id, {"true_emotion": "uncertain", "notes": s_notes}, segment=st.session_state.selected_seg_filename)
                    st.session_state["_seg_flash"] = "Segment marked as uncertain."
                    st.rerun()
                elif submit_seg:
                    write_rating(rater, conv_id, {"true_emotion": s_emotion, "notes": s_notes}, segment=st.session_state.selected_seg_filename)
                    st.session_state["_seg_flash"] = "Rating saved!"
                    st.rerun()

            if flash := st.session_state.pop("_seg_flash", None):
                st.success(flash)

    # ── Aggregate overview table for this conversation ─────────────
    if seg_files:
        st.markdown("---")
        st.markdown("### Your Segment Ratings Overview")

        overview_rows = []
        for seg in seg_files:
            sr = read_rating(rater, conv_id, seg) or {}
            overview_rows.append({
                "Segment": seg,
                "Emotion": sr.get("true_emotion", "—"),
                "Notes": sr.get("notes", ""),
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(overview_rows), hide_index=True, width='stretch')

    # ── Export section (admin only) ───────────────────────────────
    if rater == ADMIN_NAME:
        with st.expander("📥 Export All Ratings (CSV)"):
            all_raters = get_all_rater_names()
            if not all_raters:
                st.info("No ratings have been submitted yet.")
            else:
                st.markdown(f"**Raters found:** {', '.join(all_raters)}")

                # Build concat CSV
                concat_buf = io.StringIO()
                cw = csv.DictWriter(concat_buf, fieldnames=["conv_id", "rater", "emotion", "notes", "timestamp"])
                cw.writeheader()
                for conv in manifest:
                    cid = conv["conv_id"]
                    for r_name in all_raters:
                        r = read_rating(r_name, cid)
                        if r:
                            cw.writerow({"conv_id": cid, "rater": r_name, "emotion": r.get("true_emotion", ""), "notes": r.get("notes", ""), "timestamp": r.get("timestamp", "")})

                # Build segments CSV
                seg_buf = io.StringIO()
                sw = csv.DictWriter(seg_buf, fieldnames=["conv_id", "segment", "rater", "emotion", "notes", "timestamp"])
                sw.writeheader()
                for conv in manifest:
                    cid = conv["conv_id"]
                    for seg in conv.get("segments", []):
                        for r_name in all_raters:
                            r = read_rating(r_name, cid, seg)
                            if r:
                                sw.writerow({"conv_id": cid, "segment": seg, "rater": r_name, "emotion": r.get("true_emotion", ""), "notes": r.get("notes", ""), "timestamp": r.get("timestamp", "")})

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("⬇️ Download Conversation Ratings", concat_buf.getvalue(), "ratings_concat.csv", "text/csv", width='stretch')
                with dl2:
                    st.download_button("⬇️ Download Segment Ratings", seg_buf.getvalue(), "ratings_segments.csv", "text/csv", width='stretch')

                # Admin: delete rater data
                st.markdown("---")
                st.markdown("**🔧 Admin: Delete Rater Data**")
                del_col1, del_col2 = st.columns([3, 1])
                with del_col1:
                    rater_to_delete = st.selectbox("Select rater to delete", all_raters, key="admin_delete_rater", label_visibility="collapsed")
                with del_col2:
                    if st.button("🗑️ Delete All Ratings", type="primary", width='stretch'):
                        if delete_rater_ratings(rater_to_delete):
                            st.session_state["_admin_flash"] = f"Deleted all ratings for '{rater_to_delete}'."
                            st.rerun()
                if flash := st.session_state.pop("_admin_flash", None):
                    st.success(flash)

    # ── Keyboard shortcuts ────────────────────────────────────────
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
                            targetAudio = audios[0];
                        } else {
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
                Array.from(document.querySelectorAll('button p'))
                    .filter(el => el.textContent.trim() === 'Uncertain')
                    .forEach(p => { const btn = p.closest('button'); if (btn) styleBtn(btn, 'rgba(255, 189, 69, 0.2)', 'rgba(255, 189, 69, 0.5)', 'rgb(255, 189, 69)'); });
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

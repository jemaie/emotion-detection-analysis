# Diarization Evaluation Analysis

## Setup

**25 phone calls** tested across **7 diarization approaches**:

| Column | Model | Method |
|---|---|---|
| `out_mapped/pyannote` | Pyannote (0 refs) + offline speaker mapping | Diarize blind → map agent via embedding similarity |
| `out_pyannote/1_refs` | Pyannote | Pass 1 reference during diarization |
| `out_mapped/openai` | OpenAI (0 refs) + offline speaker mapping | Diarize blind → map agent via embedding similarity |
| `out_openai/1_refs` | OpenAI | Pass 1 reference |
| `out_openai/2_refs` | OpenAI | Pass 2 references |
| `out_openai/3_refs` | OpenAI | Pass 3 references |
| `out_openai/4_refs` | OpenAI | Pass 4 references |

> [!IMPORTANT]
> **Critical caveat:** Pyannote receives `num_speakers=2` as a hard constraint (both in `client.identify()` and `client.diarize()`). OpenAI's API does not offer this parameter. This makes **cross-model comparison of speaker count unfair.** However, since all calls are known to be two-party conversations, the constraint is justified domain knowledge — not artificial bias.

---

## CSV 1: [eval_num_speakers.csv](file:///c:/Users/MaierJerome/projects-ma/emotion-detection-analysis/diarizer/eval_num_speakers.csv) — Speaker Count Accuracy

| Approach | Correct (==2) | Percentage |
|---|---|---|
| Pyannote (mapped) | 25/25 | 100% |
| Pyannote (1 ref) | 25/25 | 100% |
| OpenAI (mapped) | 18/25 | 72% |
| OpenAI (1 ref) | 19/25 | 76% |
| OpenAI (2 refs) | 20/25 | 80% |
| OpenAI (3 refs) | 20/25 | 80% |
| OpenAI (4 refs) | 16/25 | 64% |

**Pyannote's 100% is expected** — it's constrained to 2 speakers. This metric is **only meaningful for comparing across OpenAI variants:**

- **0 refs (72%) → 2 refs (80%):** Slight improvement with references
- **3 refs (80%) → 4 refs (64%):** Adding more refs of the same speaker **hurts** — the model over-segments, likely because the API was designed for one-ref-per-speaker
- **2 refs appears to be the sweet spot** for OpenAI speaker count accuracy

This CSV can only *cautiously* support a cross-model argument — e.g., if manual evaluation leans only slightly toward Pyannote.

---

## CSV 2: [eval_agent_matched_by_reference.csv](file:///c:/Users/MaierJerome/projects-ma/emotion-detection-analysis/diarizer/eval_agent_matched_by_reference.csv) — Agent Identification

| Approach | Agent matched | Percentage |
|---|---|---|
| Pyannote (mapped) | 25/25 | 100% |
| Pyannote (1 ref) | 25/25 | 100% |
| OpenAI (mapped) | 25/25 | 100% |
| OpenAI (1 ref) | 16/25 | 64% |
| OpenAI (2 refs) | 20/25 | 80% |
| OpenAI (3 refs) | 25/25 | 100% |
| OpenAI (4 refs) | 25/25 | 100% |

> [!WARNING]
> **Overfitting concern at ≥3 refs:** The 100% agent-match rate at 3–4 refs is likely inflated. With multiple references of the same voice, OpenAI becomes overly aggressive at labeling segments as "agent" — great for matching, but it risks **mislabeling caller segments as agent.** This is dangerous for downstream emotion detection because caller speech could be lost.

**Interpreting with the overfitting lens:**

| Refs | Agent matched | Speaker count | Interpretation |
|---|---|---|---|
| **0 (mapped)** | 100% | 72% | Agent always identified; extra speakers are benign (see below) |
| **1** | 64% | 76% | Too little reference info for reliable agent identification |
| **2** | 80% | 80% | **Most balanced** — honest matching + segmentation |
| **3** | 100% | 80% | Overfitting begins — agent matching inflated |
| **4** | 100% | 64% | Clear overfitting — agent hogging labels, caller split into sub-speakers |

> [!NOTE]
> **Extra speakers ≠ lost caller audio.** The [assign_roles](file:///c:/Users/MaierJerome/projects-ma/emotion-detection-analysis/diarizer/run_batch_scripts/role_assign.py#4-60) logic maps the agent, then labels *all* remaining speakers as "caller" — regardless of how many there are. So when OpenAI (mapped) detects 3–4 speakers but correctly identifies the agent (which it does 100% of the time), the extra speakers simply get pooled into the caller bucket. **For downstream caller emotion detection, the over-segmentation is benign.** The only real risk is agent speech bleeding *into* the caller pool (a boundary-precision issue, not a speaker-count issue).

**Both Pyannote approaches** (mapped and 1-ref) achieve genuine 100%/100%, though Pyannote's `num_speakers=2` constraint makes agent identification trivially easier (only 2 candidates). This is defensible domain knowledge but should be acknowledged.

---

## Preliminary Verdict

- **Pyannote** leads on both metrics, but the `num_speakers=2` constraint gives it an inherent structural advantage. This is justified (calls *are* 2-party), but the advantage should be disclosed.
- **OpenAI (mapped)** is stronger than the 72% speaker-count score suggests — with 100% agent match and extra speakers harmlessly pooled as caller, it is a **genuine contender** alongside Pyannote. The offline mapping approach works well for both models.
- **OpenAI 2 refs** is the fairest *online* OpenAI variant (no offline mapping step) — most balanced between agent matching and speaker count, without overfitting artifacts.
- **Manual listening is the tiebreaker** — the CSVs cannot capture segment boundary precision or agent bleed-through. Compare Pyannote vs. OpenAI (mapped) and OpenAI 2-refs side-by-side.

---

## Manual Evaluation Guide

### 1. Listen to `caller_concat` files (compare side-by-side for the same call)

```
out_mapped/pyannote/caller_concat/<call_id>.wav
out_mapped/openai/caller_concat/<call_id>.wav
out_openai/2_refs/caller_concat/<call_id>.wav
```

**What to listen for:**
- ❌ **Agent bleed-through** — agent voice in "caller" audio (most critical for emotion detection)
- ❌ **Missing caller speech** — parts where the caller spoke but the segment is absent
- ❌ **Clipped words** — `trim_ms=200` may cut too aggressively at segment edges
- ❌ **Short garbage segments** — noise or silence that shouldn't be there
- ✅ **Clean caller-only audio** with natural sentence boundaries

### 2. Review diarized JSON files

```
out_mapped/pyannote/diarized/<call_id>.json
out_openai/2_refs/diarized/<call_id>.json
```

**Check:** Segment boundaries at natural turn-taking points, consistent speaker labeling, transcription quality.

### 3. Check summary JSONs (`index/<call_id>.summary.json`)

| Field | What it tells you |
|---|---|
| `num_segments_total` | High counts may indicate over-segmentation |
| `num_segments_caller_raw` → `_final` | Large drop = many segments too short or merged |
| `speaker_durations_sec` | Does caller/agent ratio make sense? |
| `speaker_to_role` | How many speakers detected and how assigned |

### 4. Priority calls to review

Start with calls where OpenAI struggled most:

| Call | Issue |
|---|---|
| `conv__+491729920245` | OpenAI (mapped): **4 speakers** detected |
| `conv__+49713216347` | OpenAI (mapped): **4 speakers** detected |
| `conv__+33327866663` | OpenAI (1 ref): agent **not matched** (even with 2 speakers) |
| `conv__+49706221952` | OpenAI (1 ref): agent **not matched** (even with 2 speakers) |
| `conv__+49661922317` | OpenAI (1 ref): **4 speakers** detected |

Listen to the original call alongside caller-concat to understand *why* these failed (background noise? hold music? tone shifts?).

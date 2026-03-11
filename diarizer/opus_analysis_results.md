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

> [!NOTE]
> **Why only Pyannote 0_refs and 1_ref?** Pyannote with ≥2 references is not evaluated because it is meaningless: with `num_speakers=2` and 2 references of the same speaker, the model trivially matches each of the 2 detected speakers to one of the 2 references — it becomes a forced 1:1 assignment rather than genuine identification. The `0_refs` variant diarizes blind (no references at all) and then maps the agent **offline** via embedding similarity using [run_speaker_mapper.py](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/run_speaker_mapper.py).

---

## CSV 1: [eval_num_speakers.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_num_speakers.csv) — Speaker Count Accuracy

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

## CSV 2: [eval_agent_matched_by_reference.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_agent_matched_by_reference.csv) — Agent Identification

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
> **Extra speakers ≠ lost caller audio.** The [assign_roles](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/run_batch_scripts/role_assign.py#4-60) logic maps the agent, then labels *all* remaining speakers as "caller" — regardless of how many there are. So when OpenAI (mapped) detects 3–4 speakers but correctly identifies the agent (which it does 100% of the time), the extra speakers simply get pooled into the caller bucket. **For downstream caller emotion detection, the over-segmentation is benign.** The only real risk is agent speech bleeding *into* the caller pool (a boundary-precision issue, not a speaker-count issue).

**Both Pyannote approaches** (mapped and 1-ref) achieve genuine 100%/100%, though Pyannote's `num_speakers=2` constraint makes agent identification trivially easier (only 2 candidates). This is defensible domain knowledge but should be acknowledged.

---

## Preliminary Verdict (CSVs 1 & 2)

- **Pyannote** leads on both metrics, but the `num_speakers=2` constraint gives it an inherent structural advantage. This is justified (calls *are* 2-party), but the advantage should be disclosed.
- **OpenAI (mapped)** is stronger than the 72% speaker-count score suggests — with 100% agent match and extra speakers harmlessly pooled as caller, it is a **genuine contender** alongside Pyannote. The offline mapping approach works well for both models.
- **OpenAI 2 refs** is the fairest *online* OpenAI variant (no offline mapping step) — most balanced between agent matching and speaker count, without overfitting artifacts.
- **Manual listening is the tiebreaker** — the CSVs cannot capture segment boundary precision or agent bleed-through. Compare Pyannote vs. OpenAI (mapped) and OpenAI 2-refs side-by-side.

---

## CSV 3: [eval_caller_duration.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_caller_duration.csv) — Caller Speaking Time

Total caller audio captured per approach (seconds summed across all 25 calls):

| Approach | Total (sec) | Avg/call (sec) | Zero-duration calls |
|---|---|---|---|
| **Pyannote (mapped)** | **1246** | 49.8 | 0 |
| **Pyannote (1 ref)** | **1286** | 51.5 | 0 |
| OpenAI (mapped) | 1102 | 44.1 | 0 |
| OpenAI (1 ref) | 709 | 28.4 | **10** |
| OpenAI (2 refs) | 966 | 38.7 | **5** |
| OpenAI (3 refs) | 865 | 34.6 | **4** |
| OpenAI (4 refs) | 646 | 25.8 | **9** |

> [!CAUTION]
> **OpenAI frequently loses the caller entirely.** A 0.00-second caller duration means the entire caller side was lost — no emotion detection possible for that call. OpenAI (1 ref) loses the caller in **10 out of 25 calls** (40%). Even OpenAI (2 refs) loses 5 calls. In contrast, **both Pyannote variants and OpenAI (mapped) never lose a single call.**

**Key observations:**
- **Pyannote (1 ref) captures the most caller audio** (1286s) — ~3% more than Pyannote (mapped), suggesting the reference helps slightly with boundary precision
- **OpenAI (mapped)** captures 1102s — 12% less than Pyannote, but critically **zero calls lost**. The gap may be from tighter segment boundaries or from agent bleed-through being excluded
- **OpenAI with online references** (1–4 refs) is unreliable — the zero-duration calls make these approaches unsuitable as the sole method for a full production run

---

## CSV 4: [eval_caller_segments.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_caller_segments.csv) — Caller Segment Counts

| Approach | Avg raw | Avg final | Drop rate |
|---|---|---|---|
| Pyannote (mapped) | 18.7 | 12.4 | 34.0% |
| Pyannote (1 ref) | 18.9 | 12.6 | 33.5% |
| OpenAI (mapped) | 21.5 | 12.6 | 41.2% |
| OpenAI (1 ref) | 13.8 | 8.2 | 40.8% |
| OpenAI (2 refs) | 17.6 | 10.9 | 37.8% |
| OpenAI (3 refs) | 15.8 | 9.9 | 37.5% |
| OpenAI (4 refs) | 12.2 | 7.3 | 39.8% |

**`raw`** = caller segments straight from the diarizer. **`final`** = after post-processing (trim 200ms edges, merge segments within 300ms, drop segments < 0.6s). **Drop rate** = percentage of raw segments that didn't survive filtering.

- **OpenAI (mapped) produces the most raw segments** (21.5 avg) with the highest drop rate (41.2%) — it over-segments, producing many short fragments that get filtered
- **Pyannote** has the lowest drop rate (~34%) — its segments are more naturally sized from the start
- **OpenAI reference variants** have lower raw counts, partly because zero-duration calls contribute 0 segments and pull the average down

---

## Updated Verdict (All 4 Metrics)

| Metric | Pyannote (mapped) | Pyannote (1 ref) | OpenAI (mapped) | OpenAI (1 ref) | OpenAI (2 refs) |
|---|---|---|---|---|---|
| Speaker count | ✅ 100% | ✅ 100% | ⚠️ 72% | ⚠️ 76% | ⚠️ 80% |
| Agent identified | ✅ 100% | ✅ 100% | ✅ 100% | ❌ 64% | ⚠️ 80% |
| Caller lost (0s) | ✅ 0 | ✅ 0 | ✅ 0 | ❌ 10 | ❌ 5 |
| Total caller audio | 1246s | **1286s** | 1102s | 709s | 966s |
| Segment drop rate | **34.0%** | **33.5%** | 41.2% | 40.8% | 37.8% |

**Conclusions:**
- **OpenAI (1 ref) and (2 refs) are eliminated** — losing the caller entirely in 10 and 5 calls respectively is disqualifying for a production pipeline
- **OpenAI (3/4 refs) are eliminated** — overfitting + caller loss
- **Three contenders remain:** Pyannote (mapped), Pyannote (1 ref), and OpenAI (mapped)
- All three have 0 lost calls and 100% agent identification — the difference comes down to **caller audio completeness** and **segment quality**, which can only be assessed by manual listening

---

## Manual Evaluation Guide

### 1. Listen to `caller_concat` files (compare side-by-side for the same call)

Compare the **5 remaining contenders** for each call:

```
out_mapped/pyannote/caller_concat/<call_id>.wav
out_pyannote/1_refs/caller_concat/<call_id>.wav
out_mapped/openai/caller_concat/<call_id>.wav
out_openai/1_refs/caller_concat/<call_id>.wav
out_openai/2_refs/caller_concat/<call_id>.wav
```

**What to listen for:**
- ❌ **Agent bleed-through** — agent voice in "caller" audio (most critical for emotion detection)
- ❌ **Missing caller speech** — parts where the caller spoke but the segment is absent
- ❌ **Clipped words** — `trim_ms=200` may cut too aggressively at segment edges
- ❌ **Short garbage segments** — noise or silence that shouldn't be there
- ✅ **Clean caller-only audio** with natural sentence boundaries

### 2. Manual listening scorecard

Use [eval_manual_listening.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_manual_listening.csv) — pre-filled with all 25 calls × 5 approaches. Score each on:

| Column | Scale | Meaning |
|---|---|---|
| Agent Bleed-Through | 0–3 | 0 = none, 3 = severe |
| Missing Caller Speech | 0–3 | 0 = none, 3 = large chunks missing |
| Overall Quality | 1–5 | Holistic assessment |
| Preferred | text | Which approach is best for this call |
| Notes | text | Free-text observations |

### 3. Priority calls to review first

Start with calls where OpenAI struggled most:

| Call | Issue |
|---|---|
| `conv__+491729920245` | OpenAI (mapped): **4 speakers** detected |
| `conv__+49713216347` | OpenAI (mapped): **4 speakers** detected |
| `conv__+33327866663` | OpenAI (1 ref): agent **not matched** (even with 2 speakers) |
| `conv__+49706221952` | OpenAI (1 ref): agent **not matched** (even with 2 speakers) |
| `conv__+49661922317` | OpenAI (1 ref): **4 speakers** detected |

Listen to the original call alongside caller-concat to understand *why* these failed (background noise? hold music? tone shifts?).


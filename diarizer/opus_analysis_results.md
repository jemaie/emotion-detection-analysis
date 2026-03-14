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

> [!NOTE]
> **Why only Pyannote 0_refs and 1_ref?** Pyannote with ≥2 references is not evaluated because it is meaningless: with `min_speakers=2` and 2 references of the same speaker, the model trivially matches each of the 2 detected speakers to one of the 2 references. The `0_refs` variant diarizes blind (no references at all) and then maps the agent **offline** via embedding similarity using [run_speaker_mapper.py](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/run_speaker_mapper.py).

---

## CSV 1: [eval_num_speakers.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_num_speakers.csv) — Speaker Count Accuracy

| Approach | Correct (==Expected) | Less Speakers | More Speakers |
|---|---|---|---|
| Pyannote (mapped) | 23/25 (92%) | 2 (8%) | 0 (0%) |
| Pyannote (1 ref) | 23/25 (92%) | 2 (8%) | 0 (0%) |
| OpenAI (mapped) | 20/25 (80%) | 0 (0%) | 5 (20%) |
| OpenAI (1 ref) | 22/25 (88%) | 0 (0%) | 3 (12%) |
| OpenAI (2 refs) | 21/25 (84%) | 1 (4%) | 3 (12%) |
| OpenAI (3 refs) | 17/25 (68%) | **8 (32%)*** | 0 (0%) |
| OpenAI (4 refs) | 14/25 (56%) | **10 (40%)*** | 1 (4%) |

**Pyannote is constrained to `min=2, max=3` speakers.** In the 3 complex calls that actually contain a third speaker (e.g., a system voice), Pyannote accurately detected the 3rd speaker in one call but missed it in the other two. Pyannote *never* over-segmentated (0 More Speakers). 

**OpenAI operates with no speaker constraints whatsoever**, meaning it can identify any number of speakers (including >3). Despite being completely unanchored, OpenAI (1 ref) achieves a strong **88%** accuracy rate, making it a close runner-up to Pyannote. Notably, it handled the complex 3-speaker calls slightly better than Pyannote, but its overall score is lower because it is prone to hallucinating extra speakers on standard calls (More Speakers).

**"Less Speakers" represents two completely different failure modes:**
1. **Pyannote:** Missed the 3rd speaker in complex calls.
2. **OpenAI:** At 3 and 4 references, OpenAI becomes aggressively overfit to the agent's reference voice, collapsing the *entire conversation* into just 1 speaker. For example, at 3 refs, it dropped 5 calls to 1 speaker, completely erasing the caller's distinct speech. 

- **0 refs (80%) → 1 ref (88%):** Noticeable improvement and the highest score for OpenAI. Providing just one reference helps the model anchor the conversation, reducing its tendency to over-segment (More Speakers dropping from 5 to 3).
- **2 refs (84%):** Still strong, but drops slightly.
- **3 refs (68%) → 4 refs (56%):** Adding more refs of the same speaker triggers catastrophic overfitting where the caller simply disappears from the diarization altogether.

---

## CSV 2: [eval_agent_matched_by_reference.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_agent_matched_by_reference.csv) — Agent Identification

| Approach | Agent detected | Percentage |
|---|---|---|
| Pyannote (mapped) | 25/25 | 100% |
| Pyannote (1 ref) | 25/25 | 100% |
| OpenAI (mapped) | 25/25 | 100% |
| OpenAI (1 ref) | 16/25 | 64% |
| OpenAI (2 refs) | 20/25 | 80% |
| OpenAI (3 refs) | 25/25 | 100% |
| OpenAI (4 refs) | 25/25 | 100% |

> [!WARNING]
> **Overfitting concern at ≥3 refs:** The 100% agent-detection rate at 3–4 refs is highly inflated. With multiple references of the same voice, OpenAI becomes overly aggressive at labeling segments as "agent". **This is directly supported by the "Less Speakers" data:** at 3 and 4 references, OpenAI collapsed 5 and 8 calls respectively into *only 1 speaker*, essentially categorizing the entire conversation as "agent only". This completely erases the caller's speech, making it catastrophic for downstream emotion detection.

> [!CAUTION]
> **Detected ≠ Correctly Mapped:** The CSV purely evaluates whether *someone* in the conversation was tagged as the agent by the model. It does not verify if that tag was applied to the correct person. Manual listening revealed cases where the caller's voice was incorrectly mapped as the agent. A 100% detection rate simply means the model isn't failing to output the role; it does not guarantee accuracy.

**Interpreting with the overfitting lens:**

| Refs | Agent detected | Correct Speakers | Less Speakers | More Speakers | Interpretation |
|---|---|---|---|---|---|
| **0 (mapped)** | 100% | 80% | 0% | 20% | Agent always identified; over-segmentation is benign (see below) |
| **1** | 64% | 88% | 0% | 12% | **Best speaker count accuracy** but catastrophic failure on agent detection |
| **2** | 80% | 84% | 4% | 12% | **Most balanced** — honest detection + segmentation |
| **3** | 100% | 68% | **32%** | 0% | Overfitting begins — agent detection inflated and callers lost |
| **4** | 100% | 56% | **40%** | 4% | Clear overfitting — agent hogging labels, caller lost |

> [!NOTE]
> **Extra speakers ("More Speakers") = Lost caller audio?** The [assign_roles](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/run_batch_scripts/role_assign.py#4-60) logic maps the agent, finds the remaining speaker with the longest duration, and labels them "caller". *Any additional speakers are labeled "other".* Therefore, when OpenAI hallucinates 3-4 speakers instead of 2, the extra fragments do not get pooled into the caller bucket; they are assigned "other" and excluded from downstream analysis. **For emotion detection, over-segmentation directly resulting in "More Speakers" *could* mean fragments of the caller's voice are lost, though it could also mean fragments of the *agent's* voice are lost, depending on whose speech was wrongly split into a third person.** Missing speakers ("Less Speakers") remains instantly fatal if the caller is completely erased into an agent-only call. A manual check is necessary to determine if the lost 'other' fragments are large enough to actually affect the emotion analysis.

**Both Pyannote approaches** (mapped and 1-ref) achieve genuine 100%/100%, largely because Pyannote naturally avoids hallucinating extra speakers (0 More Speakers). Because it almost exclusively outputs exactly 2 speakers, agent identification becomes trivially easier (only 2 candidates).

---

## Preliminary Verdict (CSVs 1 & 2)

- **Pyannote** remains highly robust for 2-party calls but showed a blind spot for the 3rd speaker (missing it in 2 out of 3 cases) despite being allowed to detect up to 3.
- **OpenAI (mapped)** is a powerhouse — with 100% agent detection and the flexibility to identify >2 speakers accurately, it is a **strong contender** alongside Pyannote. The offline mapping approach works extremely well for both models.
- **OpenAI 2 refs** is the fairest *online* OpenAI variant (no offline mapping step) — most balanced between agent detection and speaker count, without overfitting artifacts.
- **Manual listening is the absolute tiebreaker** — the CSVs cannot capture segment boundary precision, agent bleed-through, or crucially, whether a "detected" agent was actually mapped to the correct voice. Compare Pyannote vs. OpenAI (mapped) and OpenAI 2-refs side-by-side.

---

## CSV 3: [eval_caller_duration.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_caller_duration.csv) — Caller Speaking Time

Total caller audio captured per approach (seconds summed across all 25 calls):

| Metric | Pyannote (mapped) | Pyannote (1 ref) | OpenAI (mapped) | OpenAI (1 ref) | OpenAI (2 refs) | OpenAI (3 refs) | OpenAI (4 refs) |
|---|---|---|---|---|---|---|---|
| **Total (sec)** | **1246** | **1286** | 1102 | 709 | 966 | 865 | 646 |
| **Avg (all)** | 49.8 | 51.5 | 44.1 | 28.4 | 38.7 | 34.6 | 25.8 |
| **Avg (>0s)** | **49.8** | **51.5** | 44.1 | 44.3 | 48.3 | 43.2 | 38.0 |
| **Zero-duration calls**| **0** | **0** | **0** | 9 | 5 | 5 | 8 |

> [!CAUTION]
> **OpenAI frequently loses the caller entirely.** A 0.00-second caller duration means the entire caller side was lost — no emotion detection possible for that call. OpenAI (1 ref) loses the caller in **9 out of 25 calls** (36%). Even OpenAI (2 refs) loses 5 calls. In contrast, **both Pyannote variants and OpenAI (mapped) never lose a single call.**

**Key observations:**
- **Pyannote (1 ref) captures the most caller audio** (1286s) — ~3% more than Pyannote (mapped), suggesting the reference helps slightly with boundary precision.
- **OpenAI (mapped)** captures 1102s — 12% less than Pyannote, but critically **zero calls lost**.
- **The "More Speakers" Penalty:** As noted in CSV 1, OpenAI frequently hallucinates extra speakers (More Speakers). Because the role-assignment logic forces these extra fragments into an "other" bucket, they are lost from the caller evaluation. This is why OpenAI (mapped) captures significantly less total audio than Pyannote despite never losing a full call.
- **When OpenAI references work, they capture good volume:** The `Avg (>0s)` column shows that *when* OpenAI (2 refs) doesn't lose the caller entirely, it captures an average of 48.3s of audio, rivaling Pyannote. However, its unreliability (0s calls) makes it a brittle choice for a fully automated pipeline.

---

## CSV 4: [eval_caller_segments.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_caller_segments.csv) — Caller Segment Counts

| Metric | Pyannote (mapped) | Pyannote (1 ref) | OpenAI (mapped) | OpenAI (1 ref) | OpenAI (2 refs) | OpenAI (3 refs) | OpenAI (4 refs) |
|---|---|---|---|---|---|---|---|
| **Avg raw** | 18.5 | 18.7 | 20.6 | 12.9 | 16.5 | 15.8 | 12.1 |
| **Avg final** | 11.1 | 11.2 | 9.8 | 6.0 | 7.8 | 7.4 | 5.6 |
| **Drop rate (filtered)** | 35.2% | 34.7% | 33.4% | 35.3% | 31.7% | 31.1% | 32.3% |
| **Overlap Split rate** | 54.9% | 54.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Merge rate (combined)**| 5.4% | 5.6% | 18.8% | 18.6% | 21.1% | 22.3% | 21.8% |
| **Total reduction** | 40.0% | 40.0% | 52.2% | 53.9% | 52.8% | 53.4% | 54.1% |

**`raw`** = caller segments straight from the diarizer. **`final`** = after post-processing (merge segments within 300ms, *then* drop segments < 0.6s). **Drop rate** = percentage of raw segments deleted. **Overlap Split rate** (labeled Drop rate in CSV) = number of sub-segments (relative to raw total) born from fracturing a caller segment around an overlapping agent. **Merge rate** = percentage of raw segments fused together.

- **TRIM_MS=0 is a definitive win for OpenAI:** Because OpenAI already produces tight, hallucination-free boundaries, skipping the 200ms safety trim allows its heavily fragmented raw output to successfully merge. Across all OpenAI variants, ~18-22% of raw segments are now merging into continuous, unbroken blocks of caller speech.
- **OpenAI has zero overlapping speech:** OpenAI perfectly serializes the conversation. Its Overlap Split rate is **0.0%**, meaning it literally never models two people speaking at precisely the same time.
- **Pyannote experiences massive overlap fracturing:** Pyannote's raw output naturally consists of longer segments, but it frequently models overlapping speakers. When this occurs, the post-processor must excise the agent's overlapping audio, which splits a single caller segment into two sub-segments. This is why Pyannote has a massive **Overlap Split rate (~55%)**. *Note: Because overlap splitting temporarily increases the total number of segments before the duration filter kicks in, the individual rate percentages will not sum perfectly to the Total Reduction.* (Also note that Pyannote now natively merges ~5.5% of its segments).

---

## Updated Verdict (All 4 Metrics)

| Metric | Pyannote (mapped) | Pyannote (1 ref) | OpenAI (mapped) | OpenAI (1 ref) | OpenAI (2 refs) |
|---|---|---|---|---|---|
| Correct (Expected) | ✅ 92% | ✅ 92% | ⚠️ 80% | ⚠️ 88% | ⚠️ 84% |
| Less Speakers | ✅ 8% | ✅ 8% | ✅ 0% | ✅ 0% | ✅ 4% |
| More Speakers | ✅ 0% | ✅ 0% | ⚠️ 20% | ⚠️ 12% | ⚠️ 12% |
| Agent detected | ✅ 100% | ✅ 100% | ✅ 100% | ❌ 64% | ⚠️ 80% |
| Caller lost (0s) | ✅ 0 | ✅ 0 | ✅ 0 | ❌ 9 | ❌ 5 |
| Total caller audio | 1246s | **1281s** | 1077s | 678s | 927s |
| Drop rate (filtered) | 35.2% | 34.7% | **33.4%** | 35.3% | 31.7% |
| Overlap Split rate | 54.9% | 54.0% | **0.0%** | **0.0%** | **0.0%** |
| Merge rate | 5.4% | 5.6% | **18.8%** | 18.6% | 21.1% |

**Conclusions:**
- **OpenAI (1 ref) and (2 refs) are eliminated** — losing the caller entirely in 9 and 5 calls respectively is disqualifying for a production pipeline
- **OpenAI (3/4 refs) are eliminated** — overfitting + caller loss
- **Three contenders remain:** Pyannote (mapped), Pyannote (1 ref), and OpenAI (mapped)
- All three have 0 lost calls and 100% agent detection — **but this detection rate must be treated with caution.** The CSV only proves the model assigned *an* agent role; it does not prove it accurately mapped the agent's voice. The final difference comes down to **caller audio completeness**, **segment boundary quality**, and **accuracy of the agent mapping** (avoiding false assignments). These can only be assessed by manual listening.

---

## CSV 5: [eval_manual_listening.csv](file:///c:/Users/jemai/projects/emotion-detection-analysis/diarizer/eval_manual_listening.csv) — Manual Audio Quality Assessment

To break the tie between the quantitative metrics, human listening tests were conducted across the 25 calls, scoring the audio on a 1-5 scale (where 5 is perfect/none) for Agent Bleed-Through, Clipping, and Overall Quality. 

### The Verdict: Pyannote is the Definitive Winner

The manual evaluation revealed a massive disparity in actual audio quality that the quantitative CSVs could not capture. **Pyannote completely dominated the manual evaluation.** Because multiple methods could be tied for "Preferred" on a single call, the exact breakdown across the 25 calls is as follows:

- **Pyannote (1 ref):** Preferred in **23** calls
- **Pyannote (mapped):** Preferred in **21** calls
- **OpenAI (mapped):** Preferred in **2** calls
- **OpenAI (1 ref & 2 refs):** Preferred in **0** calls

**Average Manual Evaluation Scores (1-5 scale):**
*(Higher is better. 5 = Perfect/None)*

| Approach | Agent IDed (out of 25) | Agent Bleed-Through | Clipping | Overall Quality |
|---|---|---|---|---|
| **Pyannote (1 ref)** | **25/25** | **4.60** | **4.48** | 4.32 |
| **Pyannote (mapped)**| 23/25 | 4.57 | **4.48** | **4.35** |
| OpenAI (mapped) | 21/25 | 3.52 | 4.05 | 3.52 |
| OpenAI (2 refs) | 20/25 | 3.35 | 3.95 | 3.35 |
| OpenAI (1 ref) | 15/25 | 3.20 | 4.07 | 3.27 |

**Key drivers for this victory:**

1. **Minimal Agent Bleed-Through (~4.6/5):** Pyannote achieved excellent scores for preventing the agent's voice from bleeding into the caller's audio track. By contrast, OpenAI struggled significantly with bleed-through, dropping down to averages of 3.2–3.5. For an emotion detection pipeline, agent bleed-through is a fatal flaw, as the classifier will analyze the agent's emotions instead of the caller's. This completely disqualified OpenAI.
2. **Superior Boundary Precision (Clipping):** Pyannote achieved excellent scores for avoiding clipping (~4.5), noticeably outperforming OpenAI (~4.0). Before the latest revisions, Pyannote had seemed prone to clipping, but the adjusted metrics prove it delivers exceptionally clean boundaries without sacrificing the caller's actual words. Pyannote sweeps OpenAI across the board.
3. **No Missing Speech:** Neither model suffered from randomly missing chunks of caller speech within the segments they *did* identify. All models scored well here (mostly marked FALSE for missing speech).

**Final Conclusion for the Pipeline:**
While OpenAI looked highly competitive on paper, the manual listening test proved that its segment boundaries are fundamentally too messy (specifically regarding agent bleed-through) for sensitive downstream audio analysis. 

**Definitive Recommendation: `Pyannote (1-ref)`**
While both Pyannote variants produced virtually identical, high-quality audio, **Pyannote (1-ref)** is the ultimate winner. In the two calls where it beat out Pyannote (mapped), it was entirely because the offline mapping logic failed to identify the agent correctly. *(Note: This mapping failure is also exactly why Pyannote mapped appears to have a slightly higher 'Overall Quality' average of 4.35 vs 4.32. The mapped variant completely failed to process two difficult calls, thus excluding those lower-scoring calls from its average, whereas 1-ref successfully handled them and factored them into its score.)* Using Pyannote with 1 online reference perfectly balances robust mapping with pristine caller boundary precision.


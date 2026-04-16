================================================================================
                      EMOTION RATING TOOL — INSTRUCTIONS
================================================================================


1. LOGIN
--------

Upon opening the application, the rater is prompted to enter their name.
This name is used to identify and persist all ratings across sessions.
The same name must be used consistently in every session.


2. INTERFACE OVERVIEW
---------------------

The interface is divided into two columns:

  - Left column (Conversation):
    Displays the full concatenated caller audio for one conversation.

  - Right column (Segment):
    Displays individual segments of the same conversation.
    Segments belong to their parent conversation and represent smaller
    portions of the caller's speech within that conversation.

At the top of the page, two progress counters indicate how many
conversations and segments the rater has completed out of the total
available.

At the bottom of the page, a summary table displays all segment ratings
for the currently selected conversation, providing a quick overview of
how each segment was rated. A download button is available for exporting
the ratings.


3. RATING WORKFLOW
------------------

3.1 Listening

  Each panel contains an audio player. The rater listens to the audio
  and evaluates the perceived emotion of the caller.


3.2 Emotion Selection

  The following emotion labels are available:

    neutral, frustrated, calm, anxious, curious, confused,
    sad, angry, happy, fearful, surprised, disgusted, other

  The rater selects the single most appropriate label from the dropdown.

  Notes on the emotion set:
    - These emotions are used because established ML models use this
      set for emotion classification.
    - "angry" and "frustrated" are distinct: Frustration stems from
      obstacles blocking goals or unmet expectations, characterized by
      helplessness, impatience, or annoyance. Anger is a more intense,
      active response to perceived threats, injustice, or intentional harm.
    - "anxious" and "fearful" are distinct: Fear is an intense,
      short-term emotional response to an immediate, known danger
      or threat. In contrast, anxiety (anxiousness) is a lingering,
      future-oriented apprehension regarding potential or unknown threat.
    - "other" should be used when the perceived emotion does not match
      any of the listed labels.

 
3.3 Submission

  Three submission options are provided:

    Save Rating    The rater can identify an emotion and selects it
                   from the dropdown before submitting.

    Uncertain      The rater cannot confidently assign an emotion
                   label. Use this when:
                     - The emotion in the speech is ambiguous or unclear.
                     - The audio is too short to make a confident
                       assessment.

    Unusable       The audio is not suitable for rating. Use this when:
                     - The segment contains no meaningful speech (e.g.
                       stumbling, mumbling, unintelligible fragments).
                     - The segment is too short to carry any discernible
                       emotional content.
                     - There is excessive noise, silence, or other
                       technical issues.
                   Note: Depending on the evaluation method, unusable
                   segments may be excluded from analysis entirely.

  An optional notes field allows the rater to provide additional
  context for their decision (see Section 6 for note-taking guidelines).


3.4 Status Indicators

  After submission, a status badge is displayed next to each item:

    Pending          Not yet rated
    [Emotion label]  Rated with a specific emotion
    Uncertain        Marked as uncertain
    Unusable         Marked as unusable

  Ratings can be updated at any time by re-submitting.


4. NAVIGATION
-------------

Conversations and segments can be navigated using the dropdown menus
or the directional buttons:

  - Previous / Next Conversation — navigates between conversations.
  - Previous / Next Segment — navigates between segments within
    the selected conversation.

Keyboard Shortcuts:

    Arrow Up         Previous conversation
    Arrow Down       Next conversation
    Arrow Left       Previous segment
    Arrow Right      Next segment
    Space            Play / pause segment audio
    Shift + Space    Play / pause conversation audio

Keyboard shortcuts are disabled while a text input field is focused.


5. RATING GUIDELINES
---------------------

5.1 Context and the Realtime Advantage

  Unlike batch ML models that classify isolated audio clips, both the
  human rater and a realtime model have the advantage of context: the
  ability to hear the full conversation and understand how it develops
  over time. This is a legitimate advantage and should be used.

  In practice, this means:
    - Segments may be rated based on the emotional context of the
      surrounding conversation, not only in isolation.
    - If a segment on its own sounds neutral but occurs during a
      clearly frustrated exchange, it is valid to take that context
      into account.

  Think of this as a spectrum: on one end, a traditional ML model
  classifying a single clip in isolation; on the other end, a human
  listener with full conversational context. The rater is positioned
  closer to the human end of this spectrum.


5.2 Rating Conversations vs. Segments

  Both the full conversation and each individual segment should be rated.

  For conversation-level emotion: assign the most dominant emotion
  across the conversation, similar to how an ML model would classify
  the overall concatenated audio. In most cases, this is the emotion
  that appears most frequently or spans the longest duration.


5.3 When to Use Uncertain vs. Unusable

  - Use "Uncertain" when the speech itself is clear but the emotion
    is ambiguous or too subtle to confidently label.
  - Use "Unusable" when the audio content is not usable for emotion
    rating at all (e.g. no real speech, fragments, technical issues).


5.4 Note-Taking Guidelines

  Notes may be written in shorthand or abbreviated form — they do not
  need to be full sentences. Their purpose is to provide rating context
  when the rating on its own might not make sense, for example:

    - When the emotion changes over the course of a conversation
      and the selected label reflects the dominant one.
    - When "other" is selected and the actual perceived emotion
      needs to be described.
    - When contextual information influenced the rating decision.
    - Any other case where the rating benefits from explanation.


6. EVALUATION
-------------

The comparison between human ratings and model predictions is performed
on a per-label basis: the single emotion label set by the human rater
is compared against the emotion label predicted by the model for the
same segment or conversation.

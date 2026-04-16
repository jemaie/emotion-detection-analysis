# Emotion Rating Tool — Instructions

## 1. Login

Upon opening the application, the rater is prompted to enter their name. This name is used to identify and persist all ratings across sessions. The same name must be used consistently in every session.

## 2. Interface Overview

The interface is divided into two columns:

- **Left column (Conversation):** Displays the full concatenated caller audio for one conversation.
- **Right column (Segment):** Displays individual segments of the same conversation.

At the top of the page, two progress counters indicate how many conversations and segments the rater has completed out of the total available.

## 3. Rating Workflow

### 3.1 Listening

Each panel contains an audio player. The rater listens to the audio and evaluates the perceived emotion of the caller.

### 3.2 Emotion Selection

The following emotion labels are available:

`neutral`, `frustrated`, `calm`, `anxious`, `curious`, `confused`, `sad`, `angry`, `happy`, `fearful`, `surprised`, `disgusted`, `other`

The rater selects the most appropriate label from the dropdown.

### 3.3 Submission

Three submission options are provided:

| Button | Usage |
|---|---|
| **Save Rating** | The rater can identify an emotion and selects it from the dropdown before submitting. |
| **Uncertain** | The rater cannot confidently assign an emotion label. |
| **Unusable** | The audio is not suitable for rating (e.g. excessive noise, silence, or technical issues). |

An optional notes field allows the rater to provide additional context for their decision.

### 3.4 Status Indicators

After submission, a status badge is displayed next to each item:

| Badge | Meaning |
|---|---|
| Pending | Not yet rated |
| Emotion label | Rated with a specific emotion |
| Uncertain | Marked as uncertain |
| Unusable | Marked as unusable |

Ratings can be updated at any time by re-submitting.

## 4. Navigation

Conversations and segments can be navigated using the dropdown menus or the directional buttons:

- **Previous / Next Conversation** — navigates between conversations.
- **Previous / Next Segment** — navigates between segments within the selected conversation.

### Keyboard Shortcuts

| Key | Action |
|---|---|
| Arrow Up | Previous conversation |
| Arrow Down | Next conversation |
| Arrow Left | Previous segment |
| Arrow Right | Next segment |
| Space | Play / pause segment audio |
| Shift + Space | Play / pause conversation audio |

Keyboard shortcuts are disabled while a text input field is focused.

## 5. Segment Overview

A summary table at the bottom of the page displays all segment ratings for the currently selected conversation.

## 6. Rating Guidelines

- Both the full conversation and each individual segment should be rated.
- The **Uncertain** option should be used when the emotion is ambiguous rather than guessing.
- The **Unusable** option should be used for clips with no speech, heavy background noise, or other technical issues.
- Notes should be added when the rating decision requires explanation (e.g. emotion changes within the clip).

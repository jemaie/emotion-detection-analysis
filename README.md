# Emotion Detection Pipeline

This project provides a system to analyze emotions in audio files using both OpenAI's Realtime API and a local Hugging Face model (`wav2vec2`).

## Setup

1.  **Install Dependencies**:
    The system requires Python 3.8+. Install the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This might take a while due to large libraries like torch and transformers)*

2.  **Configuration**:
    -   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    -   Add your OpenAI API Key to `.env`.

3.  **Verify Installation**:
    Run the test audio generator to ensuring libraries are working:
    ```bash
    python create_test_audio.py
    ```
    This will create `test_audio.wav`.

## Usage

### Interactive Interface (Recommended)
1.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open `interface.ipynb`.
3.  Enter the full path to an audio file (e.g., `c:\Users\MaierJerome\projects-ma\emotion-detection-analysis\test_audio.wav`).
4.  Click **Run Analysis**.
5.  Listen to the audio and view the results from both models.
6.  Vote on which analysis was better.

### Results
All results and feedback are saved to `results.csv` in the project directory.

## Troubleshooting
-   **Audio Playback**: Ensure your system's audio output is configured correctly.
-   **OpenAI API**: Needs a valid key with access to the Realtime API (beta).
-   **Local Model**: The first run will download the model (~1GB), so be patient.

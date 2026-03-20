from pathlib import Path
import logging
import asyncio
import json
import soundfile as sf
from tqdm import tqdm
from openai_client import OpenAIRealtimeClient
from storage import read_evaluation, write_evaluation, update_runner_state, get_evaluation_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - OPENAI - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

AUDIO_DIR_CONCAT = Path("data/caller_concat_24kHz")
AUDIO_DIR_SEGMENTS = Path("data/caller_segments_24kHz")

CONCAT_FILES_TO_PROCESS = sorted([f for f in AUDIO_DIR_CONCAT.iterdir() if f.suffix == ".wav"], key=lambda x: x.name)[:10]
SEGMENT_FOLDERS_TO_PROCESS = sorted([d for d in AUDIO_DIR_SEGMENTS.iterdir() if d.is_dir()], key=lambda x: x.name)[:10]
TOTAL_CONCAT_FILES = len(CONCAT_FILES_TO_PROCESS)
TOTAL_SEGMENT_FOLDERS = len(SEGMENT_FOLDERS_TO_PROCESS)
TOTAL_SEGMENTS = sum(len(list(f.rglob("*.wav"))) for f in SEGMENT_FOLDERS_TO_PROCESS)

WORKER_NAME = "openai"
MODEL_KEY = "openai_realtime_rp"
DELAY_SECONDS = 5

def cleanup_stale_states(status="stopped"):
    """Resets the state of any previously interrupted processing."""
    logger.info(f"Setting worker {WORKER_NAME} status to {status}...")
    
    update_runner_state(
        WORKER_NAME, {
            "file": None,
            "is_segment": False,
            "model": None,
            "status": status,
        }, 
        total_concat_files=TOTAL_CONCAT_FILES,
        total_segments=TOTAL_SEGMENTS,
    )

async def analyze_file(client: OpenAIRealtimeClient, audio_path: Path) -> dict:
    """Helper to analyze a single audio file and return the prediction dict."""
    try:
        if DELAY_SECONDS > 0:
            logger.info(f"Applying artificial delay of {DELAY_SECONDS} seconds before request...")
            await asyncio.sleep(DELAY_SECONDS)
            
        audio_data, samplerate = sf.read(str(audio_path))
        raw_result = await client.analyze_stream(audio_data, samplerate)
        
        # Parse result
        if raw_result:
            try:
                # Strip markdown block ticks if present
                clean_res = raw_result.strip()
                if clean_res.startswith('```json'):
                    clean_res = clean_res[7:]
                elif clean_res.startswith('```'):
                    clean_res = clean_res[3:]
                if clean_res.endswith('```'):
                    clean_res = clean_res[:-3]
                # if not clean_res.strip().endswith("}"):
                #     clean_res = clean_res.strip() + "}"
                    
                return json.loads(clean_res.strip())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response for {audio_path}: {raw_result}")
                return {"raw_output": raw_result}
        else:
            logger.error(f"No meaningful response from OpenAI Realtime API for {audio_path}")
            return {"error": "No meaningful response from OpenAI Realtime API."}
            
    except Exception as e:
        logger.error(f"Error analyzing {audio_path}: {e}")
        return {"error": str(e)}

async def process_model():
    pending_concat = []
    for audio_path in CONCAT_FILES_TO_PROCESS:
        evaluation = read_evaluation(audio_path)
        pred = evaluation.get("predictions", {}).get(MODEL_KEY)
        if not pred or "error" in pred:
            pending_concat.append(audio_path)
            
    pending_segments = []
    for folder in SEGMENT_FOLDERS_TO_PROCESS:
        missing = False
        for wav_path in folder.rglob("*.wav"):
            evaluation = read_evaluation(wav_path)
            pred = evaluation.get("predictions", {}).get(MODEL_KEY)
            if not pred or "error" in pred:
                missing = True
                break
        if missing:
            pending_segments.append(folder)

    if not pending_concat and not pending_segments:
        logger.info(f"No pending files found for {MODEL_KEY}. All done!")
        return

    logger.info(f"[{MODEL_KEY}] Found {len(pending_concat)} pending concat files and {len(pending_segments)} pending segment folders. Initializing client...")
    client = OpenAIRealtimeClient()
    
    # 1. Process Concat Files
    if pending_concat:
        pbar_concat = tqdm(pending_concat, desc=f"Running {MODEL_KEY} (Concat)", total=TOTAL_CONCAT_FILES, initial=TOTAL_CONCAT_FILES - len(pending_concat))
        for audio_path in pbar_concat:
            file_val = audio_path.name
            
            pbar_concat.set_postfix_str(f"{file_val}")
            tqdm_dict = pbar_concat.format_dict if hasattr(pbar_concat, "format_dict") else {}
            
            update_runner_state(
                WORKER_NAME, 
                {
                    "file": file_val,
                    "is_segment": False,
                    "model": MODEL_KEY,
                    "status": "processing",
                    "tqdm_dict": tqdm_dict,
                }, 
                total_concat_files=TOTAL_CONCAT_FILES,
                total_segments=TOTAL_SEGMENTS,
            )
            
            result_obj = await analyze_file(client, audio_path)
            
            with get_evaluation_lock(audio_path):
                evaluation = read_evaluation(audio_path)
                if "predictions" not in evaluation:
                    evaluation["predictions"] = {}
                evaluation["predictions"][MODEL_KEY] = result_obj
                write_evaluation(audio_path, evaluation)

    # 2. Process Segment Folders
    if pending_segments:
        # We align the tqdm bar with the number of folders here for cleaner rendering
        pbar_segments = tqdm(pending_segments, desc=f"Running {MODEL_KEY} (Segments)", total=TOTAL_SEGMENT_FOLDERS, initial=TOTAL_SEGMENT_FOLDERS - len(pending_segments))
        for folder in pbar_segments:
            file_val = folder.name
            
            pbar_segments.set_postfix_str(f"{file_val}")
            tqdm_dict = pbar_segments.format_dict if hasattr(pbar_segments, "format_dict") else {}
            
            update_runner_state(
                WORKER_NAME, 
                {
                    "file": file_val,
                    "is_segment": True,
                    "model": MODEL_KEY,
                    "status": "processing",
                    "tqdm_dict": tqdm_dict,
                }, 
                total_concat_files=TOTAL_CONCAT_FILES,
                total_segments=TOTAL_SEGMENTS,
            )
            
            for wav_path in folder.rglob("*.wav"):
                evaluation = read_evaluation(wav_path)
                pred = evaluation.get("predictions", {}).get(MODEL_KEY)
                if pred and "error" not in pred:
                    continue
                
                result_obj = await analyze_file(client, wav_path)
                
                with get_evaluation_lock(wav_path):
                    evaluation = read_evaluation(wav_path)
                    if "predictions" not in evaluation:
                        evaluation["predictions"] = {}
                    evaluation["predictions"][MODEL_KEY] = result_obj
                    write_evaluation(wav_path, evaluation)

    # Force one last update at the end
    update_runner_state(
        WORKER_NAME, 
        {
            "file": None,
            "is_segment": False,
            "model": MODEL_KEY,
            "status": "idle",
            "tqdm_dict": {},
        }, 
        total_concat_files=TOTAL_CONCAT_FILES,
        total_segments=TOTAL_SEGMENTS,
    )


def main():
    logger.info("Starting up OpenAI Realtime labeler runner process...")
    cleanup_stale_states(status="idle")
    try:
        asyncio.run(process_model())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping runner...")
    except Exception as e:
        logger.error(f"Pipeline error for {WORKER_NAME}: {e}")
    finally:
        cleanup_stale_states(status="stopped")
        logger.info("Processing complete!")


if __name__ == "__main__":
    main()

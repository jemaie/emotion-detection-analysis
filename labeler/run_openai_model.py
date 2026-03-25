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

SPECIFIC_CONVERSATIONS = [
    "conv__+4915203230182_22-08-2024_8_06_41",
    "conv__+491732965552_22-08-2024_9_46_35",
    "conv__+491729920245_24-07-2024_10_15_48",
    "conv__+49713216347_28-08-2024_11_02_25",
    "conv__+49713182652_09-09-2024_10_08_43",
    "conv__+49706262738_21-08-2024_10_58_04",
    "conv__+4971397924_02-08-2024_8_51_17",
    "conv__+49713243352_09-09-2024_14_50_42",

    # "conv__+4917642730353_12-08-2024_13_49_48",
    # "conv__+4917630400675_11-09-2024_10_31_24",
    # "conv__+4971324883446_01-08-2024_10_10_41",
    # "conv__+49713244025_05-08-2024_8_53_03",
    # "conv__+491727691557_19-08-2024_8_10_48",
    # "conv__+491784952510_12-08-2024_11_22_35",
    # "conv__+491624159130_04-07-2024_10_44_58",
    # "conv__+4971325954_06-08-2024_10_25_05",
    # "conv__+4915224862835_28-08-2024_11_22_27",
    # "conv__+497132350_19-08-2024_14_33_01",
    # "conv__+4971323406762_12-07-2024_9_20_45",
    # "conv__+49713288866_28-08-2024_10_12_59",
    # "conv__+4915158884523_01-08-2024_11_25_19"
]

for conv in SPECIFIC_CONVERSATIONS:
    f_concat = AUDIO_DIR_CONCAT / f"{conv}.wav"
    if f_concat.exists() and f_concat not in CONCAT_FILES_TO_PROCESS:
        CONCAT_FILES_TO_PROCESS.append(f_concat)

    f_seg = AUDIO_DIR_SEGMENTS / conv
    if f_seg.exists() and f_seg.is_dir() and f_seg not in SEGMENT_FOLDERS_TO_PROCESS:
        SEGMENT_FOLDERS_TO_PROCESS.append(f_seg)

# Re-sort
CONCAT_FILES_TO_PROCESS = sorted(CONCAT_FILES_TO_PROCESS, key=lambda x: x.name)
SEGMENT_FOLDERS_TO_PROCESS = sorted(SEGMENT_FOLDERS_TO_PROCESS, key=lambda x: x.name)

TOTAL_CONCAT_FILES = len(CONCAT_FILES_TO_PROCESS)
TOTAL_SEGMENT_FOLDERS = len(SEGMENT_FOLDERS_TO_PROCESS)
TOTAL_SEGMENTS = sum(len(list(f.rglob("*.wav"))) for f in SEGMENT_FOLDERS_TO_PROCESS)

WORKER_NAME = "openai"
MODEL_KEY = "openai_realtime_1_5_ft_e_2"
DELAY_SECONDS_CONCAT = 15
DELAY_SECONDS_SEGMENTS = 60

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
        audio_data, samplerate = sf.read(str(audio_path))
        raw_result = await client.analyze_stream(audio_data, samplerate)
        
        # Parse result
        if raw_result:
            try:
                return json.loads(raw_result.strip())
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
            if DELAY_SECONDS_CONCAT > 0:
                await asyncio.sleep(DELAY_SECONDS_CONCAT)

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
            if DELAY_SECONDS_SEGMENTS > 0:
                await asyncio.sleep(DELAY_SECONDS_SEGMENTS)

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
        if DELAY_SECONDS_CONCAT > 0:
            logger.info(f"Applying artificial delay of {DELAY_SECONDS_CONCAT} seconds between concat requests...")
        if DELAY_SECONDS_SEGMENTS > 0:
            logger.info(f"Applying artificial delay of {DELAY_SECONDS_SEGMENTS} seconds between segment requests...")
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

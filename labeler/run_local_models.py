from pathlib import Path
import logging
import torch
from tqdm import tqdm
from local_model_factory import get_model_factories
from storage import read_evaluation, write_evaluation, update_runner_state, get_evaluation_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - LOCAL - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

AUDIO_DIR_CONCAT = Path("data/caller_concat_16kHz")
AUDIO_DIR_SEGMENTS = Path("data/caller_segments_16kHz")

CONCAT_FILES_TO_PROCESS = sorted([f for f in AUDIO_DIR_CONCAT.iterdir() if f.suffix == ".wav"], key=lambda x: x.name)[:10]
SEGMENT_FOLDERS_TO_PROCESS = sorted([d for d in AUDIO_DIR_SEGMENTS.iterdir() if d.is_dir()], key=lambda x: x.name)[:10]
TOTAL_CONCAT_FILES = len(CONCAT_FILES_TO_PROCESS)
TOTAL_SEGMENT_FOLDERS = len(SEGMENT_FOLDERS_TO_PROCESS)
TOTAL_SEGMENTS = sum(len(list(f.rglob("*.wav"))) for f in SEGMENT_FOLDERS_TO_PROCESS)

WORKER_NAME = "local"

def cleanup_stale_states(status="stopped"):
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
        vram_mb=0,
        vram_reserved_mb=0
    )

def process_model(model_name: str, config: dict):
    pending_concat = []
    for audio_path in CONCAT_FILES_TO_PROCESS:
        evaluation = read_evaluation(audio_path)
        if model_name not in evaluation.get("predictions", {}):
            pending_concat.append(audio_path)
            
    pending_segments = []
    for folder in SEGMENT_FOLDERS_TO_PROCESS:
        missing = False
        for wav_path in folder.rglob("*.wav"):
            evaluation = read_evaluation(wav_path)
            if model_name not in evaluation.get("predictions", {}):
                missing = True
                break
        if missing:
            pending_segments.append(folder)

    if not pending_concat and not pending_segments:
        logger.info(f"No pending files found for {model_name}. All done!")
        return
        
    logger.info(f"[{model_name}] Found {len(pending_concat)} pending concat files and {len(pending_segments)} pending segment folders. Instantiating model...")
    model_instance = config["factory"]()
    
    # 1. Process Concat Files
    if pending_concat:
        pbar_concat = tqdm(pending_concat, desc=f"Running {model_name} (Concat)", total=TOTAL_CONCAT_FILES, initial=TOTAL_CONCAT_FILES - len(pending_concat))    
        for audio_path in pbar_concat:
            file_val = audio_path.name
            
            vram_mb = 0
            vram_reserved_mb = 0
            if torch.cuda.is_available():
                vram_mb = int(torch.cuda.memory_allocated() / (1024 ** 2))
                vram_reserved_mb = int(torch.cuda.memory_reserved() / (1024 ** 2))
                
            pbar_concat.set_postfix_str(f"{file_val}")
            tqdm_dict = pbar_concat.format_dict if hasattr(pbar_concat, "format_dict") else {}
            
            update_runner_state(
                WORKER_NAME, 
                {
                    "file": file_val,
                    "is_segment": False,
                    "model": model_name,
                    "status": "processing",
                    "tqdm_dict": tqdm_dict,
                }, 
                total_concat_files=TOTAL_CONCAT_FILES,
                total_segments=TOTAL_SEGMENTS,
                vram_mb=vram_mb, 
                vram_reserved_mb=vram_reserved_mb
            )
            
            result = model_instance.predict(audio_path)
            
            with get_evaluation_lock(audio_path):
                evaluation = read_evaluation(audio_path)
                if "predictions" not in evaluation:
                    evaluation["predictions"] = {}
                evaluation["predictions"][model_name] = result
                write_evaluation(audio_path, evaluation)

    # 2. Process Segment Folders
    if pending_segments:
        pbar_segments = tqdm(pending_segments, desc=f"Running {model_name} (Segments)", total=TOTAL_SEGMENT_FOLDERS, initial=TOTAL_SEGMENT_FOLDERS - len(pending_segments))    
        for folder in pbar_segments:
            file_val = folder.name
            
            vram_mb = 0
            vram_reserved_mb = 0
            if torch.cuda.is_available():
                vram_mb = int(torch.cuda.memory_allocated() / (1024 ** 2))
                vram_reserved_mb = int(torch.cuda.memory_reserved() / (1024 ** 2))
                
            pbar_segments.set_postfix_str(f"{file_val}")
            tqdm_dict = pbar_segments.format_dict if hasattr(pbar_segments, "format_dict") else {}
            
            update_runner_state(
                WORKER_NAME, 
                {
                    "file": file_val,
                    "is_segment": True,
                    "model": model_name,
                    "status": "processing",
                    "tqdm_dict": tqdm_dict,
                }, 
                total_concat_files=TOTAL_CONCAT_FILES,
                total_segments=TOTAL_SEGMENTS,
                vram_mb=vram_mb, 
                vram_reserved_mb=vram_reserved_mb
            )
            
            for wav_path in folder.rglob("*.wav"):
                evaluation = read_evaluation(wav_path)
                if model_name in evaluation.get("predictions", {}):
                    continue
                result = model_instance.predict(wav_path)
                with get_evaluation_lock(wav_path):
                    evaluation = read_evaluation(wav_path)
                    if "predictions" not in evaluation:
                        evaluation["predictions"] = {}
                    evaluation["predictions"][model_name] = result
                    write_evaluation(wav_path, evaluation)
            
    # Force one last update at the end of the loops for this model
    update_runner_state(
        WORKER_NAME, 
        {
            "file": None,
            "is_segment": False,
            "model": model_name,
            "status": "idle",
            "tqdm_dict": {},
        }, 
        total_concat_files=TOTAL_CONCAT_FILES,
        total_segments=TOTAL_SEGMENTS,
    )
        
    logger.info(f"[{model_name}] Unloading model to free memory...")
    del model_instance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    logger.info("Starting up the labeler model runner process...")
    cleanup_stale_states(status="idle")
    try:
        local_factories = get_model_factories()
        for config in local_factories:
            process_model(config["name"], config)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping runner...")
    except Exception as e:
        logger.error(f"Pipeline error for {WORKER_NAME}: {e}")
    finally:
        cleanup_stale_states(status="stopped")
        logger.info("Processing complete!")

if __name__ == "__main__":
    main()

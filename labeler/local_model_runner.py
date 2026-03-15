import os
import logging
import torch
from tqdm import tqdm
from local_models import get_model_factories
from storage import read_evaluation, write_evaluation, update_runner_state, get_all_evaluations

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diarizer", "output", "caller_concat")
FILES_TO_PROCESS = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

def cleanup_stale_states(worker_name: str):
    """Resets the state of any previously interrupted processing."""
    logger.info(f"Cleaning up any stale processing states for {worker_name}...")
    
    update_runner_state(
        worker_name, {
            "file": None,
            "model": None,
            "status": "idle",
        }, 
        total_files=len(FILES_TO_PROCESS), 
        vram_mb=0, 
        vram_reserved_mb=0
    )
    
    # 2. Revert any evaluation file stuck in 'processing' back to 'unprocessed' 
    # (or 'processed' if it actually finished all models)
    try:
        factories = get_model_factories()
        evals = get_all_evaluations()
        for ev in evals:
            if ev.get("status") == "processing":
                all_done = False
                if "predictions" in ev:
                    all_done = all(f["name"] in ev["predictions"] for f in factories)
                
                ev["status"] = "processed" if all_done else "unprocessed"
                filename = ev.get("filename")
                if filename:
                    write_evaluation(filename, ev)
    except Exception as e:
        logger.error(f"Error cleaning up states: {e}")

def process_models(worker_name: str, models_to_run: list):
    try:
        for config in models_to_run:
            model_name = config["name"]
            pending_files = []
            
            for filename in FILES_TO_PROCESS:
                audio_path = os.path.join(AUDIO_DIR, filename)
                evaluation = read_evaluation(audio_path)
                
                preds = evaluation.get("predictions", {})
                if model_name not in preds:
                    pending_files.append((filename, audio_path))
                    
            if pending_files:
                logger.info(f"[{model_name}] Found {len(pending_files)} files to process. Instantiating model...")
                model_instance = config["factory"]()
                
                pbar = tqdm(pending_files, desc=f"Running {model_name}", total=len(FILES_TO_PROCESS), initial=len(FILES_TO_PROCESS) - len(pending_files))
                for filename, audio_path in pbar:
                    vram_mb = 0
                    vram_reserved_mb = 0
                    vram_info = ""
                    if worker_name == "local" and torch.cuda.is_available():
                        vram_mb = int(torch.cuda.memory_allocated() / (1024 ** 2))
                        vram_reserved_mb = int(torch.cuda.memory_reserved() / (1024 ** 2))
                        vram_info = f" | VRAM: {vram_mb}MB / {vram_reserved_mb}MB res"
                        
                    pbar.set_postfix_str(f"{filename}{vram_info}")
                    
                    tqdm_dict = pbar.format_dict if hasattr(pbar, "format_dict") else {}
                    
                    update_runner_state(
                        worker_name, 
                        {
                            "file": filename,
                            "model": model_name,
                            "status": "processing",
                            "tqdm_dict": tqdm_dict,
                        }, 
                        total_files=len(FILES_TO_PROCESS), 
                        vram_mb=vram_mb, 
                        vram_reserved_mb=vram_reserved_mb
                    )
                    
                    evaluation = read_evaluation(audio_path)
                    
                    if evaluation.get("status") in ("unprocessed", None):
                        evaluation["status"] = "processing"
                        write_evaluation(audio_path, evaluation)
                        
                    result = model_instance.predict(audio_path)
                    
                    evaluation = read_evaluation(audio_path)
                    if "predictions" not in evaluation:
                        evaluation["predictions"] = {}
                        
                    evaluation["predictions"][model_name] = result
                    
                    factories = get_model_factories()
                    all_done = all(f["name"] in evaluation["predictions"] for f in factories)
                    if all_done:
                        evaluation["status"] = "processed"
                        
                    write_evaluation(audio_path, evaluation)
                    
                logger.info(f"[{model_name}] Unloading model to free memory...")
                del model_instance
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error in {worker_name} worker testing loop: {e}")

def run_pipeline():            
    logger.info("Starting up sequential batch run. Models will load on demand...")
    worker_name = "local"
    cleanup_stale_states(worker_name)
    local_factories = get_model_factories()
    process_models(worker_name, local_factories)
    cleanup_stale_states(worker_name) 
    logger.info("Batch processing complete! All queued files processed.")

if __name__ == "__main__":
    logger.info("Starting up the labeler model runner process...")
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping runner...")
        cleanup_stale_states("local")

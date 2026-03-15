import os
import json
import logging
import ast
import pprint
import threading

logger = logging.getLogger(__name__)
_runner_state_lock = threading.Lock()

EVALUATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluations")
os.makedirs(EVALUATIONS_DIR, exist_ok=True)

def get_json_path(audio_filename: str) -> str:
    """Returns the path to the evaluation JSON file for a given audio file."""
    base_name = os.path.splitext(os.path.basename(audio_filename))[0]
    return os.path.join(EVALUATIONS_DIR, f"{base_name}.json")

def read_evaluation(audio_filename: str) -> dict:
    """Reads the JSON evaluation file if it exists, otherwise returns a skeleton."""
    json_path = get_json_path(audio_filename)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}")
            
    # Default skeleton
    return {
        "filename": os.path.basename(audio_filename),
        "status": "unprocessed",
        "predictions": {},
        "user_rating": None
    }

def write_evaluation(audio_filename: str, data: dict):
    """Writes the dictionary to the JSON evaluation file."""
    json_path = get_json_path(audio_filename)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error writing to {json_path}: {e}")

def get_all_evaluations() -> list[dict]:
    """Retrieves all currently stored evaluation JSONs (ignoring system state files)."""
    evals = []
    if not os.path.exists(EVALUATIONS_DIR):
        return evals
        
    for file in os.listdir(EVALUATIONS_DIR):
        if file.endswith(".json"):
            try:
                with open(os.path.join(EVALUATIONS_DIR, file), 'r', encoding='utf-8') as f:
                    evals.append(json.load(f))
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
    return evals

def update_runner_state(worker_key: str, state_update: dict, total_files: int = None, vram_mb: int = None, vram_reserved_mb: int = None):
    """Updates the state for a specific worker (e.g. 'local' or 'openai') atomically."""
    with _runner_state_lock:
        current_state = read_runner_state()
        
        if total_files is not None:
            current_state["total_files"] = total_files
            
        if vram_mb is not None:
            current_state["vram_mb"] = vram_mb
            
        if vram_reserved_mb is not None:
            current_state["vram_reserved_mb"] = vram_reserved_mb
            
        # Initialize dictionary structure if missing
        if worker_key not in current_state:
            current_state[worker_key] = {}
            
        current_state[worker_key].update(state_update)
        
        state_path = os.path.join(os.path.dirname(__file__), "runner_state.py")
        try:
            content = f"STATE = {pprint.pformat(current_state)}\n"
            with open(state_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error writing runner state: {e}")

def read_runner_state() -> dict:
    """Reads the current state of the model runner."""
    state_path = os.path.join(os.path.dirname(__file__), "runner_state.py")
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith("STATE ="):
                    # Extract string after "STATE ="
                    dict_str = content[7:].strip()
                    return ast.literal_eval(dict_str)
        except Exception as e:
            logger.error(f"Error reading runner state: {e}")
            pass
    return {"local": {"status": "idle"}, "openai": {"status": "idle"}}

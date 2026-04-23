from pathlib import Path
import json
import logging
import ast
import pprint
import filelock

logger = logging.getLogger(__name__)

EVALUATIONS_DIR = Path("output")
STATE_PATH = Path("runner_state.py")
STATE_LOCK_PATH = Path("runner_state.lock")
_runner_state_lock = filelock.FileLock(str(STATE_LOCK_PATH), timeout=10)

def get_json_path(audio_filename: Path) -> Path:
    """Returns the path to the evaluation JSON file for a given audio file."""
    parent_name = audio_filename.parent.name
    grandparent_name = audio_filename.parent.parent.name
    
    if "caller_segments" in parent_name or "caller_segments" in grandparent_name:
        conv_id = parent_name
        out_path = EVALUATIONS_DIR / "caller_segments" / conv_id / f"{audio_filename.stem}.json"
    elif "caller_phases" in parent_name or "caller_phases" in grandparent_name:
        conv_id = parent_name
        out_path = EVALUATIONS_DIR / "caller_phases" / conv_id / f"{audio_filename.stem}.json"
    else:
        out_path = EVALUATIONS_DIR / "caller_concat" / f"{audio_filename.stem}.json"
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def get_evaluation_lock(audio_filename: Path) -> filelock.FileLock:
    """Returns a FileLock for the specific evaluation file to ensure safe concurrent access."""
    lock_path = get_json_path(audio_filename).with_suffix(".json.lock")
    return filelock.FileLock(str(lock_path), timeout=10)

def read_evaluation(audio_filename: Path) -> dict:
    """Reads the JSON evaluation file if it exists, otherwise returns a skeleton."""
    json_path = get_json_path(audio_filename)
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}")
            
    # Default skeleton
    return {
        "filename": audio_filename.name,
        "predictions": {},
        "user_rating": None
    }

def write_evaluation(audio_filename: Path, data: dict):
    """Writes the dictionary to the JSON evaluation file."""
    json_path = get_json_path(audio_filename)
    
    if "predictions" in data:
        model_order = [
            "openai_realtime", "openai_realtime_2", "openai_realtime_rp", "openai_realtime_rp_2", "openai_realtime_ft", "openai_realtime_ft_2",
            "openai_realtime_1_5_ft", "openai_realtime_1_5_ft_2", "openai_realtime_1_5_ft_e", "openai_realtime_1_5_ft_e_2", "openai_realtime_1_5_ft_erp", "openai_realtime_1_5_ft_erp_2",
            "openai_realtime_1_5_ft_rp", "openai_realtime_1_5_ft_rp_2",
            "openai_realtime_1_5_ft_ns", "openai_realtime_1_5_ft_ns_2", "openai_realtime_1_5_ft_rp_ns", "openai_realtime_1_5_ft_rp_ns_2",
            "openai_realtime_russell", "openai_realtime_russell_2", "openai_realtime_plutchik", "openai_realtime_plutchik_2",
            "ehcalabres/wav2vec2", "speechbrain/wav2vec2", "superb/wav2vec2_base", "superb/wav2vec2_large",
            "superb/hubert_base", "superb/hubert_large", "iic/emotion2vec_base", "iic/emotion2vec_large"
        ]
        
        preds = data["predictions"]
        new_preds = {}
        
        # Add individual human raters first
        for k in sorted(preds.keys()):
            if k.startswith("human") and k != "human_consensus":
                new_preds[k] = preds[k]
                
        # Add consensus after raters
        if "human_consensus" in preds:
            new_preds["human_consensus"] = preds["human_consensus"]
                
        # Add models in defined order
        for m in model_order:
            if m in preds:
                new_preds[m] = preds[m]
                
        # Add remaining keys
        for k in sorted(preds.keys()):
            if k not in new_preds:
                new_preds[k] = preds[k]
                
        data["predictions"] = new_preds
        
    ordered_data = {}
    if "filename" in data: ordered_data["filename"] = data["filename"]
    if "user_rating" in data: ordered_data["user_rating"] = data["user_rating"]
    if "predictions" in data: ordered_data["predictions"] = data["predictions"]
    for k, v in data.items():
        if k not in ordered_data: ordered_data[k] = v
        
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error writing to {json_path}: {e}")

def get_all_evaluations() -> dict[str, list[dict]]:
    """Retrieves all currently stored evaluation JSONs (ignoring system state files)."""
    evals = {"concat": [], "segments": [], "phases": []}
    if not EVALUATIONS_DIR.exists():
        return evals
        
    for file in EVALUATIONS_DIR.rglob("*.json"):
        if "emotion_frameworks" in file.parts:
            continue
        if file.is_file():
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "caller_segments" in file.parts:
                        data["conv_id"] = file.parent.name
                        evals["segments"].append(data)
                    elif "caller_phases" in file.parts:
                        data["conv_id"] = file.parent.name
                        evals["phases"].append(data)
                    else:
                        data["conv_id"] = file.stem
                        evals["concat"].append(data)
            except Exception as e:
                logger.error(f"Failed to load {file.name}: {e}")
    return evals

def update_runner_state(worker_key: str, state_update: dict, total_concat_files: int = None, total_segments: int = None, total_phases: int = None, vram_mb: int = None, vram_reserved_mb: int = None):
    """Updates the state for a specific worker (e.g. 'local' or 'openai') atomically."""
    with _runner_state_lock:
        current_state = read_runner_state()
        
        if total_concat_files is not None:
            current_state["total_concat_files"] = total_concat_files
            
        if total_segments is not None:
            current_state["total_segments"] = total_segments
            
        if total_phases is not None:
            current_state["total_phases"] = total_phases
            
        if vram_mb is not None:
            current_state["vram_mb"] = vram_mb
            
        if vram_reserved_mb is not None:
            current_state["vram_reserved_mb"] = vram_reserved_mb
            
        # Initialize dictionary structure if missing
        if worker_key not in current_state:
            current_state[worker_key] = {}
            
        current_state[worker_key].update(state_update)
        
        try:
            content = f"STATE = {pprint.pformat(current_state)}\n"
            with open(STATE_PATH, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Error writing runner state: {e}")

def read_runner_state() -> dict:
    """Reads the current state of the model runner."""
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith("STATE ="):
                    # Extract string after "STATE ="
                    dict_str = content[7:].strip()
                    return ast.literal_eval(dict_str)
        except Exception as e:
            logger.error(f"Error reading runner state: {e}")
            pass
    return {"local": {"status": "stopped"}, "openai": {"status": "stopped"}}

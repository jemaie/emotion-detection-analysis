from pathlib import Path
import logging
import torch
from tqdm import tqdm
from local_model_factory import get_model_factories
from storage import read_evaluation, write_evaluation, get_evaluation_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - LOCAL_EVAL - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

AUDIO_DIR_CONCAT = Path("data/caller_concat_16kHz")
AUDIO_DIR_SEGMENTS = Path("data/caller_segments_16kHz")

CONVERSATIONS = [
    "conv__+4915203230182_22-08-2024_8_06_41",
    "conv__+491732965552_22-08-2024_9_46_35",
    "conv__+491729920245_24-07-2024_10_15_48",
    "conv__+49713216347_28-08-2024_11_02_25",
    "conv__+49713182652_09-09-2024_10_08_43",
    "conv__+49706262738_21-08-2024_10_58_04",
    "conv__+4971397924_02-08-2024_8_51_17",
    "conv__+49713243352_09-09-2024_14_50_42",

    "conv__+4917642730353_12-08-2024_13_49_48",
    "conv__+4917630400675_11-09-2024_10_31_24",
    "conv__+4971324883446_01-08-2024_10_10_41",
    "conv__+49713244025_05-08-2024_8_53_03",
    "conv__+491727691557_19-08-2024_8_10_48",
    "conv__+491784952510_12-08-2024_11_22_35",
    "conv__+491624159130_04-07-2024_10_44_58",
    "conv__+4971325954_06-08-2024_10_25_05",
    "conv__+4915224862835_28-08-2024_11_22_27",
    "conv__+497132350_19-08-2024_14_33_01",
    "conv__+4971323406762_12-07-2024_9_20_45",
    "conv__+49713288866_28-08-2024_10_12_59",
    "conv__+4915158884523_01-08-2024_11_25_19"
]

def process_model(model_name: str, config: dict):
    logger.info(f"--- Running evaluation for local model: {model_name} ---")
    
    model_instance = None
    
    for conv in tqdm(CONVERSATIONS, desc=f"Processing {model_name}"):
        # Process concat file
        concat_file = AUDIO_DIR_CONCAT / f"{conv}.wav"
        if concat_file.exists():
            evaluation = read_evaluation(concat_file)
            if model_name not in evaluation.get("predictions", {}):
                if model_instance is None:
                    # Lazy loading to avoid keeping all models in VRAM
                    logger.info(f"[{model_name}] Instantiating model...")
                    model_instance = config["factory"]()
                
                logger.info(f"[{model_name}] Processing concat file: {concat_file.name}")
                result = model_instance.predict(concat_file)
                with get_evaluation_lock(concat_file):
                    evaluation = read_evaluation(concat_file)
                    if "predictions" not in evaluation:
                        evaluation["predictions"] = {}
                    evaluation["predictions"][model_name] = result
                    write_evaluation(concat_file, evaluation)
        else:
            logger.warning(f"Concat file not found: {concat_file}")

        # Process segments
        segment_folder = AUDIO_DIR_SEGMENTS / conv
        if segment_folder.exists() and segment_folder.is_dir():
            wav_files = sorted(list(segment_folder.rglob("*.wav")))
            for wav_path in wav_files:
                evaluation = read_evaluation(wav_path)
                if model_name not in evaluation.get("predictions", {}):
                    if model_instance is None:
                        logger.info(f"[{model_name}] Instantiating model...")
                        model_instance = config["factory"]()
                        
                    logger.info(f"[{model_name}] Processing segment file: {wav_path.name}")
                    result = model_instance.predict(wav_path)
                    with get_evaluation_lock(wav_path):
                        evaluation = read_evaluation(wav_path)
                        if "predictions" not in evaluation:
                            evaluation["predictions"] = {}
                        evaluation["predictions"][model_name] = result
                        write_evaluation(wav_path, evaluation)
        else:
            logger.warning(f"Segment folder not found: {segment_folder}")
            
    if model_instance is not None:
        logger.info(f"[{model_name}] Unloading model to free memory...")
        del model_instance
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    logger.info("Starting evaluation for specific conversations (Local Models)...")
    try:
        local_factories = get_model_factories()
        for config in local_factories:
            process_model(config["name"], config)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping runner...")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        logger.info("Processing complete!")

if __name__ == "__main__":
    main()

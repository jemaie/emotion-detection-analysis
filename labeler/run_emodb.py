import asyncio
import json
import logging
import filelock
from pathlib import Path
import soundfile as sf
import torch
from tqdm import tqdm

from scripts_core.openai_client import OpenAIRealtimeClient
from scripts_core.local_model_factory import get_model_factories

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - EMODB_RUNNER - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

EMODB_DIR = Path("data/EmoDB")
OUTPUT_DIR = Path("output/emodb")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select the first 100 files, sorted alphabetically
EMODB_FILES = sorted([f for f in EMODB_DIR.glob("*.wav")])[:100]
TOTAL_FILES = len(EMODB_FILES)

# Storage overrides to prevent modifying storage.py
def get_json_path_emodb(audio_filename: Path) -> Path:
    out_path = OUTPUT_DIR / f"{audio_filename.stem}.json"
    return out_path

def get_evaluation_lock_emodb(audio_filename: Path) -> filelock.FileLock:
    lock_path = get_json_path_emodb(audio_filename).with_suffix(".json.lock")
    return filelock.FileLock(str(lock_path), timeout=10)

def read_evaluation_emodb(audio_filename: Path) -> dict:
    json_path = get_json_path_emodb(audio_filename)
    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {json_path}: {e}")
    return {"filename": audio_filename.name, "predictions": {}, "user_rating": None}

def write_evaluation_emodb(audio_filename: Path, data: dict):
    json_path = get_json_path_emodb(audio_filename)
    
    ordered_data = {}
    if "filename" in data: ordered_data["filename"] = data["filename"]
    if "user_rating" in data: ordered_data["user_rating"] = data["user_rating"]
    if "predictions" in data:
        # Standardize predictions order if present
        preds = data["predictions"]
        ordered_data["predictions"] = {k: preds[k] for k in sorted(preds.keys())}
    
    for k, v in data.items():
        if k not in ordered_data: ordered_data[k] = v
        
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error writing to {json_path}: {e}")

# Instructions definitions
INSTRUCTIONS_OPEN_EMOTION_RP = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf Audio und Transkript.\n'
    'Gib genau EINE präzise, gebräuchliche englische Emotion zurück (z.B. neutral, frustrated, anxious, calm).\n\n'
    'Regeln:\n'
    '- Nutze sowohl die stimmliche Ausdrucksweise (Tonfall, Prosodie, Sprechtempo) als auch den Inhalt.\n'
    '- Beurteile den tatsächlichen emotionalen Zustand, nicht nur das Thema.\n'
    '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
    '- Sei konsistent und eher konservativ.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

INSTRUCTIONS_FT_RP = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf Audio und Transkript.\n'
    'Klassifiziere die Emotion in genau EINE der folgenden Kategorien:\n'
    '[neutral, happy, sad, angry, fearful, disgust, surprised, other, unknown]\n\n'
    'Regeln:\n'
    '- Nutze sowohl die stimmliche Ausdrucksweise (Tonfall, Prosodie, Sprechtempo) als auch den Inhalt.\n'
    '- Beurteile den tatsächlichen emotionalen Zustand, nicht nur das Thema.\n'
    '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
    '- Sei konsistent und eher konservativ.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

INSTRUCTIONS_FT_NEW_SET = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf dem bereitgestellten Audio und Transkript.\n\n'
    'Klassifiziere die Emotion in genau EINE der folgenden Kategorien:\n'
    '[neutral, calm, happy, curious, annoyed, frustrated, angry, confused, anxious, concerned, sad, surprised, disgusted, relieved, grateful, other]\n\n'
    'Regeln:\n'
    '- Verwende sowohl die stimmliche Ausdrucksweise als auch den Inhalt des Gesagten.\n'
    '- Berücksichtige Tonfall, Prosodie, Sprechgeschwindigkeit und Wortwahl.\n'
    '- Bestimme die Emotion basierend auf dem tatsächlichen emotionalen Zustand des Sprechers, nicht nur anhand des Themas.\n'
    '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
    '- Sei konsistent und eher konservativ in deiner Einschätzung.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

INSTRUCTIONS_FT_RP_NEW_SET = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf Audio und Transkript.\n'
    'Klassifiziere die Emotion in genau EINE der folgenden Kategorien:\n'
    '[neutral, calm, happy, curious, annoyed, frustrated, angry, confused, anxious, concerned, sad, surprised, disgusted, relieved, grateful, other]\n\n'
    'Regeln:\n'
    '- Nutze sowohl die stimmliche Ausdrucksweise (Tonfall, Prosodie, Sprechtempo) als auch den Inhalt.\n'
    '- Beurteile den tatsächlichen emotionalen Zustand, nicht nur das Thema.\n'
    '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
    '- Sei konsistent und eher konservativ.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

INSTRUCTIONS_RUSSELL = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf Audio und Transkript.\n'
    'Klassifiziere den emotionalen Zustand nach dem Russell-Zirkumplex-Modell (Valenz und Arousal) in genau EINE der folgenden Kategorien:\n'
    '[high_arousal_negative, low_arousal_negative, high_arousal_positive, low_arousal_positive, neutral]\n\n'
    'Definitionen:\n'
    '- high_arousal_negative: Hohe Energie, negativ (z.B. wütend, frustriert, ängstlich)\n'
    '- low_arousal_negative: Niedrige Energie, negativ (z.B. traurig, besorgt, genervt)\n'
    '- high_arousal_positive: Hohe Energie, positiv (z.B. glücklich, überrascht)\n'
    '- low_arousal_positive: Niedrige Energie, positiv (z.B. ruhig, erleichtert)\n'
    '- neutral: Keine klare Ausprägung in Valenz oder Arousal.\n\n'
    'Regeln:\n'
    '- Nutze sowohl die stimmliche Ausdrucksweise (Tonfall, Prosodie, Sprechtempo) als auch den Inhalt.\n'
    '- Sei konsistent und eher konservativ.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

INSTRUCTIONS_PLUTCHIK = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf Audio und Transkript.\n'
    'Klassifiziere den emotionalen Zustand nach Plutchiks Rad der Emotionen in genau EINE der folgenden Kategorien:\n'
    '[anger, fear, joy, sadness, surprise, anticipation, disgust, trust, neutral]\n\n'
    'Regeln:\n'
    '- Nutze sowohl die stimmliche Ausdrucksweise (Tonfall, Prosodie, Sprechtempo) als auch den Inhalt.\n'
    '- Beurteile den tatsächlichen emotionalen Zustand, nicht nur das Thema.\n'
    '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
    '- Sei konsistent und eher konservativ.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

INSTRUCTIONS_EMOTION2VEC_COMPAT = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf Audio und Transkript.\n'
    'Klassifiziere die Emotion in genau EINE der folgenden Kategorien:\n'
    '[angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown]\n\n'
    'Regeln:\n'
    '- Nutze sowohl die stimmliche Ausdrucksweise (Tonfall, Prosodie, Sprechtempo) als auch den Inhalt.\n'
    '- Beurteile den tatsächlichen emotionalen Zustand, nicht nur das Thema.\n'
    '- Verwende "other", wenn die Emotion nicht in die anderen Kategorien passt (z.B. Langeweile).\n'
    '- Sei konsistent und eher konservativ.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

FT_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised", "other", "unknown"]
NEW_SET_EMOTIONS = ["neutral", "calm", "happy", "curious", "annoyed", "frustrated", "angry", "confused", "anxious", "concerned", "sad", "surprised", "disgusted", "relieved", "grateful", "other"]
RUSSELL_EMOTIONS = ["high_arousal_negative", "low_arousal_negative", "high_arousal_positive", "low_arousal_positive", "neutral"]
PLUTCHIK_EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise", "anticipation", "disgust", "trust", "neutral"]
EMOTION2VEC_COMPAT_EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

VERSIONS = {
    "openai_realtime_1_5_ft_rp": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS_FT_RP,
        "allowed_emotions": FT_EMOTIONS
    },
    "openai_realtime_1_5_ft_rp_2": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS_FT_RP,
        "allowed_emotions": FT_EMOTIONS
    },
    "openai_realtime_1_5_ft_erp": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS_OPEN_EMOTION_RP,
        "allowed_emotions": None
    },
    "openai_realtime_1_5_ft_erp_2": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS_OPEN_EMOTION_RP,
        "allowed_emotions": None
    },
    # "openai_realtime_1_5_ft_ns": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_FT_NEW_SET,
    #     "allowed_emotions": NEW_SET_EMOTIONS
    # },
    # "openai_realtime_1_5_ft_ns_2": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_FT_NEW_SET,
    #     "allowed_emotions": NEW_SET_EMOTIONS
    # },
    # "openai_realtime_1_5_ft_rp_ns": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_FT_RP_NEW_SET,
    #     "allowed_emotions": NEW_SET_EMOTIONS
    # },
    # "openai_realtime_1_5_ft_rp_ns_2": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_FT_RP_NEW_SET,
    #     "allowed_emotions": NEW_SET_EMOTIONS
    # },
    # "openai_realtime_russell": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_RUSSELL,
    #     "allowed_emotions": RUSSELL_EMOTIONS
    # },
    # "openai_realtime_russell_2": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_RUSSELL,
    #     "allowed_emotions": RUSSELL_EMOTIONS
    # },
    # "openai_realtime_plutchik": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_PLUTCHIK,
    #     "allowed_emotions": PLUTCHIK_EMOTIONS
    # },
    # "openai_realtime_plutchik_2": {
    #     "model_name": "gpt-realtime-1.5",
    #     "instructions": INSTRUCTIONS_PLUTCHIK,
    #     "allowed_emotions": PLUTCHIK_EMOTIONS
    # },
    "openai_realtime_1_5_emotion2vec_compat": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS_EMOTION2VEC_COMPAT,
        "allowed_emotions": EMOTION2VEC_COMPAT_EMOTIONS
    },
    "openai_realtime_1_5_emotion2vec_compat_2": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS_EMOTION2VEC_COMPAT,
        "allowed_emotions": EMOTION2VEC_COMPAT_EMOTIONS
    },
}

async def analyze_file_openai(client: OpenAIRealtimeClient, audio_path: Path) -> dict:
    try:
        audio_data, samplerate = sf.read(str(audio_path))
        raw_result = await client.analyze_stream(audio_data, samplerate)
        if raw_result:
            try:
                return json.loads(raw_result.strip())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from response for {audio_path}: {raw_result}")
                return {"raw_output": raw_result}
        else:
            return {"error": "No meaningful response from OpenAI Realtime API."}
    except Exception as e:
        return {"error": str(e)}

async def process_openai_model(model_key: str, config: dict):
    pending_files = []
    for audio_path in EMODB_FILES:
        evaluation = read_evaluation_emodb(audio_path)
        pred = evaluation.get("predictions", {}).get(model_key)
        if not pred or "error" in pred:
            pending_files.append(audio_path)

    if not pending_files:
        logger.info(f"No pending files found for OpenAI model {model_key}. All done!")
        return

    logger.info(f"[{model_key}] Found {len(pending_files)} pending files. Initializing client...")
    client = OpenAIRealtimeClient(
        model_name=config.get("model_name"), 
        instructions=config.get("instructions"), 
        allowed_emotions=config.get("allowed_emotions", None)
    )

    pbar = tqdm(pending_files, desc=f"Running {model_key}", total=TOTAL_FILES, initial=TOTAL_FILES - len(pending_files))
    for audio_path in pbar:
        result_obj = await analyze_file_openai(client, audio_path)
        with get_evaluation_lock_emodb(audio_path):
            evaluation = read_evaluation_emodb(audio_path)
            if "predictions" not in evaluation:
                evaluation["predictions"] = {}
            evaluation["predictions"][model_key] = result_obj
            write_evaluation_emodb(audio_path, evaluation)

def process_local_model(model_name: str, config: dict):
    pending_files = []
    for audio_path in EMODB_FILES:
        evaluation = read_evaluation_emodb(audio_path)
        pred = evaluation.get("predictions", {}).get(model_name)
        if not pred or "error" in pred or "error" in str(pred.get("emotion", "")):
            pending_files.append(audio_path)

    if not pending_files:
        logger.info(f"No pending files found for local model {model_name}. All done!")
        return

    logger.info(f"[{model_name}] Found {len(pending_files)} pending files. Instantiating model...")
    model_instance = config["factory"]()
    
    pbar = tqdm(pending_files, desc=f"Running {model_name}", total=TOTAL_FILES, initial=TOTAL_FILES - len(pending_files))
    for audio_path in pbar:
        result = model_instance.predict(audio_path)
        with get_evaluation_lock_emodb(audio_path):
            evaluation = read_evaluation_emodb(audio_path)
            if "predictions" not in evaluation:
                evaluation["predictions"] = {}
            evaluation["predictions"][model_name] = result
            write_evaluation_emodb(audio_path, evaluation)

    del model_instance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

async def main():
    logger.info(f"Found {TOTAL_FILES} files to process in EmoDB.")
    
    # 1. Process Local Models
    logger.info("--- Starting Local Models ---")
    local_factories = get_model_factories()
    for config in local_factories:
        try:
            process_local_model(config["name"], config)
        except Exception as e:
            logger.error(f"Error processing local model {config['name']}: {e}")

    # 2. Process OpenAI Models
    logger.info("--- Starting OpenAI Models ---")
    for version_key, config in VERSIONS.items():
        try:
            await process_openai_model(version_key, config)
        except Exception as e:
            logger.error(f"Error processing OpenAI model {version_key}: {e}")
            
    logger.info("EmoDB processing complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, stopping runner...")

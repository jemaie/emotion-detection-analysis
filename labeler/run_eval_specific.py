from pathlib import Path
import logging
import asyncio
import json
import soundfile as sf
from tqdm import tqdm
from openai_client import OpenAIRealtimeClient
from storage import read_evaluation, write_evaluation, get_evaluation_lock

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - EVAL - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

DELAY_SECONDS = 1

AUDIO_DIR_CONCAT = Path("data/caller_concat_24kHz")
AUDIO_DIR_SEGMENTS = Path("data/caller_segments_24kHz")

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

INSTRUCTIONS = (
    'Du bist ein System zur Emotionserkennung.\n'
    'Analysiere die Emotion des Sprechers basierend auf dem bereitgestellten Audio und Transkript.\n\n'
    
    'Klassifiziere die Emotion in genau EINE der folgenden Kategorien:\n'
    '[neutral, happy, sad, angry, fearful, disgust, surprised, other, unknown]\n\n'
    'Regeln:\n'
    '- Verwende sowohl die stimmliche Ausdrucksweise als auch den Inhalt des Gesagten.\n'
    '- Berücksichtige Tonfall, Prosodie, Sprechgeschwindigkeit und Wortwahl.\n'
    '- Bestimme die Emotion basierend auf dem tatsächlichen emotionalen Zustand des Sprechers, nicht nur anhand des Themas.\n'
    '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
    '- Sei konsistent und eher konservativ in deiner Einschätzung.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_emotion` aufrufen!'
)

VERSIONS = {
    "openai_realtime_ft": {
        "model_name": None,
        "instructions": INSTRUCTIONS
    },
    "openai_realtime_ft_2": {
        "model_name": None,
        "instructions": INSTRUCTIONS
    },
    "openai_realtime_1_5_ft": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS
    },
    "openai_realtime_1_5_ft_2": {
        "model_name": "gpt-realtime-1.5",
        "instructions": INSTRUCTIONS
    },
    "openai_realtime_1_5_ft_e": {
        "model_name": "gpt-realtime-1.5",
        "instructions": None,
        "use_open_emotion": True
    },
    "openai_realtime_1_5_ft_e_2": {
        "model_name": "gpt-realtime-1.5",
        "instructions": None,
        "use_open_emotion": True
    }
}

async def analyze_file(client: OpenAIRealtimeClient, audio_path: Path) -> dict:
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
            logger.error(f"No meaningful response for {audio_path}")
            return {"error": "No meaningful response from OpenAI Realtime API."}
    except Exception as e:
        logger.error(f"Error analyzing {audio_path}: {e}")
        return {"error": str(e)}

async def process_version(version_key: str, config: dict):
    model_name = config["model_name"]
    instructions = config["instructions"]
    
    logger.info(f"--- Running evaluation for version: {version_key} using model: {model_name} ---")
    client = OpenAIRealtimeClient(model_name=model_name, instructions=instructions, use_open_emotion=config.get("use_open_emotion", False))
    
    for conv in tqdm(CONVERSATIONS, desc=f"Processing {version_key}"):
        # Process concat file
        concat_file = AUDIO_DIR_CONCAT / f"{conv}.wav"
        if concat_file.exists():
            evaluation = read_evaluation(concat_file)
            pred = evaluation.get("predictions", {}).get(version_key)
            if not pred or "error" in pred:
                logger.info(f"[{version_key}] Processing concat file: {concat_file.name}")
                if DELAY_SECONDS > 0:
                    await asyncio.sleep(DELAY_SECONDS)
                result_obj = await analyze_file(client, concat_file)
                with get_evaluation_lock(concat_file):
                    evaluation = read_evaluation(concat_file)
                    if "predictions" not in evaluation:
                        evaluation["predictions"] = {}
                    evaluation["predictions"][version_key] = result_obj
                    write_evaluation(concat_file, evaluation)
        else:
            logger.warning(f"Concat file not found: {concat_file}")
            
        # Process segments
        segment_folder = AUDIO_DIR_SEGMENTS / conv
        if segment_folder.exists() and segment_folder.is_dir():
            wav_files = sorted(list(segment_folder.rglob("*.wav")))
            for wav_path in wav_files:
                evaluation = read_evaluation(wav_path)
                pred = evaluation.get("predictions", {}).get(version_key)
                if not pred or "error" in pred:
                    logger.info(f"[{version_key}] Processing segment file: {wav_path.name}")
                    if DELAY_SECONDS > 0:
                        await asyncio.sleep(DELAY_SECONDS)
                    result_obj = await analyze_file(client, wav_path)
                    with get_evaluation_lock(wav_path):
                        evaluation = read_evaluation(wav_path)
                        if "predictions" not in evaluation:
                            evaluation["predictions"] = {}
                        evaluation["predictions"][version_key] = result_obj
                        write_evaluation(wav_path, evaluation)
        else:
            logger.warning(f"Segment folder not found: {segment_folder}")

async def main():
    for version_key, config in VERSIONS.items():
        await process_version(version_key, config)
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())

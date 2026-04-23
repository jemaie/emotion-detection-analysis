import os
import logging
import asyncio
import numpy as np
import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from agents.realtime import (
    RealtimeAgent, 
    RealtimeRunner, 
    RealtimeModelSendRawMessage
)
from agents import FunctionTool, RunContextWrapper
from pydantic import BaseModel, Field
from typing import Any

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - SITUATION - %(levelname)s - %(message)s"))
logger.addHandler(handler)

INSTRUCTIONS_SITUATION = (
    'Du bist ein System zur Situationserkennung bei telefonischen Anfragen.\n'
    'Analysiere die GANZHEITLICHE Situation, in der sich der Anrufer befindet, basierend auf dem bereitgestellten Audio.\n'
    'Was ist der Grund für den Anruf? Welches Problem oder Anliegen schildert die Person?\n'
    'Zusätzlich: Wie wurde das Anliegen beantwortet oder gelöst? Welches Ergebnis oder welcher Vorschlag wurde am Ende gemacht?\n\n'

    'Regeln:\n'
    '- Beschreibe die Situation in eigenen Worten kurz, neutral und sachlich auf Deutsch.\n'
    '- Konzentriere dich auf den Kern des Problems.\n'
    '- Ordne der Situation zudem eine prägnante Kategorie (1-3 Wörter) zu.\n'
    '- Beschreibe die Lösung, das Ergebnis oder die Antwort des Agenten ebenfalls sachlich und prägnant.\n\n'
    'WICHTIG: Du MUSST für deine Antwort zwingend die Funktion `record_situation` aufrufen!'
)

class SituationArgs(BaseModel):
    category: str = Field(
        description="Eine prägnante Kategorie für die Situation (1-3 Wörter), z.B. 'Beschwerde', 'Technisches Problem', 'Auskunft'."
    )
    summary: str = Field(
        description="Eine kurze und präzise sachliche Zusammenfassung der Situation des Anrufers (max. 15 Wörter)."
    )
    resolution: str = Field(
        description="Eine kurze Zusammenfassung der Antwort, Lösung oder des Vorschlags des Agenten (max. 15 Wörter)."
    )

class SituationRealtimeClient:
    def __init__(self, api_key=None, model_name="gpt-realtime-1.5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    async def analyze_stream(self, audio_data, samplerate=24000):
        async def run_situation_record(ctx: RunContextWrapper[Any], args: str) -> str:
            try:
                parsed = SituationArgs.model_validate_json(args)
                return json.dumps({"category": parsed.category, "summary": parsed.summary, "resolution": parsed.resolution})
            except Exception as e:
                return f"Validation Error: {str(e)}"
            
        record_situation_tool = FunctionTool(
            name="record_situation",
            description="Speichert das Ergebnis der Situationsanalyse. (Zwingend aufzurufen)",
            params_json_schema=SituationArgs.model_json_schema(),
            on_invoke_tool=run_situation_record,
        )
        
        agent = RealtimeAgent(
            name="SituationAnalyzer",
            instructions=INSTRUCTIONS_SITUATION,
            tools=[record_situation_tool]
        )
        
        runner = RealtimeRunner(
            starting_agent=agent,
            config={
                "model_settings": {
                    "model_name": self.model_name,
                    "tool_choice": "required",
                    "turn_detection": None
                }
            }
        )
        
        final_response = None
        
        try:
            async with asyncio.timeout(120): # increased timeout for full recordings
                async with await runner.run() as session:
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)

                    pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                    await session.send_audio(pcm_data, commit=True)
                    await session.model.send_event(RealtimeModelSendRawMessage(
                        message={"type": "response.create"}
                    ))
                    
                    async for session_event in session:
                        if session_event.type in ("history_updated", "history_added"):
                            history = getattr(session_event, 'history', []) if session_event.type == "history_updated" else [getattr(session_event, 'item', None)]
                            for item in history:
                                if item and hasattr(item, 'role') and item.role == "assistant":
                                    if hasattr(item, 'content'):
                                        for content_part in item.content:
                                            transcript = getattr(content_part, 'transcript', None)
                                            if transcript:
                                                final_response = transcript

                        if session_event.type == "raw_model_event":
                            raw_model_event = session_event.data
                            if hasattr(raw_model_event, "data") and isinstance(raw_model_event.data, dict):
                                server_event = raw_model_event.data
                                server_event_type = server_event.get("type")
                                
                                if server_event_type == "response.function_call_arguments.done":
                                    if server_event.get("name") == "record_situation":
                                        final_response = server_event.get("arguments")
                                
                                if server_event_type == "response.output_item.done":
                                    item = server_event.get("item", {})
                                    if item.get("role") == "assistant":
                                        if item.get("type") == "function_call" and item.get("name") == "record_situation":
                                            final_response = item.get("arguments")
                                        else:
                                            for part in item.get("content", []):
                                                if part.get("type") == "audio" and part.get("transcript"):
                                                    final_response = part.get("transcript")
                                
                                if server_event_type == "response.done":
                                    if final_response:
                                        break

                        if session_event.type == "agent_end":
                            if final_response:
                                break
                        
                        if session_event.type == "error":
                            logger.error(f"Realtime error event: {getattr(session_event, 'error', 'Unknown error')}")
                            break

        except asyncio.TimeoutError:
            logger.error("OpenAI Realtime session timed out.")
        except Exception as e:
            logger.error(f"Error in OpenAI Realtime session: {e}")
                
        return final_response

async def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    AUDIO_DIR = BASE_DIR / "data/normalized_24kHz"
    OUTPUT_JSON = BASE_DIR / "output/situation_analysis.json"
    
    # We will use the 31 specific files we analyzed earlier
    # Extracted from our previous listing of caller_concat
    CONCAT_DIR = BASE_DIR / "output/caller_concat"
    if not CONCAT_DIR.exists():
        logger.error("Could not find caller_concat directory to get the file list.")
        return

    target_filenames = [f.stem + ".wav" for f in CONCAT_DIR.glob("*.json")]
    
    logger.info(f"Loaded {len(target_filenames)} target files to process.")
    
    # Load existing results to allow resuming
    results = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
            
    client = SituationRealtimeClient()
    
    pbar = tqdm(target_filenames, desc="Analyzing Situations")
    for filename in pbar:
        if filename in results and "error" not in results[filename]:
            continue
            
        audio_path = AUDIO_DIR / filename
        if not audio_path.exists():
            logger.warning(f"Audio file missing: {audio_path}")
            continue
            
        pbar.set_postfix_str(filename)
        
        try:
            audio_data, samplerate = sf.read(str(audio_path))
            raw_result = await client.analyze_stream(audio_data, samplerate)
            
            if raw_result:
                try:
                    parsed_result = json.loads(raw_result.strip())
                    results[filename] = parsed_result
                except json.JSONDecodeError:
                    results[filename] = {"error": "Invalid JSON", "raw": raw_result}
            else:
                results[filename] = {"error": "No meaningful response from model."}
                
        except Exception as e:
            logger.error(f"Failed processing {filename}: {e}")
            results[filename] = {"error": str(e)}
            
        # Save incrementally
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    asyncio.run(main())

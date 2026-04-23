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
from typing import Any, List, Literal

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - PHASES - %(levelname)s - %(message)s"))
logger.addHandler(handler)

INSTRUCTIONS_PHASES = (
    'Du bist ein System zur Gesprächsanalyse bei telefonischen Anfragen.\n'
    'Deine Aufgabe ist es, das vorliegende Audio-Gespräch in seine chronologischen Bestandteile (Phasen) zu segmentieren.\n\n'
    
    'Gute Telefonate bestehen in der Regel aus folgenden theoretischen Phasen:\n'
    '1. Begrüßen (Orientierung geben, "Was kann ich für Sie tun?", Identifikation des Themas)\n'
    '2. Das Anliegen klären (Rückfragen stellen, aktiv zuhören, Problemverstehen)\n'
    '3. Lösen (Lösungen aufzeigen, Maßnahmen treffen oder an Experten weiterleiten)\n'
    '4. Verabschieden (Aussprechen des Danks, "Auf Wiederhören", letzte Formalitäten)\n\n'
    
    'WICHTIG (ITERATIVER PROZESS): In der Realität ist ein Gespräch oft nicht linear! '
    'Es kann vorkommen, dass nach einem ersten Lösungsversuch ("3. Lösen") erneut Fragen geklärt werden müssen ("2. Das Anliegen klären") und dann wieder gelöst wird ("3. Lösen"). '
    'Du darfst und SOLLST so viele Phasen generieren, wie chronologisch im Audio vorkommen (z.B. 1, dann 2, dann 3, dann wieder 2, dann 3, dann 4).\n\n'
    
    'Regeln:\n'
    '- Identifiziere für jede erkannte Sequenz, wann sie im Audio grob startet und endet.\n'
    '- Schätze die Zeitstempel im Format "MM:SS".\n'
    '- Fasse kurz und sachlich zusammen, was in diesem Abschnitt besprochen wurde.\n'
    '- Du MUSST für deine finale Ausgabe zwingend die Funktion `record_phases` aufrufen!'
)

class Phase(BaseModel):
    phase_name: Literal["1. Begrüßen", "2. Das Anliegen klären", "3. Lösen", "4. Verabschieden"] = Field(description="Kategorie der Phase")
    start_time: str = Field(description="Geschätzte Startzeit im Format MM:SS.")
    end_time: str = Field(description="Geschätzte Endzeit im Format MM:SS.")
    summary: str = Field(description="Kurze Zusammenfassung des Inhalts dieser Phase. (max. 15 Wörter)")

class PhaseSegmentationArgs(BaseModel):
    phases: List[Phase]

class PhasesRealtimeClient:
    def __init__(self, api_key=None, model_name="gpt-realtime-1.5"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

    async def analyze_stream(self, audio_data, samplerate=24000):
        async def run_phases_record(ctx: RunContextWrapper[Any], args: str) -> str:
            try:
                parsed = PhaseSegmentationArgs.model_validate_json(args)
                phases_dict = [p.model_dump() for p in parsed.phases]
                return json.dumps({"phases": phases_dict})
            except Exception as e:
                return f"Validation Error: {str(e)}"
            
        record_phases_tool = FunctionTool(
            name="record_phases",
            description="Speichert die Segmentierung der 4 Phasen. (Zwingend aufzurufen)",
            params_json_schema=PhaseSegmentationArgs.model_json_schema(),
            on_invoke_tool=run_phases_record,
        )
        
        agent = RealtimeAgent(
            name="PhaseAnalyzer",
            instructions=INSTRUCTIONS_PHASES,
            tools=[record_phases_tool]
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
            async with asyncio.timeout(120): # Increased timeout for full recordings
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
                                    if server_event.get("name") == "record_phases":
                                        final_response = server_event.get("arguments")
                                
                                if server_event_type == "response.output_item.done":
                                    item = server_event.get("item", {})
                                    if item.get("role") == "assistant":
                                        if item.get("type") == "function_call" and item.get("name") == "record_phases":
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
    OUTPUT_JSON = BASE_DIR / "output/phases_analysis.json"
    
    CONCAT_DIR = BASE_DIR / "output/caller_concat"
    if not CONCAT_DIR.exists():
        logger.error("Could not find caller_concat directory to get the file list.")
        return

    target_filenames = [f.stem + ".wav" for f in CONCAT_DIR.glob("*.json")]
    logger.info(f"Loaded {len(target_filenames)} target files to process.")
    
    results = {}
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
            
    client = PhasesRealtimeClient()
    
    pbar = tqdm(target_filenames, desc="Analyzing Phases")
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

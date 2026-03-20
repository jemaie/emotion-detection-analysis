import os
import logging
import asyncio
import numpy as np
from dotenv import load_dotenv
from agents.realtime import (
    RealtimeAgent, 
    RealtimeRunner, 
    RealtimeModelSendRawMessage
)

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIRealtimeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.instructions = (
            'Du bist ein System zur Emotionserkennung.\n'
            'Analysiere die Emotion des Sprechers basierend auf dem bereitgestellten Audio und Transkript.\n\n'
            
            'Klassifiziere die Emotion in genau EINE der folgenden Kategorien:\n'
            '[neutral, happy, sad, angry, fearful, disgust, surprised, unknown]\n\n'

            'Wichtige Regeln:\n'
            '- Verwende sowohl die stimmliche Ausdrucksweise als auch den Inhalt des Gesagten.\n'
            '- Berücksichtige Tonfall, Prosodie, Sprechgeschwindigkeit und Wortwahl.\n'
            '- Bestimme die Emotion basierend auf dem tatsächlichen emotionalen Zustand des Sprechers, nicht nur anhand des Themas.\n'
            '- Verwende "neutral", wenn keine klare Emotion erkennbar ist.\n'
            '- Verwende "unknown", wenn Audio oder Transkript zu unklar, verrauscht oder unvollständig sind.\n'
            '- Gib eine kurze Begründung (Deutsch, maximal 8 Wörter).\n'
            '- Sei konsistent und eher konservativ in deiner Einschätzung.\n'
            '- Gib ausschließlich ein einziges gültiges JSON-Objekt zurück.\n'
            '- Beginne direkt mit "{" und ende mit "}".\n\n'

            'Die Antwort muss exakt folgendem Schema entsprechen:\n'
            '{\n'
            '  "emotion": "neutral|happy|sad|angry|fearful|disgust|surprised|unknown",\n'
            '  "reason": "<maximal 8 Wörter auf Deutsch>"\n'
            '}'
        )

    async def analyze_stream(self, audio_data, samplerate=24000):
        """
        Stream audio to OpenAI Realtime API using openai-agents and get emotion analysis.
        audio_data: numpy array of audio samples.
        """
        logger.info("Initializing OpenAI Realtime Agent...")
        
        agent = RealtimeAgent(
            name="EmotionAnalyzer",
            instructions=self.instructions
        )
        
        # Disable VAD to have full control over the turn completion
        runner = RealtimeRunner(
            starting_agent=agent,
            config={
                "model_settings": {
                    # "model_name": "gpt-realtime-2025-08-28",
                    "turn_detection": None
                }
            }
        )
        
        # logger.info("Starting Realtime Session...")
        final_response = None
        
        try:
            # Use a timeout to prevent hanging forever
            async with asyncio.timeout(60):
                async with await runner.run() as session:
                    # logger.info("Connected to OpenAI via openai-agents.")
                    
                    # Ensure audio is mono
                    if len(audio_data.shape) > 1:
                        # logger.info(f"Converting {audio_data.shape[1]}-channel audio to mono...")
                        audio_data = np.mean(audio_data, axis=1)

                    # Convert float32 numpy audio to pcm16 bytes
                    pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                    
                    # logger.info(f"Sending audio data ({len(pcm_data)} bytes)...")
                    await session.send_audio(pcm_data, commit=True)
                    
                    # Explicitly trigger a response since VAD is disabled
                    await session.model.send_event(RealtimeModelSendRawMessage(
                        message={"type": "response.create"}
                    ))
                    
                    # logger.info("Waiting for response...")
                    
                    async for session_event in session:
                        # Case 1: Capture from specialized history events
                        if session_event.type in ("history_updated", "history_added"):
                            history = getattr(session_event, 'history', []) if session_event.type == "history_updated" else [getattr(session_event, 'item', None)]
                            for item in history:
                                if item and hasattr(item, 'role') and item.role == "assistant":
                                    if hasattr(item, 'content'):
                                        for content_part in item.content:
                                            transcript = getattr(content_part, 'transcript', None)
                                            if transcript:
                                                final_response = transcript

                        # Case 2: Capture from raw server events (very robust)
                        if session_event.type == "raw_model_event":
                            raw_model_event = session_event.data
                            # RealtimeModelRawServerEvent has a .data field which is the raw dict
                            if hasattr(raw_model_event, "data") and isinstance(raw_model_event.data, dict):
                                server_event = raw_model_event.data
                                server_event_type = server_event.get("type")
                                
                                if server_event_type == "response.output_item.done":
                                    item = server_event.get("item", {})
                                    if item.get("role") == "assistant":
                                        for part in item.get("content", []):
                                            if part.get("type") == "audio" and part.get("transcript"):
                                                final_response = part.get("transcript")
                                
                                # Early exit if response is fully done
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

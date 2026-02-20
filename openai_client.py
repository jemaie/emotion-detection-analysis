import os
import json
import numpy as np
import asyncio
from dotenv import load_dotenv
from agents.realtime import (
    RealtimeAgent, 
    RealtimeRunner, 
    RealtimeModelSendRawMessage
)

load_dotenv()

class OpenAIRealtimeClient:
    def __init__(self, api_key=None, log_callback=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.log_callback = log_callback
        self.instructions = (
            "You are an emotion analysis system. "
            "Listen to the user input audio. "
            "Classify the emotion of the speaker. "
            "Return a concise JSON object with 'emotion' (string) and 'confidence' (float 0-1). "
            "Possible emotions: neutral, happy, sad, angry, fearful, disgust, surprised. "
            "Do not output anything else."
        )

    def log(self, message):
        if self.log_callback:
            # Green [OPENAI]
            self.log_callback(f"\033[92m[OPENAI]\033[0m {message}")

    async def analyze_stream(self, audio_data, samplerate=24000):
        """
        Stream audio to OpenAI Realtime API using openai-agents and get emotion analysis.
        audio_data: numpy array of audio samples.
        """
        self.log("Initializing OpenAI Realtime Agent...")
        
        agent = RealtimeAgent(
            name="EmotionAnalyzer",
            instructions=self.instructions
        )
        
        # Disable VAD to have full control over the turn completion
        runner = RealtimeRunner(
            starting_agent=agent,
            config={
                "model_settings": {
                    "turn_detection": None
                }
            }
        )
        
        self.log("Starting Realtime Session...")
        final_response = None
        
        try:
            # Use a timeout to prevent hanging forever
            async with asyncio.timeout(60):
                async with await runner.run() as session:
                    self.log("Connected to OpenAI via openai-agents.")
                    
                    # Ensure audio is mono
                    if len(audio_data.shape) > 1:
                        self.log(f"Converting {audio_data.shape[1]}-channel audio to mono...")
                        audio_data = np.mean(audio_data, axis=1)

                    # Convert float32 numpy audio to pcm16 bytes
                    pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                    
                    self.log(f"Sending audio data ({len(pcm_data)} bytes)...")
                    await session.send_audio(pcm_data, commit=True)
                    
                    # Explicitly trigger a response since VAD is disabled
                    await session.model.send_event(RealtimeModelSendRawMessage(
                        message={"type": "response.create"}
                    ))
                    
                    self.log("Waiting for response...")
                    
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
                                    self.log("Server event: response.output_item.done")
                                    item = server_event.get("item", {})
                                    if item.get("role") == "assistant":
                                        for part in item.get("content", []):
                                            if part.get("type") == "audio" and part.get("transcript"):
                                                final_response = part.get("transcript")
                                                self.log(f"Found transcript in raw output_item.done: {final_response}")
                                
                                # Early exit if response is fully done
                                if server_event_type == "response.done":
                                    self.log("Server event: response.done")
                                    if final_response:
                                        break

                        if session_event.type == "agent_end":
                            if final_response:
                                self.log(f"Turn ended with response: {final_response}")
                                break
                            else:
                                self.log("Turn ended without response yet...")
                        
                        if session_event.type == "error":
                            self.log(f"Realtime error event: {getattr(session_event, 'error', 'Unknown error')}")
                            break

        except asyncio.TimeoutError:
            self.log("OpenAI Realtime session timed out.")
        except Exception as e:
            self.log(f"Error in OpenAI Realtime session: {e}")
                
        return final_response

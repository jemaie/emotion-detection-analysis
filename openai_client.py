import os
import asyncio
import json
import base64
import websockets
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class OpenAIRealtimeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
        self.events = []

    async def analyze_stream(self, audio_data, samplerate=24000):
        """
        Stream audio to OpenAI Realtime API and get emotion analysis.
        audio_data: numpy array of audio samples.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        async with websockets.connect(self.url, additional_headers=headers) as websocket:
            # 1. Initialize Session
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "instructions": (
                        "You are an emotion analysis system. "
                        "Listen to the user input audio. "
                        "Classify the emotion of the speaker. "
                        "Return a concise JSON object with 'emotion' (string) and 'confidence' (float 0-1). "
                        "Possible emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised. "
                        "Do not output anything else."
                    ),
                    "input_audio_format": "pcm16",  # Assuming we convert to pcm16
                }
            }
            await websocket.send(json.dumps(session_update))

            # 2. Send Audio
            # We need to convert float32 numpy audio to pcm16 bytes
            # audio_data is typically float32 -1.0 to 1.0
            pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Send in chunks if it's large, but for simplicity here we might send one append or smaller chunks
            # The API expects base64 encoded chunks
            
            encoded_audio = base64.b64encode(pcm_data).decode("utf-8")
            
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }
            await websocket.send(json.dumps(audio_event))
            
            # 3. Commit Buffer (trigger generation)
            await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            
            # 4. Request Response
            await websocket.send(json.dumps({"type": "response.create"}))

            # 5. Receive Responses
            final_response = None
            
            try:
                async for message in websocket:
                    event = json.loads(message)
                    self.events.append(event)
                    
                    if event["type"] == "response.text.done":
                        final_response = event["text"]
                    
                    if event["type"] == "response.done":
                        # We might check if we got the text
                        break
            except Exception as e:
                print(f"Error in OpenAI stream: {e}")
                
            return final_response


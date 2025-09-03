import asyncio
import numpy as np
from datetime import datetime
from app.utilities.constants import Constants
from typing import Dict
from fastapi import HTTPException, WebSocket
from app.utilities import sken_logger
from app.utilities.client import SpeakerClassificationClient
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})

class ConnectionManager:
    def __init__(self):
        self.asr_connections: Dict[str, dict] = {}
        self.tts_connections: Dict[str, dict] = {}
        self.lock = asyncio.Lock()
        
    async def connect(self, client_id: str, websocket: WebSocket, sample_rate: int,  
                      connection_type: str, chunk_size: int = 0, 
                      client: SpeakerClassificationClient  = None, audio_format = "pcm"):
        async with self.lock:
            await websocket.accept()
            if connection_type == "asr":
                self.asr_connections[client_id] = {
                    'websocket': websocket,
                    'buffer': np.array([], dtype=np.int16) if Constants.fetch_constant("asr_type")=="nemo" else np.array([],dtype=np.float32),
                    'format': audio_format,
                    "snippet_start_time": 0,
                    "current_time": 0,
                    "silence_count": 0,
                    "sample_rate": sample_rate,
                    "chunk_size": chunk_size,
                    "last_activity": datetime.now(),
                    "memory": ConversationBufferWindowMemory(k=5,human_prefix="human")
                }
            elif connection_type == "tts":
                self.tts_connections[client_id] = {
                    'websocket': websocket,
                    'client': client,
                }
    
    async def disconnect(self, client_id: str, connection_type: str):
        async with self.lock:
            if connection_type == "asr":
                if client_id in self.asr_connections:
                    del self.asr_connections[client_id]
            elif connection_type == "tts":
                if client_id in self.tts_connections:
                    del self.tts_connections[client_id]

    async def send_message(self, client_id: str, message: str, connection_type: str):
        try:
            if connection_type == "asr":
                websocket = self.asr_connections[client_id]['websocket']
            else:
                websocket = self.tts_connections[client_id]['websocket']
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {e}")
            await self.disconnect(client_id)
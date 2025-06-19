from groq import Groq
from typing import Optional
from app.utilities import sken_logger
from app.utilities.env_util import EnvironmentVariableRetriever
import torch 
import os
import tempfile
from scipy.io import wavfile
from app.utilities.helper import Helper
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pydub import AudioSegment
from app.utilities.constants import Constants
import asyncio
from concurrent.futures import ThreadPoolExecutor


logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "nemo"})

class WhisperASR():
    def __init__(self):
        self.api_key = EnvironmentVariableRetriever.get_env_variable("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.api_key)
        self.client_caches: Dict[str] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._cache_lock = asyncio.Lock()
        
    async def transcribe_chunk(self, signal: np.ndarray, client_id: str) -> str:
        """
        Transcribe an audio chunk with error handling and performance monitoring
        """
        try:
            audio_segment = AudioSegment(signal.tobytes(), frame_rate=16000, sample_width=signal.dtype.itemsize, 
                                   channels=1)
            logger.info(f"Transcribing chunk for client {client_id}")
            result = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self.process_audio_transcription,
                audio_segment,
                self.client_caches.get(client_id, None),
            )

            # processing_time = time.time() - start_time 
            # logger.info(f"Transcription completed in {processing_time:.2f} seconds")           
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def process_audio_transcription(
        self, 
        audio_segment: np.ndarray,
        client_cache: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Process audio transcription with the model
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                audio_segment.export(temp_audio_path, format="wav")
            # Send the audio file to Groq Whisper
            with open(temp_audio_path, "rb") as file:
                response = self.groq_client.audio.transcriptions.create(
                    file=(temp_audio_path, file.read()),
                    model="distil-whisper-large-v3-en",
                    response_format="verbose_json",
                )

            os.remove(temp_audio_path)  # Clean up
            return response.text
        except Exception as e:
            logger.error(f"Error in ASR: {e}")
            return None

    async def transcribe_chunks(
        self, 
        signals: List[np.ndarray], 
        client_ids: List[str]
    ) -> List[str]:
        """
        Batch process multiple audio chunks simultaneously
        """
        tasks = [
            self.transcribe_chunk(signal, client_id) 
            for signal, client_id in zip(signals, client_ids)
        ]
        return await asyncio.gather(*tasks)

    async def clear_cache(self, client_id: str) -> bool:
        """
        Clear and reinitialize cache for a specific client
        """
        try:
            logger.info(f"Cache cleared and reinitialized for client {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache for client {client_id}: {str(e)}")
            return False

    async def remove_client(self, client_id: str):
        """
        Remove a client's cache from memory
        """
        async with self._cache_lock:
            if client_id in self.client_caches:
                del self.client_caches[client_id]
                logger.info(f"Removed cache for client {client_id}")
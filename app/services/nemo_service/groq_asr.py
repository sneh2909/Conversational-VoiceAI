import os
import time
import tempfile
import asyncio
import numpy as np
from typing import Dict
from dataclasses import dataclass, field
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from groq import Groq
from app.utilities import sken_logger
from app.utilities.env_util import EnvironmentVariableRetriever
import soundfile as sf  # For writing numpy audio to WAV

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "whisper"})


@dataclass
class WhisperClientCache:
    buffer_text: str = ""
    last_access: float = field(default_factory=time.time)


class GroqWhisperASR:
    def __init__(self):
        self.api_key = EnvironmentVariableRetriever.get_env_variable("GROQ_API_KEY")
        self.groq_client = Groq(api_key=self.api_key)
        self.client_caches: Dict[str, WhisperClientCache] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._cache_lock = asyncio.Lock()
        self._model_lock = Lock()  # Placeholder if needed later

    @classmethod
    async def create(cls, *args, **kwargs) -> "GroqWhisperASR":
        instance = cls()
        return instance

    async def init_client_cache(self, client_id: str):
        if client_id not in self.client_caches:
            self.client_caches[client_id] = WhisperClientCache()
            logger.info(f"Initialized Whisper cache for client {client_id}")

    async def transcribe_chunk(self, signal: np.ndarray, client_id: str) -> str:
        try:
            await self.init_client_cache(client_id)

            if signal is None or len(signal) == 0:
                logger.warning("transcribe_chunk called with empty audio signal.")
                return ""

            prev_text = self.client_caches[client_id].buffer_text

            transcription = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._transcribe_audio,
                signal,
                prev_text
            )

            if transcription:
                self.client_caches[client_id].buffer_text = transcription
                self.client_caches[client_id].last_access = time.time()

            return transcription

        except Exception as e:
            logger.error(f"Whisper transcription failed for client {client_id}: {e}", exc_info=True)
            return ""

    def _transcribe_audio(self, audio: np.ndarray, initial_prompt: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                sf.write(temp_audio_path, audio, samplerate=16000)

            with open(temp_audio_path, "rb") as file:
                response = self.groq_client.audio.transcriptions.create(
                    file=(temp_audio_path, file.read()),
                    model="distil-whisper-large-v3-en",
                    response_format="verbose_json",
                )

            os.remove(temp_audio_path)
            return response.text.strip() if response else ""

        except Exception as e:
            logger.error(f"Error in ASR: {e}", exc_info=True)
            return ""

    async def clear_cache(self, client_id: str) -> bool:
        try:
            if client_id in self.client_caches:
                self.client_caches[client_id].buffer_text = ""
                logger.info(f"Cleared Whisper text cache for client {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Whisper cache for {client_id}: {e}")
            return False

    async def remove_client(self, client_id: str):
        if client_id in self.client_caches:
            del self.client_caches[client_id]
            logger.info(f"Removed Whisper cache for client {client_id}")

    async def cleanup(self):
        try:
            self.client_caches.clear()
            self._thread_pool.shutdown(wait=True)
            logger.info("Whisper cleanup completed successfully")
        except Exception as e:
            logger.error(f"Whisper cleanup error: {e}", exc_info=True)
            raise
import torch
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import asyncio
from faster_whisper import WhisperModel
from app.utilities import sken_logger

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "whisper"})


@dataclass
class WhisperClientCache:
    # This buffer is for the transcript text, not audio.
    buffer_text: str = ""
    last_access: float = field(default_factory=time.time)


class CacheAwareWhisper:
    _model = None
    _model_lock = Lock()

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.client_caches: Dict[str, WhisperClientCache] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    @classmethod
    async def create(cls, *args, **kwargs) -> 'CacheAwareWhisper': # Accept dummy args from factory
        instance = cls()
        await instance._initialize()
        return instance

    async def _initialize(self):
        # This lock ensures the model is loaded only once.
        with self._model_lock:
            if CacheAwareWhisper._model is None:
                logger.info(f"Loading Whisper model 'medium' onto {self.device} with compute_type {self.compute_type}...")
                CacheAwareWhisper._model = WhisperModel("medium", device=self.device, compute_type=self.compute_type)
                logger.info("Whisper model loaded successfully.")

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

            # # Normalize audio
            # audio_data = signal.astype(np.float32) / 32768.0
            
            # so we run it in a separate thread to not block the asyncio event loop.
            segments, _ = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._transcribe_audio,
                signal,
                self.client_caches[client_id].buffer_text # Pass previous text as a prompt
            )

            # Join all transcribed segments into a single string.
            transcription = " ".join(seg.text.strip() for seg in segments).strip()
            
            # FIX: We now store the latest full transcript to use as a prompt for the next chunk.
            # This helps Whisper maintain context and handle overlapping speech.
            if transcription:
                self.client_caches[client_id].buffer_text = transcription
                self.client_caches[client_id].last_access = time.time()

            # FIX: Return only the new transcription.
            return transcription

        except Exception as e:
            logger.error(f"Whisper transcription failed for client {client_id}: {e}", exc_info=True)
            return ""

    def _transcribe_audio(self, audio: np.ndarray, initial_prompt: str):
        return CacheAwareWhisper._model.transcribe(audio, beam_size=2, initial_prompt=initial_prompt,no_repeat_ngram_size=10)

    async def clear_cache(self, client_id: str) -> bool:
        """Clears the text buffer for a specific client."""
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
            logger.error(f"Whisper cleanup error: {e}")
            raise
import torch
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import asyncio
from faster_whisper import WhisperModel
from app.utilities import sken_logger

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "whisper"})


@dataclass
class WhisperClientCache:
    buffer_text: str = ""
    last_access: float = time.time()


class CacheAwareWhisper:
    _model = None
    _model_lock = Lock()

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client_caches: Dict[str, WhisperClientCache] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

    @classmethod
    async def create(cls) -> 'CacheAwareWhisper':
        instance = cls()
        await instance._initialize()
        return instance

    async def _initialize(self):
        with self._model_lock:
            if CacheAwareWhisper._model is None:
                logger.info("Loading Whisper model...")
                CacheAwareWhisper._model = WhisperModel("medium", device=self.device, compute_type="float16")

    async def init_client_cache(self, client_id: str):
        if client_id not in self.client_caches:
            self.client_caches[client_id] = WhisperClientCache()
            logger.info(f"Initialized Whisper cache for client {client_id}")

    async def transcribe_chunk(self, signal: np.ndarray, client_id: str) -> str:
        try:
            await self.init_client_cache(client_id)

            if signal is None or len(signal) == 0:
                raise ValueError("Empty or invalid audio signal")

            # # Normalize audio
            # audio_data = signal.astype(np.float32) / 32768.0
            
            # Skip if signal is mostly silence
            # if np.abs(signal).mean() < 0.01:
            #     logger.debug(f"Skipping transcription for client {client_id} due to silence.")
            #     return self.client_caches[client_id].buffer_text

            # Run transcription in thread pool
            segments, _ = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._transcribe_audio,
                signal
            )

            transcription = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
            if transcription:
                self.client_caches[client_id].buffer_text += transcription + " "
                self.client_caches[client_id].last_access = time.time()

            return self.client_caches[client_id].buffer_text

        except Exception as e:
            logger.error(f"Whisper transcription failed for client {client_id}: {e}")
            return ""

    def _transcribe_audio(self, audio: np.ndarray):
        return CacheAwareWhisper._model.transcribe(audio,beam_size=2)

    async def clear_cache(self, client_id: str) -> bool:
        try:
            if client_id in self.client_caches:
                self.client_caches[client_id].buffer_text = ""
                logger.info(f"Cleared Whisper cache for client {client_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear Whisper cache: {e}")
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
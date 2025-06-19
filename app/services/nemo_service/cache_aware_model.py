import copy
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from threading import Lock
from dataclasses import dataclass
from omegaconf import OmegaConf, open_dict
from app.utilities.constants import Constants
from app.utilities import sken_logger
from app.utilities.helper import Helper
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "nemo"})

@dataclass
class ClientCache:
    """Data class to store client-specific cache information"""
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    previous_hypotheses: Optional[List] = None
    pred_out_stream: Optional[torch.Tensor] = None
    pre_encode_cache: Optional[torch.Tensor] = None
    step_num: int = 0
    last_access: float = time.time()

class CacheAware:
    _model = None
    _preprocessor = None
    _model_lock = Lock()

    def __init__(self, lookahead_size: int):
        """
        Initialize the class instance (private constructor)
        """
        self.model_path = Constants.fetch_constant("model_config")["model_path"]
        self.lookahead_size = lookahead_size
        self.client_caches: Dict[str, ClientCache] = {}
        self.device = torch.device(Constants.fetch_constant("model_config")["device"])
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._cache_lock = asyncio.Lock()

    @classmethod
    async def create(cls, lookahead_size: int) -> 'CacheAware':
        """
        Factory method to create and initialize CacheAware instance
        """
        instance = cls(lookahead_size)
        await instance._initialize()
        return instance

    async def _initialize(self):
        """
        Initialize the model and configurations
        """
        with self._model_lock:
            if CacheAware._model is None:
                CacheAware._model = nemo_asr.models.ASRModel.restore_from(
                    self.model_path, map_location=self.device)
                CacheAware._preprocessor = await self._init_preprocessor()
                logger.info("Model initialized successfully")

        await self._initialize_model()

    async def _initialize_model(self):
        """
        Initialize model configuration and validate settings
        """
        try:
            config = Constants.fetch_constant("model_config")
            self.decoder_type = config["decoder_type"]
            self.encoder_step_length = config["encoder_step_length"]

            self._validate_lookahead_size()
            await self._configure_model()
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    async def _init_preprocessor(self) -> EncDecCTCModelBPE:
        """
        Initialize audio preprocessor with optimized settings
        """
        cfg = copy.deepcopy(CacheAware._model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        
        preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(self.device)
        return preprocessor

    async def init_client_cache(self, client_id: str):
        """
        Initialize or reset cache for a specific client
        """
        async with self._cache_lock:
            cache_last_channel, cache_last_time, cache_last_channel_len = (
                CacheAware._model.encoder.get_initial_cache_state(batch_size=1)
            )
            
            num_channels = CacheAware._model.cfg.preprocessor.features
            pre_encode_cache_size = CacheAware._model.encoder.streaming_cfg.pre_encode_cache_size[1]
            
            self.client_caches[client_id] = ClientCache(
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                pre_encode_cache=torch.zeros(
                    (1, num_channels, pre_encode_cache_size), 
                    device=self.device
                )
            )
            logger.info(f"Initialized cache for client {client_id}")

    async def transcribe_chunk(self, signal: np.ndarray, client_id: str) -> str:
        """
        Transcribe an audio chunk with error handling and performance monitoring
        """
        start_time = time.time()
        
        try:
            if signal is None or len(signal) == 0:
                raise ValueError("Empty or invalid audio signal")
            if not isinstance(signal, np.ndarray):
                raise TypeError("Signal must be a numpy array")

            if client_id not in self.client_caches:
                await self.init_client_cache(client_id)

            async with self._cache_lock:
                client_cache = self.client_caches[client_id]
                client_cache.last_access = time.time()

            # Normalize audio
            audio_data = signal.astype(np.float32) / 32768.0

            # Process audio in thread pool
            processed_signal, processed_signal_length = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self.preprocess_audio,
                audio_data
            )

            # Update cache and process
            processed_signal = torch.cat(
                [client_cache.pre_encode_cache, processed_signal], 
                dim=-1
            )
            processed_signal_length += client_cache.pre_encode_cache.shape[2]
            
            # Update pre-encode cache
            client_cache.pre_encode_cache = processed_signal[:, :, -CacheAware._model.encoder.streaming_cfg.pre_encode_cache_size[1]:]

            # Process transcription
            logger.info(f"Transcribing chunk for client {client_id}")
            result = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self.process_audio_transcription,
                client_cache,
                processed_signal,
                processed_signal_length
            )

            processing_time = time.time() - start_time 
            # logger.info(f"Transcription completed in {processing_time:.2f} seconds")           
            return result

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def process_audio_transcription(
        self, 
        client_cache: ClientCache,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor
    ) -> str:
        """
        Process audio transcription with the model
        """
        with torch.no_grad():
            (
                client_cache.pred_out_stream,
                transcribed_texts,
                client_cache.cache_last_channel,
                client_cache.cache_last_time,
                client_cache.cache_last_channel_len,
                client_cache.previous_hypotheses,
            ) = CacheAware._model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=client_cache.cache_last_channel,
                cache_last_time=client_cache.cache_last_time,
                cache_last_channel_len=client_cache.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=client_cache.previous_hypotheses,
                previous_pred_out=client_cache.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        return Helper.extract_transcriptions(transcribed_texts)[0]

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
            await self.init_client_cache(client_id)
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

    async def cleanup(self):
        """
        Release resources and clear all caches
        """
        try:
            # Clear all client caches
            async with self._cache_lock:
                client_ids = list(self.client_caches.keys())
                for client_id in client_ids:
                    await self.remove_client(client_id)

            # Clear GPU memory if applicable
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def _validate_lookahead_size(self):
        """
        Validate the lookahead size configuration
        """
        if self.lookahead_size not in [0, 80, 480, 1040]:
            raise ValueError("Lookahead size must be one of: 0, 80, 480, 1040 ms")

    async def _configure_model(self):
        """
        Configure model decoder and attention settings
        """
        left_context_size = CacheAware._model.encoder.att_context_size[0]
        lookahead_steps = int(self.lookahead_size / self.encoder_step_length)
        CacheAware._model.encoder.set_default_att_context_size(
            [left_context_size, lookahead_steps]
        )

        with open_dict(CacheAware._model.cfg.decoding):
            CacheAware._model.cfg.decoding.strategy = (
                Constants.fetch_constant("model_config")["decoding_strategy"]
            )
            CacheAware._model.cfg.decoding.preserve_alignments = False
            
            if hasattr(CacheAware._model, "joint"):
                CacheAware._model.cfg.decoding.greedy.max_symbols = 10
                CacheAware._model.cfg.decoding.fused_batch_size = -1

        CacheAware._model.change_decoding_strategy(CacheAware._model.cfg.decoding)
        CacheAware._model.eval()

    def preprocess_audio(
        self, 
        audio: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess audio data for model input
        """
        audio_signal = torch.from_numpy(audio).unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio.shape[0]], device=self.device)
        
        return CacheAware._preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len
        )
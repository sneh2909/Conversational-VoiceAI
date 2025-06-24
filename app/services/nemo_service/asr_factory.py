from app.services.nemo_service.groq_asr import GroqWhisperASR
from app.services.nemo_service.cache_aware_model import CacheAware
from app.services.nemo_service.whisper import CacheAwareWhisper

class ASRFactory:
    @staticmethod
    async def create_asr_pipeline(type: str, lookahead_size: int = None):
        """Create ASR pipeline based on type"""
        if type == "whisper":
            return await CacheAwareWhisper.create()
        elif type == "nemo":
            return await CacheAware.create(lookahead_size)
        elif type == "groq":
            return await GroqWhisperASR.create()
        else:
            raise ValueError(f"Unknown ASR pipeline type: {type}")
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS
import os
from app.services.tts.tts_interface import TTSInterface
from app.utilities.constants import Constants
import io
from app.utilities import sken_logger

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})


class ChatterBox(TTSInterface):
    def __init__(self):
        self.device = "cuda:1"
        self.model = ChatterboxTTS.from_pretrained(device=self.device)


    def generate_audio(self, text: str, audio_prompt_path: str = None, output_dir: str = "temp_tts_audio"):
        if not self.model or not text.strip():
            return None

        os.makedirs(output_dir, exist_ok=True)
        
        try:
            if audio_prompt_path!=None:
                generated_wav = self.model.generate(text, audio_prompt_path=audio_prompt_path, exaggeration=Constants.fetch_constant("tts_exaggeration"),
                cfg_weight=Constants.fetch_constant("tts_cfg_weight"))
            generated_wav=self.model.generate(text=text)
            return generated_wav,self.model.sr
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    
    # def generate_audio(self, text, audio_prompt_path=None):
    #     """
    #     Generate audio with streaming chunks
        
    #     Args:
    #         text: Text to convert to speech
    #         audio_prompt_path: Optional audio prompt file path
            
    #     Yields:
    #         tuple: (audio_chunk, sample_rate) for each generated chunk
    #     """
    #     try:
    #         # Set up generation parameters
    #         if audio_prompt_path is not None:
    #             # Stream with audio prompt
    #             for audio_chunk, metrics in self.model.generate_stream(
    #                 text, 
    #                 audio_prompt_path=audio_prompt_path,
    #                 exaggeration=Constants.fetch_constant("tts_exaggeration"),
    #                 cfg_weight=Constants.fetch_constant("tts_cfg_weight"),
    #                 chunk_size=25  # Smaller chunks for lower latency
    #             ):
    #                 yield audio_chunk, self.model.sr
    #         else:
    #             # Stream without audio prompt
    #             for audio_chunk, metrics in self.model.generate_stream(
    #                 text=text,
    #                 exaggeration=Constants.fetch_constant("tts_exaggeration") if hasattr(Constants, 'fetch_constant') else 0.7,
    #                 cfg_weight=Constants.fetch_constant("tts_cfg_weight") if hasattr(Constants, 'fetch_constant') else 0.3,
    #                 chunk_size=25
    #             ):
    #                 yield audio_chunk, self.model.sr
                    
    #     except Exception as e:
    #         print(f"Error in streaming generation: {e}")
    #         # Fallback to non-streaming if streaming fails
    #         if audio_prompt_path is not None:
    #             generated_wav = self.model.generate(
    #                 text, 
    #                 audio_prompt_path=audio_prompt_path,
    #                 exaggeration=Constants.fetch_constant("tts_exaggeration"),
    #                 cfg_weight=Constants.fetch_constant("tts_cfg_weight")
    #             )
    #         else:
    #             generated_wav = self.model.generate(text=text)
            
    #         yield generated_wav, self.model.sr

    async def generate_audio_stream(self, text: str, audio_prompt_path: str = None):
        if not self.model or not text.strip():
            return

        try:
            for audio_chunk, metrics in self.model.generate_stream(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=Constants.fetch_constant("tts_exaggeration"),
                cfg_weight=Constants.fetch_constant("tts_cfg_weight"),
                chunk_size=50
            ):
                # buffer = io.BytesIO()
                # # torchaudio.save(buffer, audio_chunk, sample_rate=self.model.sr, format="wav")
                # buffer.seek(0)
                # chunk_data = buffer.read()
                yield audio_chunk,self.model.sr

                if metrics.latency_to_first_chunk:
                    print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")

        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")


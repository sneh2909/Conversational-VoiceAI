import torch
import torchaudio
from sarvamai import SarvamAI
import os
from app.services.tts.tts_interface import TTSInterface
from app.utilities.env_util import EnvironmentVariableRetriever
import io
from app.utilities import sken_logger
import tempfile
from app.utilities.constants import Constants
import numpy as np

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})


class TextToSpeech(TTSInterface):
    def __init__(self):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # Initialize Sarvam AI client
        # You should move this API key to environment variables or Constants for security
        self.client = SarvamAI(api_subscription_key=EnvironmentVariableRetriever.get_env_variable("SARVAM_API_KEY"))
        self.sr = 22050  # Default sample rate, you can adjust this based on Sarvam's output
        
        # Default Sarvam TTS parameters - you can make these configurable via Constants
        self.model =  "bulbul:v2"
        self.speaker = "karun"
        self.target_language = "hi-IN"

    def generate_audio(self, text: str, audio_prompt_path: str = None, output_dir: str = "temp_tts_audio"):
        if not text.strip():
            return None

        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Generate audio using Sarvam API
            response = self.client.text_to_speech.convert(
                target_language_code=self.target_language,
                text=text,
                model=self.model,
                speaker=self.speaker
            )
            
            # Extract audio data from response
            # The response contains base64 encoded audio, decode it
            import base64
            if hasattr(response, 'audios') and response.audios:
                # Get the first audio from the response
                audio_base64 = response.audios[0]
                audio_bytes = base64.b64decode(audio_base64)
            elif hasattr(response, 'audio'):
                # Alternative attribute name
                audio_base64 = response.audio
                audio_bytes = base64.b64decode(audio_base64)
            else:
                # If response structure is different, log it for debugging
                logger.error(f"Unexpected response structure: {type(response)}, attributes: {dir(response)}")
                return None
            
            # Save to temporary file to load with torchaudio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Save the decoded audio data to temporary file
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Load the audio file with torchaudio to get tensor format
            generated_wav, sample_rate = torchaudio.load(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Update sample rate
            self.sr = sample_rate
            
            return generated_wav, self.sr
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    async def generate_audio_stream(self, text: str, audio_prompt_path: str = None):
        if not text.strip():
            return

        try:
            # Sarvam API doesn't support streaming directly, so we'll simulate it
            # by generating the full audio and then chunking it
            response = self.client.text_to_speech.convert(
                target_language_code=self.target_language,
                text=text,
                model=self.model,
                speaker=self.speaker
            )
            
            # Extract and decode audio data from response
            import base64
            if hasattr(response, 'audios') and response.audios:
                audio_base64 = response.audios[0]
                audio_bytes = base64.b64decode(audio_base64)
            elif hasattr(response, 'audio'):
                audio_base64 = response.audio
                audio_bytes = base64.b64decode(audio_base64)
            else:
                logger.error(f"Unexpected response structure in streaming: {type(response)}")
                return
            
            # Save to temporary file to load with torchaudio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            # Load the audio file
            audio_tensor, sample_rate = torchaudio.load(temp_path)
            self.sr = sample_rate
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Chunk the audio for streaming simulation
            chunk_size = Constants.fetch_constant("tts_chunk_size") or 1024  # samples per chunk
            total_samples = audio_tensor.shape[1]
            
            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                audio_chunk = audio_tensor[:, start_idx:end_idx]
                
                # Convert to buffer format similar to original implementation
                # buffer = io.BytesIO()
                # torchaudio.save(buffer, audio_chunk, sample_rate=self.sr, format="wav")
                # buffer.seek(0)
                # chunk_data = buffer.read()
                yield audio_chunk,self.sr

        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")
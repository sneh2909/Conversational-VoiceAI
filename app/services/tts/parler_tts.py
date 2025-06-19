import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread
import io
import torchaudio
from app.services.tts.tts_interface import TTSInterface
from app.utilities import sken_logger
from app.utilities.constants import Constants

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "parler"})

class ParlerBox(TTSInterface):
    def __init__(self):
        self.device = "cuda:1"
        self.dtype = torch.bfloat16
        self.model_name = "ai4bharat/indic-parler-tts"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(self.model_name).to(self.device, dtype=self.dtype)
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)

        # Extract audio parameters
        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        self.frame_rate = self.model.audio_encoder.config.frame_rate

    def generate(self, text: str, description: str):
        """
        Synchronously generate full audio for given text and description using Parler-TTS.

        Args:
            text (str): The input text to synthesize.
            description (str): The voice style prompt.

        Returns:
            Tuple[int, np.ndarray]: sampling_rate, audio_array
        """
        try:
            # Tokenize inputs
            description_inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
            prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # Generate audio tensor (not streamed)
            audio_tensor = self.model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )

            # Convert to numpy array
            audio_arr = audio_tensor.cpu().numpy().squeeze()
            return self.sampling_rate, audio_arr

        except Exception as e:
            logger.error(f"Error in non-streaming TTS generation: {e}")
            return None, None


    async def generate_audio_stream(self, text: str, description: str=Constants.fetch_constant("parler_description"), chunk_duration_sec: float = 0.5):
        """
        Async version that yields encoded WAV chunks.

        Args:
            text: Text to be converted.
            description: Voice prompt.
            chunk_duration_sec: Time duration for audio chunks.

        Yields:
            Byte chunks of WAV-encoded audio.
        """
        try:
            chunk_duration_sec=1
            play_steps = int(self.frame_rate * chunk_duration_sec)
            streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=play_steps)
            
            # Tokenize description and text prompt
            inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
            prompt = self.tokenizer(text, return_tensors="pt").to(self.device)

            # Build generation arguments
            generation_kwargs = dict(
                input_ids=inputs.input_ids,
                prompt_input_ids=prompt.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_attention_mask=prompt.attention_mask,
                streamer=streamer,
                do_sample=True,
                temperature=1.0,
                min_new_tokens=10,
            )

            # Start generation thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream audio chunks
            for audio_chunk in streamer:
                if audio_chunk.shape[0] == 0:
                    break
                yield audio_chunk,self.sampling_rate
        except Exception as e:
            logger.error(f"Error in Parler-TTS async streaming: {e}")

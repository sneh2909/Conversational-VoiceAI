from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from app.utilities.singletons_factory import SkenSingleton
from app.utilities import sken_logger
import re
import torch
import torchaudio
import numpy as np
import os 
from app.utilities.constants import Constants
from scipy.io import wavfile


logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "nemo"})

class Helper(metaclass=SkenSingleton):
    """
    Added helper functionalities 
    """
    @staticmethod
    def extract_transcriptions(hyps: Hypothesis | str):
        """
        The transcribed_texts returned by CTC and RNNT models are different.
        This method would extract and return the text section of the hypothesis.
        """
        try:
            if isinstance(hyps[0], Hypothesis):
                transcriptions = []
                for hyp in hyps:
                    transcriptions.append(hyp.text)
            else:
                transcriptions = hyps
            return transcriptions
        except Exception as exe:
            logger.error(f"Error in extract_transcriptions: {str(exe)}")
            raise

    @staticmethod
    def calculate_chunk(lookahead_size: int, encoder_step_length: int, sample_rate: int):
        """Calculate Chunk size
        
        Args:
            lookahead_size (int): lookahead size in ms
            encoder_step_length (int): encoder step length in ms
            sample_rate (int): sample rate of signal
        
        Returns:
            chunk size(int)
        """
        chunksize = lookahead_size+encoder_step_length
        chunksize_frame = (sample_rate*chunksize//1000)-1
        return chunksize_frame
    
    @staticmethod
    def contains_hindi(text: str) -> bool:
    # Unicode range for Devanagari script (includes Hindi)
        hindi_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(hindi_pattern.search(text))
    
    @staticmethod
    async def _adjust_sample_rate(original_sample_rate: int, signal: np.ndarray, target_sample_rate: int) -> np.ndarray:
        """ Adjust the sample rate of the audio signal to match the model's sample rate.

        Args:
            sample_rate (int): Sample rate of input audio signal
            signal (np.ndarray): the input audio signal

        Returns:
            np.ndarray: The audio signal with the adjusted sample rate 16000.
        """
        
        resample_transform = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        tensor_signal = torch.from_numpy(signal).float()  # Convert to float32(required by Resample)
        signal = resample_transform(tensor_signal)
        signal = np.array(signal.cpu(), dtype=np.int16)
        return signal
    
    @staticmethod
    async def save_audio_to_file(audio_data, file_name, sampling_rate):
        audio_dir = Constants.fetch_constant("audio_save_folder")
        os.makedirs(audio_dir, exist_ok=True)
        file_path = os.path.join(audio_dir, file_name)
        pcm = np.frombuffer(audio_data, dtype=np.int16)
        wavfile.write(file_path, sampling_rate, pcm)
        return file_path
from app.utilities import sken_logger
from app.utilities.constants import Constants
import numpy as np
class SpeakerClassificationClient:
    """
    Represents a client connected to the server used for audio Steams.

    This class maintains the state for each connected client, including their
    unique identifier, audio buffer, configuration, and a counter for processed audio files.

    Attributes:
        client_id (str): A unique identifier for the client.
        buffer (bytearray): A buffer to store incoming audio data.
        sampling_rate (int): The sampling rate of the audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
    """
    def __init__(self, client_id, sampling_rate, samples_width):
        self.client_id = client_id
        self.buffer = bytearray()
        self.total_samples = 0
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.file_counter = 0
        self.agent_embedding = 0

    def append_audio_data(self, audio_data):
        self.buffer.extend(audio_data)
        self.total_samples += len(audio_data) / self.samples_width
    
    def clear_buffer(self):
        self.buffer.clear()
    
    def set_agent_embedding(self, embedding):
        self.agent_embedding = embedding
        
    def set_call_id(self, call_id):
        self.call_id = call_id
    
    def get_agent_embedding(self) -> np.ndarray:
        """
        This method returns the agent embedding associated with the client.

        Returns:
            np.ndarray: The agent embedding of the client.
        """
        return self.agent_embedding
    
    def get_file_name(self)-> str:
        return f"{self.client_id}_{self.file_counter}.wav"
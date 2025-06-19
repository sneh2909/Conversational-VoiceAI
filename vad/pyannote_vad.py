import os
from pyannote.core import Segment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from app.services.vad.vad_interface import VADInterface
from app.services.client import Client
from app.utilities.constants import Constants
from app.utilities.audio_utils import save_audio_to_file

class PyannoteVAD(VADInterface):
    """
    Pyannote-based implementation of the VADInterface.
    """
    def __init__(self):
        """
        Initializes Pyannote's VAD pipeline.

        Args:
            model_name (str): The model name for Pyannote.
            auth_token (str, optional): Authentication token for Hugging Face.
        """
        access_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
        model = Model.from_pretrained("pyannote/segmentation",use_auth_token = access_token)
        self.vad_pipeline = VoiceActivityDetection(segmentation=model)
        self.vad_pipeline.instantiate(Constants.fetch_constant("pyannote_hyperparameter"))
    
    async def detect_activity(self, client:Client):
        audio_file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
        vad_results = self.vad_pipeline(audio_file_path)
        os.remove(audio_file_path)
        vad_segments = []
        if len(vad_results) > 0:
            for segment in vad_results.itersegments():
                vad_segments.append({"start": segment.start,"end": segment.end,"confidence": 1.0})
        return vad_segments




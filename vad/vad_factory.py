from app.services.vad.pyannote_vad import PyannoteVAD
from app.services.vad.webrtc_vad import WebRTCVAD

class VADFactory:
    """
    Factory for creating instances of VAD systems.
    """

    @staticmethod
    def create_vad_pipeline(type):
        """
        Creates a VAD pipeline based on the specified type.

        Args:
            type (str): The type of VAD pipeline to create (e.g., 'pyannote').
            kwargs: Additional arguments for the VAD pipeline creation.

        Returns:
            VADInterface: An instance of a class that implements VADInterface.
        """
        if type == "pyannote":
            return PyannoteVAD()
        elif type == "webrtc":
            return WebRTCVAD()
        else:
            raise ValueError(f"Unknown VAD pipeline type: {type}")
# from app.services.tts.chatterbox import ChatterBox
from app.services.tts.sarvam import TextToSpeech as SarvamTextToSpeech
from app.services.tts.parler_tts import ParlerBox


class TTSFactory:
    @staticmethod
    def create_tts_pipeline(type):
        if type == "chatterbox":
            # return ChatterBox()
            pass
        if type == "sarvam":
            return SarvamTextToSpeech()
        if type=="parler":
            return ParlerBox()
        else:
            raise ValueError(f"Unknown TTS pipeline type: {type}")
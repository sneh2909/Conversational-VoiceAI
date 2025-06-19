
class TTSInterface:
    
    async def generate_audio(self, text: str):
        """
        Generate audio from text using the TTS model.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
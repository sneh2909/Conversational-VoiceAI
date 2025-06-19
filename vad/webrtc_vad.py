import os
import contextlib
import wave
import webrtcvad
from app.services.vad.vad_interface import VADInterface
from app.services.client import Client
from app.utilities.constants import Constants
from app.utilities.audio_utils import save_audio_to_file

class WebRTCVAD(VADInterface):
    """
    WebRTCVAD-based implementation of the VADInterface.
    """

    def __init__(self):
        """
        Initializes WebRTC's VAD pipeline.
        """
        self.vad = webrtcvad.Vad(3)

    def read_wave(self, path):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate, num_channels

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        while offset + frame_size < len(audio):
            yield audio[offset:offset + frame_size]
            offset += frame_size

    async def detect_activity(self, client: Client):
        audio_file_path = await save_audio_to_file(client.scratch_buffer, client.get_file_name())
        pcm_data, sample_rate, num_channels = self.read_wave(audio_file_path)
        frames = self.frame_generator(30, pcm_data, sample_rate)
        vad_segments = []
        frame_duration = Constants.fetch_constant("webrt_hyperparameter")["frame_duration"]
        timestamp = 0.0
        speech_start = None

        for frame in frames:
            is_speech = self.vad.is_speech(frame, sample_rate)

            if is_speech:
                if speech_start is None:
                    speech_start = timestamp
            else:
                if speech_start is not None:
                    end_time = timestamp
                    vad_segments.append({"start": round(speech_start,2), "end": round(end_time,2), "confidence": 1.0})
                    speech_start = None

            timestamp += frame_duration

        if speech_start is not None:
            end_time = timestamp
            vad_segments.append({"start": round(speech_start,2), "end": round(end_time,2), "confidence": 1.0})

        vad_segments = [d for d in vad_segments if d['end'] - d['start'] >= 1]

        os.remove(audio_file_path)
        
        return vad_segments
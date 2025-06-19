import pyaudio
import wave
import threading
import os
import tempfile
from groq import Groq

class Transcorder:
    def __init__(self, api_key, sample_rate=44100, channels=1, chunk=1024):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.recording = False
        self.frames = []
        self.thread = None
        self.temp_wav = "/home/sneh/projects/voice_to_voice"
        self.client = Groq(api_key=self.api_key)

        self.p = pyaudio.PyAudio()
        self.stream = None

    def _record_audio(self, record_seconds=None):
        """Internal method to record audio until self.recording is False or record_seconds reached."""
        self.frames = []
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=self.channels,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        print("Recording started...")
        if record_seconds:
            # Record for fixed duration (optional)
            for _ in range(0, int(self.sample_rate / self.chunk * record_seconds)):
                if not self.recording:
                    break
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            self.recording = False
        else:
            # Record until stopped externally
            while self.recording:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
        print("Recording stopped.")

    def start_recording(self):
        if self.recording:
            print("Already recording!")
            return
        self.recording = True
        self.thread = threading.Thread(target=self._record_audio)
        self.thread.start()

    def stop_recording(self):
        if not self.recording:
            print("Not recording!")
            return None
        self.recording = False
        self.thread.join()
        self.thread = None

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Save WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            self.temp_wav = f.name
        wf = wave.open(self.temp_wav, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Send to Groq Whisper ASR
        with open(self.temp_wav, "rb") as file:
            response = self.client.audio.transcriptions.create(
                file=(self.temp_wav, file.read()),
                model="distil-whisper-large-v3-en",
                response_format="verbose_json",
            )
        os.remove(self.temp_wav)
        return response.text

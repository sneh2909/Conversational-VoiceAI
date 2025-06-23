from base64 import b64encode
import os
from datetime import datetime
import numpy as np
import audioop
import io
import base64
import torch
import asyncio # Import asyncio to create background tasks
from pydub.silence import detect_silence
from contextlib import asynccontextmanager
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Query
import torchaudio
import soundfile as sf

from app.utilities.constants import Constants
from app.utilities.connectionmanager import ConnectionManager
from app.services.translation_service.indic_trans import IndicTrans
from app.utilities import sken_logger
from app.utilities.helper import Helper
from app.utilities.env_util import EnvironmentVariableRetriever
from app.services.chatbot.chatbot_factory import ChatbotFactory
from app.services.tts.tts_factory import TTSFactory
from app.services.nemo_service.asr_factory import ASRFactory


logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})
        
# Initialize connection manager
manager = ConnectionManager()
lookahead_size = Constants.fetch_constant("model_config")["lookahead_size"] 
encoder_step_length = Constants.fetch_constant("model_config")["encoder_step_length"]
model_path = Constants.fetch_constant("model_config")["model_path"]
model_sample_rate = Constants.fetch_constant("model_config")["sample_rate"]
chatbot_pipeline = ChatbotFactory.create_chatbot(Constants.fetch_constant("chatbot_type"))
tts_pipeline = TTSFactory.create_tts_pipeline(Constants.fetch_constant("tts_type"))
asr_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model
    global indic_translator
    indic_translator = IndicTrans()
    # Use factory to create ASR model
    asr_type = Constants.fetch_constant("asr_type")  # Add this to your constants.yaml
    asr_model = await ASRFactory.create_asr_pipeline(asr_type, lookahead_size)
    
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="app/templates")
root = os.path.dirname(__file__)
app.mount('/static', StaticFiles(directory=r"app/static"), name='static')

# --- Helper Function ---
def encode_audio_to_base64(audio_chunk, sample_rate):
    """Encodes a NumPy audio chunk to a base64 WAV string."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_chunk, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    return audio_b64

# --- HTML Endpoints ---
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/page2")
async def get_page2(request: Request):
    return templates.TemplateResponse("transcript.html", {"request": request})

async def handle_ai_response_and_tts(websocket: WebSocket, client_id: str, transcript: str):
    """
    Handles getting a response from the chatbot, streaming text to the UI,
    and generating/streaming TTS audio sentence by sentence.
    """
    try:
        connection_data = manager.asr_connections[client_id]
        stream = chatbot_pipeline.response(
            query=transcript,
            memory=connection_data.get('memory', []),
            system_prompt=Constants.fetch_constant("chatbot_system_prompt")
        )

        response = {"human": transcript, "AI": ""}
        full_ai_response = ""
        buffered_text_for_tts = ""

        # Iterate through the streaming response from the chatbot
        for chunk in stream:
            text = chunk.content if hasattr(chunk, 'content') else chunk
            if not text or text.lower() == "none":
                continue

            # Stream the text to the UI as it arrives
            full_ai_response += text
            response["AI"] = full_ai_response

            # Buffer text for sentence-based TTS generation
            buffered_text_for_tts += text

            # When a sentence terminator is found, generate audio for the sentence
            if any(p in buffered_text_for_tts for p in ['.', '\n', '!', '?', ':', ',', ';']):
                # Find the position of the last terminator
                last_terminator_pos = -1
                for p in ['.', '\n', '!', '?', ':', ',', ';']:
                    last_terminator_pos = max(last_terminator_pos, buffered_text_for_tts.rfind(p))

                if last_terminator_pos != -1:
                    sentence_to_speak = buffered_text_for_tts[:last_terminator_pos + 1]
                    buffered_text_for_tts = buffered_text_for_tts[last_terminator_pos + 1:]

                    if sentence_to_speak.strip():
                        logger.info(f"Generating audio for sentence: '{sentence_to_speak.strip()}'")
                        tts_stream = tts_pipeline.generate_audio_stream(sentence_to_speak.strip())
                        async for audio_chunk, sample_rate in tts_stream:
                            audio_b64 = encode_audio_to_base64(audio_chunk, sample_rate)
                            await websocket.send_json({"type": "audio_chunk", "audio_data": audio_b64})

        # After the loop, process any remaining text in the buffer
        if buffered_text_for_tts.strip():
            logger.info(f"Flushing remaining TTS buffer: '{buffered_text_for_tts.strip()}'")
            tts_stream = tts_pipeline.generate_audio_stream(buffered_text_for_tts.strip())
            async for audio_chunk, sample_rate in tts_stream:
                audio_b64 = encode_audio_to_base64(audio_chunk, sample_rate)
                await websocket.send_json({"type": "audio_chunk", "audio_data": audio_b64})

        await websocket.send_json(response)

        # Save the full conversation context at the end
        if full_ai_response.strip():
            manager.asr_connections[client_id]['memory'].save_context({"human": transcript}, {"AI": full_ai_response})
            logger.info("Saved conversation context.")

    except Exception as e:
        logger.error(f"Error in AI response/TTS handler for {client_id}: {e}", exc_info=True)

# --- Background Task for Processing a Single Utterance ---
async def process_user_utterance(websocket: WebSocket, client_id: str, audio_to_transcribe: np.ndarray, model_sample_rate: int):
    """
    Transcribes a chunk of audio and then passes it to the AI/TTS handler.
    """
    try:
        logger.info(f"Background Task: Transcribing {len(audio_to_transcribe)/model_sample_rate:.2f}s of audio for {client_id}")
        transcript = await asr_model.transcribe_chunk(audio_to_transcribe, client_id)
        await asr_model.clear_cache(client_id)
        await websocket.send_json({
        "start": 0,
        "end": 2,
        "text": transcript}) 
        

        if not transcript or not transcript.strip() or transcript.lower() == "none" or transcript==".": 
            logger.info("Background Task: Transcription was empty, skipping.")
            return

        logger.info(f"Background Task: Transcript for {client_id}: {transcript}")
        
        # Call the dedicated handler for AI and TTS processing
        # await handle_ai_response_and_tts(websocket, client_id, transcript)

    except Exception as e:
        logger.error(f"Error in user utterance processing task for {client_id}: {e}", exc_info=True)

# --- Main WebSocket Endpoint ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint_transcript(
    websocket: WebSocket,
    client_id: str,
    sample_rate: int = Query(...),
    format: str = Query(...)
):
    try:
        await manager.connect(client_id=client_id, websocket=websocket, sample_rate=sample_rate, connection_type="asr", audio_format=format)
        logger.info(f"WebSocket connection established for client {client_id}")
        await asr_model.init_client_cache(client_id)
        connection_data = manager.asr_connections[client_id]

        SILENCE_THRESHOLD_MS = 700
        MIN_AUDIO_DURATION_S = 1.0
        SILENCE_DBFS_THRESH = -40

        while True:
            audio_bytes = await websocket.receive_bytes()
            connection_data['last_activity'] = datetime.now()

            if connection_data['format'] == 'pcm':
                signal = np.frombuffer(audio_bytes, dtype=np.float32)
            else:
                logger.warning(f"Unsupported audio format: {format}")
                continue

            if sample_rate != model_sample_rate:
                signal = await Helper._adjust_sample_rate(sample_rate, signal, model_sample_rate)

            connection_data['buffer'] = np.concatenate((connection_data['buffer'], signal))

            pcm_audio_data = (connection_data['buffer'] * 32767).astype(np.int16).tobytes()
            audio_segment = AudioSegment(data=pcm_audio_data, sample_width=2, frame_rate=model_sample_rate, channels=1)

            if len(audio_segment) > SILENCE_THRESHOLD_MS:
                last_chunk_for_vad = audio_segment[-SILENCE_THRESHOLD_MS:]
                silence = detect_silence(last_chunk_for_vad, min_silence_len=SILENCE_THRESHOLD_MS, silence_thresh=SILENCE_DBFS_THRESH)

                if silence and (len(audio_segment) / 1000.0) > MIN_AUDIO_DURATION_S:
                    audio_to_transcribe = connection_data['buffer'].copy()
                    connection_data['buffer'] = np.array([], dtype=np.float32)
                    
                    # Launch the processing in a non-blocking background task
                    asyncio.create_task(
                        process_user_utterance(websocket, client_id, audio_to_transcribe, model_sample_rate)
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for client {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint for {client_id}: {e}", exc_info=True)
    finally:
        if manager.is_client_connected(client_id, "asr"):
            await asr_model.remove_client(client_id)
            manager.disconnect(client_id, "asr")
            logger.info(f"Cleaned up resources for client {client_id}")
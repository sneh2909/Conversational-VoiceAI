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
silence_count_threshold = Constants.fetch_constant("model_config")["silence_count"]
chatbot_pipeline = ChatbotFactory.create_chatbot(Constants.fetch_constant("chatbot_type"))
tts_pipeline = TTSFactory.create_tts_pipeline(Constants.fetch_constant("tts_type"))
asr_model = None

def silence_calculator():
    chunk_size_s = round((lookahead_size + encoder_step_length)/1000,2)
    silence_threshold_s = chunk_size_s*silence_count_threshold
    return silence_threshold_s
silence_threshold_s  = silence_calculator()

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
                        if Constants.fetch_constant("tts_type")=="sarvam":
                            audio_array, sample_rate = tts_pipeline.generate_audio(sentence_to_speak.strip())
                            audio_b64 = encode_audio_to_base64(audio_array,sample_rate=sample_rate)
                            await websocket.send_json({
                                    "type": "audio_chunk",
                                    "audio_data": audio_b64
                                })
                        else:
                            logger.info(f"Generating audio for sentence: '{sentence_to_speak.strip()}'")
                            tts_stream = tts_pipeline.generate_audio_stream(sentence_to_speak.strip())
                            async for audio_chunk, sample_rate in tts_stream:
                                audio_b64 = encode_audio_to_base64(audio_chunk, sample_rate)
                                await websocket.send_json({"type": "audio_chunk", "audio_data": audio_b64})

        # After the loop, process any remaining text in the buffer
        if buffered_text_for_tts.strip():
            if Constants.fetch_constant("tts_type")=="sarvam":
                audio_array, sample_rate = tts_pipeline.generate_audio(buffered_text_for_tts.strip())
                audio_b64 = encode_audio_to_base64(audio_array,sample_rate=sample_rate)
                await websocket.send_json({
                        "type": "audio_chunk",
                        "audio_data": audio_b64
                    })
            else:
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

async def process_user_utterance(websocket: WebSocket, client_id: str, audio_to_transcribe: np.ndarray, model_sample_rate: int):
    """
    Transcribes a chunk of audio and then passes it to the AI/TTS handler.
    """
    try:
        logger.info(f"Background Task: Transcribing {len(audio_to_transcribe)/model_sample_rate:.2f}s of audio for {client_id}")
        transcript = await asr_model.transcribe_chunk(audio_to_transcribe, client_id)
        await asr_model.clear_cache(client_id)
        

        if not transcript or not transcript.strip() or transcript.lower() == "none" or transcript==".": 
            logger.info("Background Task: Transcription was empty, skipping.")
            return
        # await websocket.send_json({
        #     "type": "transcript",
        #     "text": transcript}) 

        logger.info(f"Background Task: Transcript for {client_id}: {transcript}")
        if len(transcript.split()) > 2:
        # Call the dedicated handler for AI and TTS processing
            await handle_ai_response_and_tts(websocket, client_id, transcript)

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
        if Constants.fetch_constant("asr_type")=="nemo":
            await perform_nemo(websocket, client_id, sample_rate, format)
        else:
            await perform_whisper(websocket, client_id, sample_rate, format)

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for client {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint for {client_id}: {e}", exc_info=True)
    finally:
        await manager.disconnect(client_id, connection_type="asr")
        await asr_model.remove_client(client_id)

async def perform_whisper(websocket, client_id, sample_rate, format):
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
            # await manager.send_message(client_id, transcript)

async def perform_nemo(websocket, client_id, sample_rate, format):
    CHUNKSIZE = Helper.calculate_chunk(lookahead_size, encoder_step_length, model_sample_rate)
    await manager.connect(client_id=client_id, websocket=websocket, sample_rate=sample_rate, chunk_size=CHUNKSIZE, connection_type="asr", audio_format=format)
    logger.info(f"WebSocket connection established for client {client_id} with sample_rate {sample_rate}")
    await asr_model.init_client_cache(client_id)

    while True:
        audio_bytes = await websocket.receive_bytes()
        connection_data = manager.asr_connections[client_id]
        connection_data['last_activity'] = datetime.now()

        if format == 'mulaw':
            audio_bytes = audioop.ulaw2lin(audio_bytes, 2)

        signal = np.frombuffer(audio_bytes, dtype=np.int16)
        # if sample_rate != model_sample_rate:
        #     signal = await Helper._adjust_sample_rate(sample_rate, signal, model_sample_rate)

        connection_data['buffer'] = np.concatenate((connection_data['buffer'], signal))

        if len(connection_data['buffer']) >= connection_data['chunk_size']:
            chunk = connection_data['buffer'][:connection_data['chunk_size']]
            connection_data['buffer'] = connection_data['buffer'][connection_data['chunk_size']:]

            asyncio.create_task(
                process_user_utterance_nemo(websocket, client_id, chunk)
            )


async def process_user_utterance_nemo(websocket, client_id, chunk: np.ndarray):
    connection_data = manager.asr_connections[client_id]

    try:
        transcript, silence_detected, duration = await process_audio(chunk, client_id, manager.asr_connections)
        connection_data['current_time'] += duration

        if connection_data['snippet_start_time'] is None:
            connection_data['snippet_start_time'] = connection_data['current_time'] - duration

        if silence_detected:
            if transcript:
                await websocket.send_json({
                    "start": round(connection_data['snippet_start_time'], 2),
                    "end": round(connection_data['current_time'] - silence_threshold_s, 2),
                    "text": transcript
                })
                connection_data['snippet_start_time'] = None

            if len(transcript.split()) < 2:
                return

            await handle_ai_response_and_tts(websocket, client_id, transcript)

    except Exception as e:
        logger.error(f"Error in Nemo utterance processing task for {client_id}: {e}", exc_info=True)

def encode_audio_to_base64(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """Convert NumPy audio to WAV and return base64-encoded string."""
    # Save audio chunk to in-memory buffer
    if audio_array.ndim == 1:
        audio_array=torch.tensor(audio_array).unsqueeze(0)  # [1, N]
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_array, sample_rate=sample_rate, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

async def process_audio(signal, client_id:str, active_connections:dict):
    """
    Processes a chunk of audio data, detects silence, and transcribes the audio. 
    If silence is detected for a specified threshold, it clears the cache and transcribes the audio.

    Args:
        signal (numpy.ndarray): The audio data to be processed.
        client_id (str): A unique identifier for the client who sent the audio data.
        active_connections (dict): A dictionary tracking the active WebSocket connections and the state of each client, 
                                   including buffers, silence counts, and WebSocket connections.

    Returns:
        tuple: A tuple containing:
            - str: The transcribed text from the processed audio chunk.
            - bool: A flag indicating if silence was detected (True if silence detected, False otherwise).
            - float: The duration of the processed audio segment in seconds.

    Raises:
        Exception: Logs an error if any exception occurs during audio processing and returns empty string and False.
    """
    try:
        audio_segment = AudioSegment(signal.tobytes(), frame_rate=model_sample_rate, sample_width=signal.dtype.itemsize, 
                                   channels=1)
        
        silence_chunks = detect_silence(audio_segment, min_silence_len=160, silence_thresh=-45)
        if silence_chunks:
            active_connections[client_id]["silence_count"] += 1
        else:
            active_connections[client_id]["silence_count"] = 0
        if active_connections[client_id]["silence_count"] > silence_count_threshold:
            logger.info(f"client id {client_id} Silence detected, clearing cache.")
            text = await asr_model.transcribe_chunk(signal, client_id) 
            # if Constants.fetch_constant("asr_type") == "nemo" else await whisper_asr.transcribe_audio(audio_segment)
            await asr_model.clear_cache(client_id)
            active_connections[client_id]["silence_count"] = 0
            return text, True, audio_segment.duration_seconds
        else:
            text = await asr_model.transcribe_chunk(signal, client_id) 
            return text, False, audio_segment.duration_seconds
            
    except Exception as exe:
        logger.error(f"Error during processing audio: {exe}")
        return "", False,0
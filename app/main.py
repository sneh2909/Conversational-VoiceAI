from base64 import b64encode
import os
from datetime import datetime
import numpy as np
import audioop
import io
import base64
import torch
from pydub.silence import detect_silence
from contextlib import asynccontextmanager
from pydub import AudioSegment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Query
import torchaudio
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

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/page2")
async def get_page2(request: Request):
    return templates.TemplateResponse("transcript.html", {"request": request})

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
                    # --- Transcription ---
                    transcript = await asr_model.transcribe_chunk(audio_to_transcribe, client_id)
                    await asr_model.clear_cache(client_id) # Clear text buffer after transcription

                    if not transcript or not transcript.strip() or transcript == "" or transcript==".":
                        logger.info("VAD: Transcription was empty, skipping.")
                        continue
                    
                    logger.info(f"VAD: Transcript for {client_id}: {transcript}")

                    stream = chatbot_pipeline.response(
                        query=transcript,
                        memory=connection_data['memory'],
                        system_prompt=Constants.fetch_constant("chatbot_system_prompt")
                    )

                    response = {
                        "human": transcript,
                        "AI": ""
                    }
                    buffered_text = ""
                    for chunk in stream:
                    # for chunk in chatbot_pipeline.response_with_tools(
                    #         query="Tell me about product features",
                    #         memory=connection_data['memory'],
                    #         org_id="1",
                    #         product_ids=[1]
                    #     ):

                        print(chunk.content, end="")
                        
                        text = chunk.content
                        if text == "" or text == "None" or text is None:
                            continue

                        response["AI"] += text
                        buffered_text += text  # No need to strip or add spaces

                        # Check if a sentence ends (period detected)
                        if any(p in buffered_text for p in ['.', '\n', '!', '?', ':', ',', ';']):
                            # Split the buffered text at the last period
                            sentences = buffered_text.rsplit('.', 1)
                            complete_sentence = sentences[0].strip() + '.'
                            remaining_buffer = sentences[1] if len(sentences) > 1 else ""

                            # Generate and send audio
                            if complete_sentence.strip():
                                async for audio_chunk,sample_rate in tts_pipeline.generate_audio_stream(buffered_text):
                                    # sd.play(audio_chunk, samplerate=sample_rate)
                                    audio_b64 = encode_audio_to_base64(audio_chunk,sample_rate=sample_rate)
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "audio_data": audio_b64
                                    })

                            # Update buffer with leftover (partial) sentence
                            buffered_text = remaining_buffer
                        
                        # Count words in the buffer
                    #     word_count = len(buffered_text.split())

                    #     if word_count >= 7:  # You can change this to 5 or 10 as needed
                    #         # Generate and send audio
                    #         audio_array, sample_rate = tts_pipeline.generate_audio(buffered_text)
                    #         audio_b64 = encode_audio_to_base64(audio_array, sample_rate=sample_rate)

                    #         await websocket.send_json({
                    #             "type": "audio_chunk",
                    #             "audio_data": audio_b64
                    #         })
                    #         # Clear the buffer after sending
                    #         buffered_text = ""
                        

                    # Optionally flush remaining buffer
                    if buffered_text.strip():
                        async for audio_chunk,sample_rate in tts_pipeline.generate_audio_stream(buffered_text):
                            audio_b64 = encode_audio_to_base64(audio_chunk,sample_rate=sample_rate)
                            await websocket.send_json({
                                "type": "audio_chunk",
                                "audio_data": audio_b64
                            })
                    manager.asr_connections[client_id]['memory'].save_context({"human":transcript},{"AI":response["AI"]})

                    await websocket.send_json(response)                
            # await manager.send_message(client_id, transcript)
    except WebSocketDisconnect as exe:
        logger.error(f"WebSocket connection closed for client {client_id}: {exe}")
    except Exception as exe:
        logger.error(f"Error Occurred for client {client_id}: {exe}", exc_info=True)
    finally:
        await manager.disconnect(client_id, connection_type="asr")
        await asr_model.remove_client(client_id)

def encode_audio_to_base64(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    """Convert NumPy audio to WAV and return base64-encoded string."""
    # Save audio chunk to in-memory buffer
    if audio_array.ndim == 1:
        audio_array=torch.tensor(audio_array).unsqueeze(0)  # [1, N]
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_array, sample_rate=sample_rate, format="wav")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

async def generate_transcripts(websocket, client_id, connection_data, signal):
    while len(connection_data['buffer']) > connection_data['chunk_size']:
        chunk = connection_data['buffer'][:connection_data['chunk_size']]
        connection_data['buffer'] = connection_data['buffer'][connection_data['chunk_size']:]
        transcript, silence_detected, duration = await process_audio(signal, client_id, manager.asr_connections)
        connection_data['current_time'] += duration

        if connection_data['snippet_start_time'] == None:
            connection_data['snippet_start_time'] = connection_data['current_time']-duration

        if silence_detected:
            if transcript:
                # if Helper.contains_hindi(transcript):
                    # transcript = await indic_translator.get_translations([transcript])
                await websocket.send_json({
                        "start": round(connection_data['snippet_start_time'], 2),
                        "end": round(connection_data['current_time']-silence_threshold_s, 2),
                        "text": transcript})    
            connection_data['snippet_start_time'] = None
        # await manager.send_message(client_id,  transcript, connection_type="asr")
            return transcript


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
        # Normaliz`e float32 to int16 range
        int16_signal = (signal * 32767).astype(np.int16)

        # Create AudioSegment from int16 signal
        audio_segment = AudioSegment(
            data=int16_signal.tobytes(),
            sample_width=int16_signal.dtype.itemsize,  # 2 bytes for int16
            frame_rate=model_sample_rate,
            channels=1
        )
         # Detect silence (adjust threshold and duration as needed)
        silence_chunks = detect_silence(
                audio_segment,
                min_silence_len=160,        # in ms
                silence_thresh=-45,         # in dBFS
            )
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
    
    

    # """
    # Set the language preference for a specific client
    # """
    # client_id = websocket.client.id
    # if language not in LANGUAGE_CONFIGS:
    #     await websocket.send_text(f"Unsupported language: {language}. Supported languages: {list(LANGUAGE_CONFIGS.keys())}")
    #     return

    # client_languages[client_id] = language
    # if hasattr(manager, 'asr_connections') and client_id in manager.asr_connections:
    #     manager.asr_connections[client_id]['language'] = language
    #     manager.asr_connections[client_id]['system_prompt'] = LANGUAGE_CONFIGS[language]['system_prompt']
    #     logger.info(f"Updated language for active client {client_id} to {language}")

    # logger.info(f"Language changed for client {client_id}: {language}")
    # await websocket.send_text(f"Language successfully changed to {LANGUAGE_CONFIGS[language]['name']}")
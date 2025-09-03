// Unified WebSocket Audio Client for Whisper and Nemo
// Configuration object to switch between models
const CONFIG = {
    // Change this to switch between 'whisper' and 'nemo'
    model: 'whisper', // or 'nemo'
    
    // Audio settings
    sampleRate: 16000,
    bufferSize: 16384,
    
    // WebSocket settings
    getWebSocketUri: function() {
        const client_id = Date.now();
        const baseUrl = window.location.href.replace('index.jsp', '').replace('http', 'ws');
        const modifiedUrl = 'ws://' + baseUrl.split("/")[2] + '/';
        return `${modifiedUrl}ws/${client_id}?sample_rate=${this.sampleRate}&format=pcm&model=${this.model}`;
    }
};

// Global variables
let websocket;
let context;
let processor;
let globalStream;
let startTime;
let endTime;
let isRecording = false;

// Audio playback management
let audioQueue = [];
let isPlayingAudio = false;
let currentAudio = null;

// Model-specific handlers
const modelHandlers = {
    whisper: {
        processMessage: function(event) {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "audio_chunk") {
                    handleAudioChunk(data.audio_data);
                } else if (data.human || data.AI) {
                    updateTranscription(data);
                } else {
                    updateTranscription(JSON.stringify(data, null, 2));
                }
            } catch (e) {
                console.error("Received non-JSON message or parse error:", event.data, e);
            }
        },
        
        processAudio: function(e) {
            // Whisper expects raw Float32 PCM data
            const pcmData = e.inputBuffer.getChannelData(0);
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(pcmData.buffer);
            }
        },
        
        updateTranscription: function(data) {
            const transcriptionDiv = document.getElementById('transcripts');
            
            if (typeof data === 'string') {
                transcriptionDiv.innerHTML += data + '<br/>';
            } else {
                // Handle structured data with human/AI pairs
                let utteranceId = `utterance-${Date.now()}`;
                if (data.human) {
                    utteranceId = `utterance-${data.human.replace(/\s+/g, '-')}`;
                }
                
                let utteranceDiv = document.getElementById(utteranceId);
                
                if (!utteranceDiv && data.human) {
                    utteranceDiv = document.createElement('div');
                    utteranceDiv.id = utteranceId;
                    utteranceDiv.className = 'utterance';
                    utteranceDiv.innerHTML = `<strong>Human:</strong> <span class="human-text">${data.human}</span><br><strong>AI:</strong> <span class="ai-text"></span>`;
                    transcriptionDiv.appendChild(utteranceDiv);
                }
                
                if (utteranceDiv && data.AI) {
                    const aiTextSpan = utteranceDiv.querySelector('.ai-text');
                    aiTextSpan.innerHTML += data.AI;
                }
            }
            
            transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
        }
    },
    
    nemo: {
        processMessage: function(event) {
            console.log("Message from server:", typeof event.data, event.data);
            
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === "audio_chunk") {
                    console.log("Received audio chunk, adding to queue");
                    handleAudioChunk(data.audio_data);
                    updateAudioStatus("Received audio chunk");
                } else if (data.human && data.AI) {
                    updateTranscription(`<strong>Human:</strong> ${data.human}<br><strong>AI:</strong> ${data.AI}`);
                } else {
                    updateTranscription(JSON.stringify(data, null, 2));
                }
            } catch (e) {
                const transcript_data = event.data;
                updateTranscription(transcript_data);
            }
        },
        
        processAudio: function(e) {
            // Nemo expects downsampled Int16 data
            const inputSampleRate = context.sampleRate;
            const outputSampleRate = CONFIG.sampleRate;
            
            const left = e.inputBuffer.getChannelData(0);
            const downsampledBuffer = downsampleBuffer(left, inputSampleRate, outputSampleRate);
            const audioData = convertFloat32ToInt16(downsampledBuffer);
            
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(audioData);
            }
        },
        
        updateTranscription: function(transcript_data) {
            const transcriptionDiv = document.getElementById('transcripts');
            transcriptionDiv.innerHTML += transcript_data + '<br/>';
            transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
            console.log(transcript_data);
        }
    }
};

function initWebSocket() {
    const websocketAddress = CONFIG.getWebSocketUri();
    console.log(`Initializing WebSocket for ${CONFIG.model} model:`, websocketAddress);
    
    if (!websocketAddress) {
        console.log("WebSocket address is required.");
        return;
    }
    
    websocket = new WebSocket(websocketAddress);
    websocket.binaryType = "arraybuffer";

    websocket.onopen = () => {
        console.log(`WebSocket connection established for ${CONFIG.model}`);
        document.getElementById('startButton').disabled = false;
        updateModelStatus();
    };

    websocket.onclose = event => {
        console.log("WebSocket connection closed", event);
        alert("Connection lost. Please refresh the page.");
        document.getElementById('startButton').disabled = true;
        document.getElementById('endButton').disabled = true;
    };

    websocket.onmessage = event => {
        // Use model-specific message handler
        modelHandlers[CONFIG.model].processMessage(event);
    };
    
    websocket.onerror = error => {
        console.error("WebSocket Error:", error);
        alert("A connection error occurred. Please check the console and refresh the page.");
    };
}

function handleAudioChunk(base64AudioData) {
    console.log("Processing audio chunk, length:", base64AudioData.length);
    audioQueue.push(base64AudioData);
    updateAudioQueueStatus();
    
    if (!isPlayingAudio) {
        playNextAudio();
    }
}

async function playNextAudio() {
    if (audioQueue.length === 0) {
        isPlayingAudio = false;
        updateAudioStatus("Audio queue empty");
        return;
    }

    isPlayingAudio = true;
    const base64AudioData = audioQueue.shift();
    updateAudioQueueStatus();
    updateAudioStatus("Playing AI response...");

    try {
        const audioDataUrl = createAudioDataUrl(base64AudioData);
        currentAudio = new Audio(audioDataUrl);

        currentAudio.onended = () => {
            currentAudio = null;
            updateAudioStatus("Playback finished");
            playNextAudio();
        };

        currentAudio.onerror = (error) => {
            console.error("Audio playback error:", error);
            updateAudioStatus("Audio error");
            currentAudio = null;
            setTimeout(playNextAudio, 100);
        };

        await currentAudio.play();
    } catch (error) {
        console.error("Error playing audio:", error);
        updateAudioStatus("Playback error: " + error.message);
        currentAudio = null;
        setTimeout(playNextAudio, 100);
    }
}

function createAudioDataUrl(base64Data) {
    let mimeType = 'audio/wav'; // Default
    
    try {
        const binaryString = atob(base64Data.substring(0, 100));
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        const header = Array.from(bytes.slice(0, 12)).map(b => b.toString(16).padStart(2, '0')).join('');
        
        if (header.includes('fff3') || header.includes('fff2')) {
            mimeType = 'audio/mp3';
        } else if (header.includes('4f676753')) {
            mimeType = 'audio/ogg';
        } else if (header.includes('52494646')) {
            mimeType = 'audio/wav';
        } else if (header.includes('664c6143')) {
            mimeType = 'audio/flac';
        }
    } catch (e) {
        console.warn("Could not detect audio format, using default wav:", e);
    }
    
    console.log("Detected MIME type:", mimeType);
    return `data:${mimeType};base64,${base64Data}`;
}

function updateTranscription(data) {
    // Use model-specific transcription handler
    modelHandlers[CONFIG.model].updateTranscription(data);
}

function updateAudioStatus(status) {
    const statusDiv = document.getElementById('audioStatus');
    if (statusDiv) {
        statusDiv.textContent = status;
        console.log("Audio Status:", status);
    }
}

function updateAudioQueueStatus() {
    const queueDiv = document.getElementById('audioQueue');
    if (queueDiv) {
        queueDiv.textContent = `Queue: ${audioQueue.length} items`;
    }
}

function updateModelStatus() {
    const modelDiv = document.getElementById('currentModel');
    if (modelDiv) {
        modelDiv.textContent = `Current Model: ${CONFIG.model.toUpperCase()}`;
    }
}

function switchModel() {
    // Stop current recording if active
    if (isRecording) {
        stopRecording();
    }
    
    // Stop all audio
    stopAllAudio();
    
    // Close current WebSocket
    if (websocket) {
        websocket.close();
    }
    
    // Switch model
    CONFIG.model = CONFIG.model === 'whisper' ? 'nemo' : 'whisper';
    
    // Clear transcripts
    document.getElementById('transcripts').innerHTML = '';
    
    // Reinitialize WebSocket with new model
    setTimeout(() => {
        initWebSocket();
        console.log(`Switched to ${CONFIG.model} model`);
    }, 1000);
}

function startRecording() {
    if (isRecording) return;
    isRecording = true;
    startTime = Date.now();

    stopAllAudio();
    updateAudioStatus("Recording...");
    
    // Clear transcripts for new session
    if (CONFIG.model === 'whisper') {
        document.getElementById('transcripts').innerHTML = '';
    }

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    
    // Create AudioContext with appropriate sample rate
    const contextOptions = CONFIG.model === 'whisper' ? 
        { sampleRate: CONFIG.sampleRate } : {};
    
    context = new AudioContext(contextOptions);
    
    const mediaConstraints = { 
        audio: true, 
        video: false 
    };
    
    // Add sample rate constraint for Nemo
    if (CONFIG.model === 'nemo') {
        mediaConstraints.audio = { sampleRate: CONFIG.sampleRate };
    }
    
    navigator.mediaDevices.getUserMedia(mediaConstraints).then(stream => {
        globalStream = stream;
        const input = context.createMediaStreamSource(stream);
        
        processor = context.createScriptProcessor(CONFIG.bufferSize, 1, 1);
        processor.onaudioprocess = e => {
            // Use model-specific audio processing
            modelHandlers[CONFIG.model].processAudio(e);
        };
        
        input.connect(processor);
        processor.connect(context.destination);
    }).catch(error => {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please grant permission and refresh.');
        isRecording = false;
    });

    document.getElementById('startButton').disabled = true;
    document.getElementById('endButton').disabled = false;
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    if (globalStream) {
        globalStream.getTracks().forEach(track => track.stop());
    }
    if (processor) {
        processor.disconnect();
        processor = null;
    }
    if (context && context.state !== 'closed') {
        context.close().then(() => context = null);
    }

    document.getElementById('startButton').disabled = false;
    document.getElementById('endButton').disabled = true;
    updateMetrics();
    updateAudioStatus("Recording stopped.");
}

function stopAllAudio() {
    audioQueue = [];
    updateAudioQueueStatus();
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    isPlayingAudio = false;
    updateAudioStatus("Audio stopped");
}

// Utility functions for Nemo model
function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
    if (inputSampleRate === outputSampleRate) return buffer;
    
    const sampleRateRatio = inputSampleRate / outputSampleRate;
    const newLength = Math.round(buffer.length / sampleRateRatio);
    const result = new Float32Array(newLength);
    let offsetResult = 0, offsetBuffer = 0;
    
    while (offsetResult < result.length) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
        let accum = 0, count = 0;
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    return result;
}

function convertFloat32ToInt16(buffer) {
    const buf = new Int16Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
        buf[i] = Math.max(-1, Math.min(1, buffer[i])) * 0x7FFF;
    }
    return buf.buffer;
}

function updateMetrics() {
    const durationSpan = document.getElementById('duration');
    const duration = Math.floor((Date.now() - startTime) / 1000);
    durationSpan.textContent = `${duration}s`;
    document.getElementById('metrics').style.display = 'block';
}

// Additional utility functions
async function clearCache() {
    try {
        const response = await fetch('/clearcache', {
            method: 'GET',
        });

        if (response.ok) {
            console.log('Cache cleared successfully');
            alert('Cache cleared successfully');
        } else {
            alert('Failed to clear cache');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error clearing cache');
    }
}

function clearAudioQueue() {
    stopAllAudio();
    alert('Audio queue cleared');
}

function testAudio() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
    
    updateAudioStatus("Test beep played");
}

// Initialize on page load
window.onload = function() {
    initWebSocket();
    updateModelStatus();
    
    // Add model switch button functionality
    const switchButton = document.getElementById('switchModelButton');
    if (switchButton) {
        switchButton.addEventListener('click', switchModel);
    }
};
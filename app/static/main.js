// let websocket;
// let context;
// let processor;
// let globalStream;
// let startTime;
// let endTime;
// const client_id = Date.now()
// const actualSampleRate = 16000;   
// const baseUrl = window.location.href.replace('index.jsp', '').replace('http', 'ws');
// const modifedurl = 'ws://'+baseUrl.split("/")[2]+'/'
// const websocket_uri = `${modifedurl}ws/${client_id}?sample_rate=${actualSampleRate}&format=pcm`;
// const bufferSize = 16384;
// let isRecording = false;

// // Audio playback queue management
// let audioQueue = [];
// let isPlayingAudio = false;
// let currentAudio = null;

// function initWebSocket() {
//     const websocketAddress = websocket_uri;
//     if (!websocketAddress) {
//         console.log("WebSocket address is required.");
//         return;
//     }
//     websocket = new WebSocket(websocketAddress);
//     websocket.onopen = () => {
//         console.log("WebSocket connection established");
//     };
//     websocket.onopen = function (e) {
//         if (websocket.readyState === 1) {
//             console.log("coming to socket ready state")
//         }
//     };
//     websocket.onclose = event => {
//         console.log("WebSocket connection closed", event);
//     };

//     websocket.onmessage = event => {
//         console.log("Message from server:", typeof event.data, event.data);
        
//         try {
//             // Try to parse as JSON first
//             const data = JSON.parse(event.data);
            
//             if (data.type === "audio_chunk") {
//                 // Handle audio chunk
//                 console.log("Received audio chunk, adding to queue");
//                 handleAudioChunk(data.audio_data);
//                 updateAudioStatus("Received audio chunk");
//             } else if (data.human && data.AI) {
//                 // Handle transcript response
//                 updateTranscription(`<strong>Human:</strong> ${data.human}<br><strong>AI:</strong> ${data.AI}`);
//             } else {
//                 // Handle other JSON responses
//                 updateTranscription(JSON.stringify(data, null, 2));
//             }
//         } catch (e) {
//             // If not JSON, treat as plain text transcript
//             const transcript_data = event.data;
//             updateTranscription(transcript_data);
//         }
//     };
// }

// function handleAudioChunk(base64AudioData) {
//     console.log("Processing audio chunk, length:", base64AudioData.length);
    
//     // Add audio to queue
//     audioQueue.push(base64AudioData);
//     updateAudioQueueStatus();
    
//     // Start playing if not already playing
//     if (!isPlayingAudio) {
//         playNextAudio();
//     }
// }

// async function playNextAudio() {
//     if (audioQueue.length === 0) {
//         isPlayingAudio = false;
//         updateAudioStatus("Audio queue empty");
//         updateAudioQueueStatus();
//         return;
//     }
    
//     isPlayingAudio = true;
//     const base64AudioData = audioQueue.shift();
//     updateAudioQueueStatus();
//     updateAudioStatus("Playing audio...");
    
//     try {
//         // Stop current audio if playing
//         if (currentAudio) {
//             currentAudio.pause();
//             currentAudio = null;
//         }

//         // Detect audio format and create appropriate data URL
//         const audioDataUrl = createAudioDataUrl(base64AudioData);
//         console.log("Created audio data URL:", audioDataUrl.substring(0, 100) + "...");
        
//         // Create audio element
//         currentAudio = new Audio();
        
//         // Set up event listeners
//         currentAudio.onloadstart = () => {
//             console.log("Audio loading started");
//             updateAudioStatus("Loading audio...");
//         };
        
//         currentAudio.oncanplay = () => {
//             console.log("Audio can start playing");
//             updateAudioStatus("Audio ready to play");
//         };
        
//         currentAudio.onplay = () => {
//             console.log("Audio started playing");
//             updateAudioStatus("Playing audio");
//         };
        
//         currentAudio.onended = () => {
//             console.log("Audio playback finished");
//             currentAudio = null;
//             updateAudioStatus("Audio finished");
//             // Play next audio in queue
//             playNextAudio();
//             // setTimeout(() => playNextAudio(), 10); // Small delay between chunks
//         };
        
//         currentAudio.onerror = (error) => {
//             console.error("Audio playback error:", error);
//             console.error("Audio error details:", currentAudio.error);
//             updateAudioStatus("Audio error: " + (currentAudio.error ? currentAudio.error.message : "Unknown error"));
//             currentAudio = null;
//             // Continue with next audio even if current one fails
//             setTimeout(() => playNextAudio(), 10);
//         };
        
//         currentAudio.onpause = () => {
//             console.log("Audio paused");
//         };
        
//         // Set the source
//         currentAudio.src = audioDataUrl;
        
//         // Start playing
//         const playPromise = currentAudio.play();
        
//         if (playPromise !== undefined) {
//             await playPromise;
//             console.log("Audio playing successfully");
//         }
        
//     } catch (error) {
//         console.error("Error playing audio:", error);
//         updateAudioStatus("Playback error: " + error.message);
//         currentAudio = null;
//         // Continue with next audio even if current one fails
//         setTimeout(() => playNextAudio(), 100);
//     }
// }

// function createAudioDataUrl(base64Data) {
//     // Try to detect audio format from base64 data
//     let mimeType = 'audio/wav'; // Default
    
//     try {
//         // Decode a small portion to check headers
//         const binaryString = atob(base64Data.substring(0, 100));
//         const bytes = new Uint8Array(binaryString.length);
//         for (let i = 0; i < binaryString.length; i++) {
//             bytes[i] = binaryString.charCodeAt(i);
//         }
        
//         // Check for common audio format signatures
//         const header = Array.from(bytes.slice(0, 12)).map(b => b.toString(16).padStart(2, '0')).join('');
        
//         if (header.includes('fff3') || header.includes('fff2')) {
//             mimeType = 'audio/mp3';
//         } else if (header.includes('4f676753')) { // OggS
//             mimeType = 'audio/ogg';
//         } else if (header.includes('52494646')) { // RIFF
//             mimeType = 'audio/wav';
//         } else if (header.includes('664c6143')) { // fLaC
//             mimeType = 'audio/flac';
//         }
//     } catch (e) {
//         console.warn("Could not detect audio format, using default wav:", e);
//     }
    
//     console.log("Detected MIME type:", mimeType);
//     return `data:${mimeType};base64,${base64Data}`;
// }

// function updateTranscription(transcript_data) {
//     const transcriptionDiv = document.getElementById('transcripts');
//     transcriptionDiv.innerHTML += transcript_data + '<br/>';
    
//     // Auto-scroll to bottom
//     transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
//     console.log(transcript_data)
// }

// function updateAudioStatus(status) {
//     const statusDiv = document.getElementById('audioStatus');
//     if (statusDiv) {
//         statusDiv.textContent = status;
//         console.log("Audio Status:", status);
//     }
// }

// function updateAudioQueueStatus() {
//     const queueDiv = document.getElementById('audioQueue');
//     if (queueDiv) {
//         queueDiv.textContent = `Queue: ${audioQueue.length} items`;
//     }
// }

// function startRecording() {
//     if (isRecording) return;
//     isRecording = true;
//     startTime = Date.now(); 
    
//     // Clear audio queue when starting new recording
//     stopAllAudio();
//     updateAudioStatus("Recording started");
    
//     const AudioContext = window.AudioContext || window.webkitAudioContext;
//     context = new AudioContext();
//     navigator.mediaDevices.getUserMedia({ audio: true, sampleRate: 16000 }).then(stream => {
//         globalStream = stream;
//         const input = context.createMediaStreamSource(stream);
//         processor = context.createScriptProcessor(bufferSize, 1, 1);
//         processor.onaudioprocess = e => processAudio(e);
//         input.connect(processor);
//         processor.connect(context.destination);
//     }).catch(error => console.error('Error accessing microphone', error));
//     document.getElementById('startButton').disabled = true;
//     document.getElementById('endButton').disabled = false;
// }

// function stopAllAudio() {
//     // Clear audio queue
//     audioQueue = [];
//     updateAudioQueueStatus();
    
//     // Stop current audio
//     if (currentAudio) {
//         currentAudio.pause();
//         currentAudio = null;
//     }
    
//     isPlayingAudio = false;
//     updateAudioStatus("Audio stopped");
//     console.log("All audio stopped and queue cleared");
// }

// function linearToMuLaw(sample) {
//     const MU = 255;
//     const MAX = 32768;
//     const sign = (sample >> 8) & 0x80;
//     if (sign) sample = -sample;
//     if (sample > MAX) sample = MAX;

//     const magnitude = Math.log1p((MU * sample) / MAX) / Math.log1p(MU);
//     const muLawSample = (~(sign | (magnitude * 127)) & 0xFF);

//     return muLawSample;
// }

// function stopRecording() {
//     if (!isRecording) return;
//     isRecording = false;

//     if (globalStream) {
//         globalStream.getTracks().forEach(track => track.stop());
//     }
//     if (processor) {
//         processor.disconnect();
//         processor = null;
//     }
//     if (context) {
//         context.close().then(() => context = null);
//     }
//     document.getElementById('startButton').disabled = false;
//     document.getElementById('endButton').disabled = true;
//     updateMetrics(); // Display duration and process time
//     updateAudioStatus("Recording stopped");
// }

// function downsampleBuffer(buffer, inputSampleRate, outputSampleRate) {
//     if (inputSampleRate === outputSampleRate) {
//         return buffer;
//     }
//     const sampleRateRatio = inputSampleRate / outputSampleRate;
//     const newLength = Math.round(buffer.length / sampleRateRatio);
//     const result = new Float32Array(newLength);
//     let offsetResult = 0;
//     let offsetBuffer = 0;
//     while (offsetResult < result.length) {
//         const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
//         let accum = 0, count = 0;
//         for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
//             accum += buffer[i];
//             count++;
//         }
//         result[offsetResult] = accum / count;
//         offsetResult++;
//         offsetBuffer = nextOffsetBuffer;
//     }
//     return result;
// }

// function processAudio(e) {
//     const inputSampleRate = context.sampleRate;
//     const outputSampleRate = actualSampleRate;

//     const left = e.inputBuffer.getChannelData(0);
//     const downsampledBuffer = downsampleBuffer(left, inputSampleRate, outputSampleRate);
//     const audioData = convertFloat32ToInt16(downsampledBuffer);
//     // convertToMuLaw(downsampledBuffer);
    
//     if (websocket && websocket.readyState === WebSocket.OPEN) {
//         websocket.send(audioData);
//     }
// }

// function convertFloat32ToInt16(buffer) {
//     const buf = new Int16Array(buffer.length);
//     for (let i = 0; i < buffer.length; i++) {
//         buf[i] = Math.max(-1, Math.min(1, buffer[i])) * 0x7FFF;
//     }
//     return buf.buffer;
// }

// function convertToMuLaw(buffer) {
//     const muLawBuffer = new Uint8Array(buffer.length);
//     for (let i = 0; i < buffer.length; i++) {
//         const sample = Math.max(-1, Math.min(1, buffer[i]));
//         const int16Sample = sample < 0 ? sample * 32768 : sample * 32767;
//         muLawBuffer[i] = linearToMuLaw(int16Sample);
//     }
//     return muLawBuffer.buffer;
// }

// function updateMetrics() {
//     // Calculate and display duration and process time
//     const durationSpan = document.getElementById('duration');
//     const duration = Math.floor((Date.now() - startTime) / 1000);
    
//     durationSpan.textContent = `${duration}s`;
//     // Display the metrics
//     document.getElementById('metrics').style.display = 'block';
// }

// async function clearCache() {
//     try {
//         const response = await fetch('/clearcache', {
//             method: 'GET',
//         });

//         if (response.ok) {
//             console.log('Cache cleared successfully');
//             alert('Cache cleared successfully');
//         } else {
//             alert('Failed to clear cache');
//         }
//     } catch (error) {
//         console.error('Error:', error);
//         alert('Error clearing cache');
//     }
// }

// // Manual controls for debugging
// function clearAudioQueue() {
//     stopAllAudio();
//     alert('Audio queue cleared');
// }

// function testAudio() {
//     // Test with a simple beep
//     const audioContext = new (window.AudioContext || window.webkitAudioContext)();
//     const oscillator = audioContext.createOscillator();
//     const gainNode = audioContext.createGain();
    
//     oscillator.connect(gainNode);
//     gainNode.connect(audioContext.destination);
    
//     oscillator.frequency.value = 800;
//     oscillator.type = 'sine';
//     gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
//     gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
//     oscillator.start(audioContext.currentTime);
//     oscillator.stop(audioContext.currentTime + 0.5);
    
//     updateAudioStatus("Test beep played");
// }


// // async function handleLanguageChange(languageCode) {
// //     try {
// //         // Update UI to show loading state        
// //         // Send API request to backend
// //         const response = await fetch('/api/set-language', {
// //             method: 'POST',
// //             headers: {
// //                 'Content-Type': 'application/json',
// //             },
// //             body: JSON.stringify({
// //                 language: languageCode,
// //                 client_id: document.getElementById('clientId').textContent
// //             })
// //         });

// //         if (!response.ok) {
// //             throw new Error(`HTTP error! status: ${response.status}`);
// //         }

// //         const result = await response.json();
        
// //         if (result.success) {
// //             currentLLMLanguage = languageCode;
// //             updateLanguageUI(languageCode, 'success');
            
// //             // Add confirmation message to transcript
// //             addSystemMessage(`Language changed to ${AVAILABLE_LANGUAGES[languageCode].name} ${AVAILABLE_LANGUAGES[languageCode].flag}`);
            
// //             console.log(`Language successfully changed to: ${languageCode}`);
// //         } else {
// //             throw new Error(result.error || 'Failed to change language');
// //         }

// //     } catch (error) {
// //         console.error('Error changing language:', error);
// //         updateLanguageUI(currentLLMLanguage, 'error');
        
// //         // Show error message to user
// //         addSystemMessage(`❌ Failed to change language: ${error.message}`, 'error');
        
// //         // Reset language selector to current language
// //         document.getElementById('llmLanguageSelect').value = currentLLMLanguage;
// //     }
// // }
// window.onload = initWebSocket;

let websocket;
let context;
let processor;
let globalStream;
let startTime;
let endTime;
const client_id = Date.now();
const actualSampleRate = 16000;
const baseUrl = window.location.href.replace('index.jsp', '').replace('http', 'ws');
const modifedurl = 'ws://' + baseUrl.split("/")[2] + '/';
const websocket_uri = `${modifedurl}ws/${client_id}?sample_rate=${actualSampleRate}&format=pcm`;
const bufferSize = 16384;
let isRecording = false;

let audioQueue = [];
let isPlayingAudio = false;
let currentAudio = null;

function initWebSocket() {
    const websocketAddress = websocket_uri;
    if (!websocketAddress) {
        console.log("WebSocket address is required.");
        return;
    }
    websocket = new WebSocket(websocketAddress);
    websocket.binaryType = "arraybuffer"; // Important for raw Float32

    websocket.onopen = () => {
        console.log("WebSocket connection established");
    };

    websocket.onclose = event => {
        console.log("WebSocket connection closed", event);
    };

    websocket.onmessage = event => {
        console.log("Message from server:", typeof event.data, event.data);
        try {
            const data = JSON.parse(event.data);
            if (data.type === "audio_chunk") {
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
    };
}

function handleAudioChunk(base64AudioData) {
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
        updateAudioQueueStatus();
        return;
    }

    isPlayingAudio = true;
    const base64AudioData = audioQueue.shift();
    updateAudioQueueStatus();
    updateAudioStatus("Playing audio...");

    try {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }

        const audioDataUrl = createAudioDataUrl(base64AudioData);
        currentAudio = new Audio();
        currentAudio.src = audioDataUrl;

        currentAudio.onended = () => {
            currentAudio = null;
            updateAudioStatus("Audio finished");
            playNextAudio();
        };

        currentAudio.onerror = (error) => {
            console.error("Audio playback error:", error);
            updateAudioStatus("Audio error");
            currentAudio = null;
            setTimeout(() => playNextAudio(), 10);
        };

        await currentAudio.play();
    } catch (error) {
        console.error("Error playing audio:", error);
        updateAudioStatus("Playback error: " + error.message);
        currentAudio = null;
        setTimeout(() => playNextAudio(), 100);
    }
}

function createAudioDataUrl(base64Data) {
    let mimeType = 'audio/wav';
    try {
        const binaryString = atob(base64Data.substring(0, 100));
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const header = Array.from(bytes.slice(0, 12)).map(b => b.toString(16).padStart(2, '0')).join('');
        if (header.includes('fff3') || header.includes('fff2')) mimeType = 'audio/mp3';
        else if (header.includes('4f676753')) mimeType = 'audio/ogg';
        else if (header.includes('52494646')) mimeType = 'audio/wav';
        else if (header.includes('664c6143')) mimeType = 'audio/flac';
    } catch (e) {
        console.warn("Could not detect audio format, using default wav:", e);
    }
    return `data:${mimeType};base64,${base64Data}`;
}

function updateTranscription(transcript_data) {
    const transcriptionDiv = document.getElementById('transcripts');
    transcriptionDiv.innerHTML += transcript_data + '<br/>';
    transcriptionDiv.scrollTop = transcriptionDiv.scrollHeight;
}

function updateAudioStatus(status) {
    const statusDiv = document.getElementById('audioStatus');
    if (statusDiv) {
        statusDiv.textContent = status;
    }
}

function updateAudioQueueStatus() {
    const queueDiv = document.getElementById('audioQueue');
    if (queueDiv) {
        queueDiv.textContent = `Queue: ${audioQueue.length} items`;
    }
}

function startRecording() {
    if (isRecording) return;
    isRecording = true;
    startTime = Date.now();

    stopAllAudio();
    updateAudioStatus("Recording started");

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    context = new AudioContext();
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        globalStream = stream;
        const input = context.createMediaStreamSource(stream);
        processor = context.createScriptProcessor(bufferSize, 1, 1);
        processor.onaudioprocess = e => processAudio(e);
        input.connect(processor);
        processor.connect(context.destination);
    }).catch(error => console.error('Error accessing microphone', error));

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
    if (context) {
        context.close().then(() => context = null);
    }
    document.getElementById('startButton').disabled = false;
    document.getElementById('endButton').disabled = true;
    updateMetrics();
    updateAudioStatus("Recording stopped");
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

function processAudio(e) {
    const inputSampleRate = context.sampleRate;
    const outputSampleRate = actualSampleRate;

    const left = e.inputBuffer.getChannelData(0);
    const downsampledBuffer = downsampleBuffer(left, inputSampleRate, outputSampleRate);

    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(downsampledBuffer.buffer);  // ✅ Send Float32 PCM directly
    }
}

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

function updateMetrics() {
    const durationSpan = document.getElementById('duration');
    const duration = Math.floor((Date.now() - startTime) / 1000);
    durationSpan.textContent = `${duration}s`;
    document.getElementById('metrics').style.display = 'block';
}

async function clearCache() {
    try {
        const response = await fetch('/clearcache', { method: 'GET' });
        if (response.ok) alert('Cache cleared successfully');
        else alert('Failed to clear cache');
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

window.onload = initWebSocket;
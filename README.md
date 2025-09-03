# Voice2Voice: Real-Time Conversational AI

A sophisticated, real-time voice-to-voice conversational AI pipeline built with FastAPI. This project integrates multiple state-of-the-art ASR, LLM, and TTS services into a seamless, low-latency voice bot experience. Its modular design allows for easy swapping of components to suit different needs.

## 🚀 Key Features

  * **End-to-End Voice Conversation**: Real-time audio streaming, processing, and response generation.
  * **Modular Architecture**: Pluggable services for ASR, LLM, and TTS using a factory pattern, making it easy to extend.
  * **Multi-Provider Support**:
      * **ASR**: Support for `Nemo`, `Whisper` (local), and `Groq Whisper` (API).
      * **LLM**: Integrates with `Azure GPT-4o/mini`, `Groq Llama3`, `Mixtral`, and on-prem TGI models via LangChain.
      * **TTS**: Generates speech using `Parler-TTS` and `Sarvam AI`.
  * **Streaming I/O**: Streams both the LLM text responses and the TTS audio chunks back to the client for minimal perceived latency.
  * **Voice Activity Detection (VAD)**: Intelligently detects pauses in speech to determine when a user has finished speaking.
  * **Web-Based Interface**: A clean and simple frontend built with HTML, Bootstrap, and JavaScript to demonstrate the pipeline's capabilities.

## 🛠️ Tech Stack

  * **Backend**: Python, FastAPI
  * **Real-time Communication**: WebSockets
  * **AI/ML Libraries**: LangChain, PyTorch, Nemo, Faster-Whisper, Transformers, Groq, Sarvam AI
  * **Frontend**: HTML, CSS (Bootstrap), JavaScript

## ⚙️ Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

  * Python 3.8+ and `pip`
  * An NVIDIA GPU with CUDA is highly recommended for running the local ASR and TTS models efficiently.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirement.txt
    ```

### Configuration

1.  **Environment Variables**:
    You'll need to set up API keys for the services you intend to use. You can set these as system environment variables. The code looks for the following keys:

      * `GROQ_API_KEY`
      * `AZURE_OPENAI_ENDPOINT`
      * `AZURE_OPENAI_API_KEY`
      * `SARVAM_API_KEY`

2.  **Service Selection**:
    Edit the `app/resources/constants.yaml` file to select the ASR, LLM, and TTS models you want to use. You also need to verify that the paths to any local models are correct.

    ```yaml
    # app/resources/constants.yaml

    # ... other configurations

    # Choose your ASR model
    asr_type: whisper # Options: whisper, nemo, groq

    # Choose your chatbot (LLM) model
    chatbot_type: gpt4omini # Options: gpt4omini, gpt4o, llama, onprem

    # Choose your TTS model
    tts_type: parler # Options: parler, sarvam

    # ... other configurations
    ```

## ▶️ How to Run

1.  **Start the FastAPI server:**
    Use Uvicorn to run the application. From the root directory of the project:

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

2.  **Access the web interface:**
    Open your web browser and navigate to `http://localhost:8000`.

3.  **Start a conversation:**

      * Enter a session ID and task details and click "Connect".
      * On the next page, click "Start Recording" and begin speaking.
      * You will see your speech transcribed in real-time, and the AI's audio response will play automatically.

      * Click "Stop Recording" to end the session.

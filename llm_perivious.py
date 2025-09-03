# llm.py
from groq import Groq

def get_llm_response(text: str, system_prompt: str, groq_client: Groq, st_session_state_status_setter: callable, st_session_state_pipeline_error_message_setter: callable):
    """
    Gets a response from the LLM based on the input text and system prompt.

    Args:
        text: The user's transcribed text.
        system_prompt: The system prompt for the LLM.
        groq_client: An initialized Groq API client.
        st_session_state_status_setter: Function to update thinking status in Streamlit.
        st_session_state_pipeline_error_message_setter: Function to set pipeline error messages.


    Returns:
        The LLM's response string, or an error/skip message.
    """
    if not groq_client:
        st_session_state_pipeline_error_message_setter("LLM error: Groq client not available.")
        return "Error: Groq client not available for LLM."
    if not text or "error" in text.lower() or "no speech detected" in text.lower() or "transcription: (" in text.lower():
        st_session_state_status_setter({"msg": "LLM: Skipped due to transcription issue.", "level": "info"})
        return "LLM: Skipped due to prior error or no input."

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model="llama3-8b-8192", # or other suitable model
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM error details: {e}") # For server-side logging
        st_session_state_pipeline_error_message_setter(f"LLM error: {str(e)}")
        return f"LLM error: {str(e)}"
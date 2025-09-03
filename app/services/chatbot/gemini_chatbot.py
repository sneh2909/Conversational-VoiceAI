import os
import google.generativeai as genai
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from app.services.chatbot.chatbot_interface import Chatbot
from app.utilities import sken_logger
from app.utilities.env_util import EnvironmentVariableRetriever

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"realtime_transcription": "nemo"})

class GeminiChatbot(Chatbot):
    """
    Google Gemini chatbot implementation using google-generativeai SDK
    
    Author: AI Assistant
    
    Classes:
        GeminiChatbot: A class to interface with Google's Gemini AI model.
        
        Methods:
            __init__(self): Initializes the GeminiChatbot class with configuration.
            get_llm(self): Returns the language model instance.
            set_temperature(self, temp): Sets the temperature for generation.
            set_max_tokens(self, max_tokens): Sets the maximum tokens for generation.
            set_top_p(self, top_p): Sets the top_p parameter for generation.
            response(self, query, memory, system_prompt): Generates a response using Gemini.
    """
    
    def __init__(self) -> None:
        """Initialize the Gemini chatbot with configuration"""
        self.model_name = "gemini-2.5-flash-lite"
        self.api_key = EnvironmentVariableRetriever.get_env_variable("GOOGLE_AI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_AI_API_KEY environment variable not set")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.provider = 'GoogleGemini'
        
        # Default system prompt
        self.system = "You are a helpful AI assistant powered by Google Gemini."
        
        # Default generation parameters
        self.temperature = 0.7
        self.max_tokens = 1000
        self.top_p = 0.9
    
    def set_temperature(self, temp: float):
        """Set the temperature for generation"""
        self.temperature = temp
    
    def set_max_tokens(self, max_tokens: int):
        """Set the maximum tokens for generation"""
        self.max_tokens = max_tokens
    
    def set_top_p(self, top_p: float):
        """Set the top_p parameter for generation"""
        self.top_p = top_p
    
    def get_llm(self):
        """Return the language model instance"""
        return self.model
    
    def response(self, query, memory: ConversationBufferWindowMemory, system_prompt=None):
        """
        Generate a response using Gemini
        
        Args:
            query (str): The user's query
            memory (ConversationBufferWindowMemory): Conversation memory
            system_prompt (str, optional): Custom system prompt
            
        Returns:
            str: Generated response from Gemini
        """
        try:
            # Get previous conversation from memory
            previous_conversation = memory.buffer_as_messages
            
            # Use provided system prompt or default
            system_prompt = system_prompt if system_prompt is not None else self.system
            
            # Build context using LangChain message objects
            context = [SystemMessage(content=system_prompt)]
            
            # Add conversation history
            if previous_conversation:
                for message in previous_conversation:
                    if isinstance(message, HumanMessage):
                        context.append(HumanMessage(content=message.content))
                    elif isinstance(message, AIMessage):
                        context.append(AIMessage(content=message.content))
            
            # Add current query
            context.append(HumanMessage(content=query))
            
            # Convert LangChain messages to Gemini format
            gemini_messages = []
            for msg in context:
                if isinstance(msg, SystemMessage):
                    gemini_messages.append({"role": "user", "parts": [msg.content]})
                elif isinstance(msg, HumanMessage):
                    gemini_messages.append({"role": "user", "parts": [msg.content]})
                elif isinstance(msg, AIMessage):
                    gemini_messages.append({"role": "model", "parts": [msg.content]})
            
            # Generate response using Gemini
            response = self.model.generate_content(
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.top_p
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error in Gemini response generation: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

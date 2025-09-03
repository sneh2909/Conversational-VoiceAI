import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import  SystemMessagePromptTemplate
from langchain.prompts.chat import AIMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from app.services.chatbot.chatbot_interface import Chatbot
from app.utilities import sken_logger
from app.utilities.constants import Constants
from groq import Groq


logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})

class MixtralChatCompletion(Chatbot):
    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key)
        self.system = "You are a helpful assistant. Answer the user's questions based on the conversation history. If you don't know the answer, say 'I don't know'."
        
    async def response(self,query,memory: ConversationBufferWindowMemory, system_prompt=None):
        previous_conversation = memory.buffer_as_messages
        system_prompt=self.system if system_prompt is None else system_prompt
        context = [{"role": "system","content":self.system}]
        if previous_conversation:
            for message in previous_conversation:
                if isinstance(message,HumanMessage):
                    context.append({"role": "user","content": message.content})
                elif isinstance(message,AIMessage):
                    context.append({"role": "assistant", "content": message.content})
                else:
                    pass
        context.append({"role":"user","content": query})
        chat_completion = await self.groq_client.chat.completions.create(messages=context,
                                                                    model=self.model,
                                                                    temperature=0.2,
                                                                    max_tokens=100,
                                                                    top_p=1,
                                                                    stop=None,
                                                                    stream=False,)
        logger.info("#############"*10)
        logger.info(context)
        response = chat_completion.choices[0].message.content
        logger.info(response)
        return {'human':query,'AI':response} 

    # llm.py

    async def get_llm_response(self,text: str, system_prompt: str, query,memory: ConversationBufferWindowMemory):
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
        if not self.groq_client:
            logger.error("LLM error: Groq client not available.")
            return "Error: Groq client not available for LLM."

        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                model="llama3-8b-8192", # or other suitable model
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            return f"LLM error: {str(e)}"
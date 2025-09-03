from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from app.services.chatbot.chatbot_interface import Chatbot

class OnPremLLM(Chatbot):
    def __init__(self) -> None:
        self.llm_client = ChatOpenAI(
            model="sarvamai/sarvam-1",
            base_url="http://172.28.1.24:8082/v1",
            api_key="-",  # Replace if needed
            temperature=0.5,
            max_tokens=200,
            streaming=True
        )
        self.system = "You are a helpful assistant. Answer the user's questions based on the conversation history. If you don't know the answer, say 'I don't know'.keep it short and simple"


    def response(self, query: str, memory: ConversationBufferWindowMemory, system_prompt=None):
        previous_conversation = memory.buffer_as_messages
        system_prompt=self.system if system_prompt is None else system_prompt
        # Build context using LangChain message objects
        context = [SystemMessage(content=system_prompt)]
        
        if previous_conversation:
            for message in previous_conversation:
                if isinstance(message, HumanMessage):
                    context.append(HumanMessage(content=message.content))
                elif isinstance(message, AIMessage):
                    context.append(AIMessage(content=message.content))
                # Skip other message types
        
        context.append(HumanMessage(content=query))
        chat_completion=self.llm_client.stream(context)
        return chat_completion

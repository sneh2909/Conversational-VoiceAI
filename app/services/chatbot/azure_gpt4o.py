import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
from app.services.chatbot.chatbot_interface import Chatbot
from app.utilities import sken_logger
from app.utilities.constants import Constants
from groq import Groq
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.chatbot.chatbot_interface import Chatbot
from app.utilities import sken_logger
from app.utilities.env_util import EnvironmentVariableRetriever
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json


logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})


class AzureOpenAIGPT4o(Chatbot):
    """
    _summary_

    Author: Sneh

    Classes:
        AzureOpenAIGPT4o: A class to interface with the Azure GPT-4 OpenAI model.

        Methods:
            __init__(self, config): Initializes the AzureOpenAIGPT4o class with the given configuration.
                Args:
                    config (dict): Configuration parameters for initializing the Azure GPT-4 OpenAI model.

            get_llm(self): Returns the language model instance.
                Returns:
                    AzureChatOpenAI: An instance of the AzureChatOpenAI class.

            generate(self, prompt, text) -> str: Generates a response from the language model based on the given prompt and text.
                Args:
                    prompt (str): The prompt to guide the language model.
                    text (str): The input text for the language model to process.

                Returns:
                    str: The generated response from the language model.
    """
    def __init__(self) -> None:
        self.model_name = "azure-gpt-4o"
        self.deployment_name = "ds-gpt-4-o"
        self.model = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            azure_endpoint=EnvironmentVariableRetriever.get_env_variable("AZURE_OPENAI_ENDPOINT"),
            api_key=EnvironmentVariableRetriever.get_env_variable("AZURE_OPENAI_API_KEY"),
            openai_api_version="2023-05-15",
            openai_api_type="azure"
        )
        self.provider = 'AzureOpenAI'

    def set_temperature(self,temp: float):
            self.model.temperature = temp
        
    def set_max_tokens(self,max_tokens:int):
        self.model.max_tokens = max_tokens
        
    def set_top_p(self,top_p:float):
        self.model.model_kwargs = {"top_p": top_p}
    
    def get_llm(self):
        return self.model    

    # def create_agent_with_tools(self, tools:list, snippets_list:list=None, llm_logger:LLMLogger=None):
    #     self.token_tracker = TokenTrackerCallback()
    #     history = json.dumps(snippets_list)
    #     llm_logger.original['input_tokens'] += LLMLogger.get_token_size(history)
        # self.agent = create_react_agent(model=self.model, tools=tools, state_modifier=history, debug=True)
        

    # async def generate_template(self, inputs:dict):
    #     response = await self.agent.ainvoke(inputs, config={"callbacks": [self.token_tracker]}, debug=True)
    #     return response['messages'][-1].content, self.token_tracker.get_total_tokens()
    
    def response(self,query,memory: ConversationBufferWindowMemory, system_prompt=None):
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
        
        # Use LangChain's streaming method
        chat_completion = self.model.stream(context)
        return chat_completion
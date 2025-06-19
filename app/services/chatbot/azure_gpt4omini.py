import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.system import SystemMessage
import requests
from app.services.chatbot.chatbot_interface import Chatbot
from app.utilities import sken_logger
from app.utilities.constants import Constants
from groq import Groq
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.chatbot.chatbot_interface import Chatbot
from app.utilities import sken_logger
from app.utilities.env_util import EnvironmentVariableRetriever
from langchain_core.tools import tool
from langchain_community.retrievers.azure_ai_search import AzureAISearchRetriever
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
import json

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})

class AzureOpenAIGPT4oMini(Chatbot):
    """
    _summary_

    Author: Maheshbabu

    Classes:
        AzureOpenAIGPT4oMini: A class to interface with the Azure GPT-4o Mini OpenAI model.

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
        self.model_name = "azure-gpt-4o-mini"
        self.deployment_name = "gpt-4o-mini"
        self.model = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            azure_endpoint=EnvironmentVariableRetriever.get_env_variable("AZURE_OPENAI_ENDPOINT"),
            api_key=EnvironmentVariableRetriever.get_env_variable("AZURE_OPENAI_API_KEY"),
            openai_api_version="2023-05-15",
            openai_api_type="azure",
            temperature=0.3,
            max_tokens=100, 
            streaming=True
        )
        self.provider = 'AzureOpenAI'
        self.system = "You are a helpful assistant. Answer the user's questions based on the conversation history. If you don't know the answer, say 'I don't know'.keep it short and simple"
        # Initialize tools
        # self.search = DuckDuckGoSearchRun()
        self.memory_saver = MemorySaver()
        
        # Create agent with tools
        self.tools = [
            self.filter_relevant_messages,
            self.get_relevant_docs,
            # self.search_duckduckgo
        ]
        
        self.agent = create_react_agent(
            model=self.model, 
            tools=self.tools,
            checkpointer=self.memory_saver
        )

        
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
    #     self.agent = create_react_agent(model=self.model, tools=tools, state_modifier=history, debug=True)
        

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
    
    @tool 
    async def filter_relevant_messages(self, state_modifier: list[dict], keywords_from_userinput: list[str]):
        """
        Filter and retrieve relevant messages from the agent's memory based on keywords.
        
        Args:
            keywords_from_userinput: Keywords extracted from user input
            state_modifier: List of conversation snippets with keywords
        
        Returns:
            list[dict]: Filtered relevant snippets
        """
        try:
            filtered_snippets = []
            logger.info("Started using filter_relevant_messages tool....")
            
            for item in state_modifier:
                snippet_keywords = item.get("Keywords", [])
                if any(keyword.lower() in map(str.lower, snippet_keywords) for keyword in keywords_from_userinput):
                    filtered_snippets.append(item)
            
            return filtered_snippets
            
        except Exception as exe:
            logger.error(f"Error while using filter relevant messages tool: {exe}", exc_info=True)
            return state_modifier

    @tool
    async def get_relevant_docs(self, query: str, org_id: str, product_ids: list[int]):
        """
        Retrieve relevant documents from vector database.
        
        Args:
            query: User query for retrieving product information
            org_id: Organization ID
            product_ids: List of product IDs
        
        Returns:
            list: Relevant information matching the query
        """
        try:
            logger.info("Started using get_relevant_docs tool....")
            relevant_contents = []

            if org_id and product_ids:
                for product_id in product_ids:
                    query_data = {
                        "query": query,
                        "org_id": org_id,
                        "product_id": int(product_id),
                        "threshold": Constants.fetch_constant('knowledge_retrieval_details')['threshold'],
                        "top_k": Constants.fetch_constant('knowledge_retrieval_details')['top_k']
                    }

                    query_payload = json.dumps(query_data)
                    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
                    response = requests.post(
                        url=Constants.fetch_constant('knowledge_retrieval_details')['retriever_url'], 
                        data=query_payload, 
                        headers=headers
                    )
                    results = json.loads(response.text)
                    
                    if len(results) >= 1:
                        for result in results:
                            relevant_contents.append(result['text'])

                    return relevant_contents
            else:
                logger.info("No docs found")
                return []
                
        except Exception as exe:            
            logger.error(f"Error while using get relevant docs tool: {exe}")
            raise exe

    @tool
    async def search_duckduckgo(self, query: str):
        """
        Search the web using DuckDuckGo.
        
        Args:
            query: Search query string
        
        Returns:
            str: Summary of search results
        """
        try:
            logger.info("Started using duckduckgo tool....")
            results = await self.search.arun(query, verbose=True)
            return results
        except Exception as exe:
            logger.error(f"Error while using search duckduckgo tool: {exe}")
            raise exe

    def response_with_tools(
        self, 
        query: str, 
        memory: ConversationBufferWindowMemory, 
        system_prompt: str = None,
        org_id: str = None,
        product_ids: list = None
    ):
        """
        Generate streaming response with tool usage.
        
        Args:
            query: User query
            memory: Conversation memory
            system_prompt: Custom system prompt
            org_id: Organization ID for retrieval
            product_ids: Product IDs for retrieval
            snippets_list: Historical conversation snippets
            thread_id: Thread ID for agent memory
        
        Yields:
            str: Streaming response chunks
        """
        system_prompt = self.system if system_prompt is None else system_prompt
        
        # Prepare agent input
        agent_input = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }
        
        # Add context from memory
        previous_conversation = memory.buffer_as_messages
        if previous_conversation:
            conversation_messages = []
            for message in previous_conversation:
                if isinstance(message, HumanMessage):
                    conversation_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    conversation_messages.append({"role": "assistant", "content": message.content})
            
            # Insert conversation history before the current query
            agent_input["messages"] = (
                [{"role": "system", "content": system_prompt}] + 
                conversation_messages + 
                [{"role": "user", "content": query}]
            )
        
        # Add additional context for tools
        if org_id and product_ids:
            tool_context = f"Organization ID: {org_id}, Product IDs: {product_ids}"
            agent_input["messages"].insert(1, {"role": "system", "content": tool_context})
        
        # Configure agent
        config = {
            "stream_mode": "messages"
        }
        
        try:
            # Stream response from agent
            for chunk in self.agent.stream(agent_input, config=config):
                if "messages" in chunk:
                    for message in chunk["messages"]:
                        if hasattr(message, 'content') and message.content:
                            # Handle different message types
                            if hasattr(message, 'type'):
                                if message.type == 'ai':
                                    yield message.content
                                elif message.type == 'tool':
                                    # Optionally yield tool usage information
                                    yield f"[Using tool: {message.name}]"
                            else:
                                yield message.content
                                
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Error: {str(e)}"
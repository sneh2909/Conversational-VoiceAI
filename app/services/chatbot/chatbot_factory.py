from app.services.chatbot.mixtral import MixtralChatCompletion
from app.services.chatbot.llama import LLamaChatCompletion
from app.services.chatbot.azure_gpt4omini import AzureOpenAIGPT4oMini
from app.services.chatbot.azure_gpt4o import AzureOpenAIGPT4o
from app.services.chatbot.huggingface_tgi import OnPremLLM

class ChatbotFactory:
    """
    Factory method to initialise a chatbot.
    """
    
    @staticmethod
    def create_chatbot(type):
        if type == "mixtral":
            return MixtralChatCompletion()
        elif type == "llama":
            return LLamaChatCompletion()
        elif type == "gpt4omini":
            return AzureOpenAIGPT4oMini()
        elif type == "gpt4o":
            return AzureOpenAIGPT4o()
        elif type=="onprem":
            return OnPremLLM()
        else:
            raise ValueError(f"Chat bot type = {type} is not implemented")
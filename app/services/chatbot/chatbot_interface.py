class Chatbot:
    """
    An interface class for the devlopment of chatbot using LLM
    """
    async def response(self,query,memory):
        raise NotImplementedError("This method should be implemented by the sub class")
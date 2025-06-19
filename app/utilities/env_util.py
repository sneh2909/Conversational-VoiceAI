import os
from app.utilities.constants import Constants
from app.utilities.singletons_factory import SkenSingleton
from app.utilities import sken_logger

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__), {"env_variable_retriever": "env"})

class EnvironmentVariableRetriever(metaclass=SkenSingleton):
    
    @classmethod
    def get_env_variable(cls, variable_name):
        """
        Retrieves the value of an environment variable.
        
        Args:
            variable_name (str): The name of the environment variable.
        
        Returns:
            str: The value of the environment variable, or None if not found.
        """
        try:
            # First, check if the variable exists in os.environ
            if variable_name in os.environ:
                value = os.environ.get(variable_name)
                logger.info(f"Successfully retrieved environment variable: {variable_name}")
                return value
            else:
                logger.warning(f"Environment variable not found: {variable_name}")
                return None
        except Exception as exe:
            logger.error(f"Error retrieving environment variable {variable_name}: {str(exe)}")
            raise
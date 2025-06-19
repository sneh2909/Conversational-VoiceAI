from logging import getLogger, INFO, Formatter, LoggerAdapter, StreamHandler, FileHandler
from app.utilities.constants import Constants
import os
import sys
import logging
logging.basicConfig(level = logging.INFO)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)

def get_logger(name, level=INFO, file_name = Constants.fetch_constant("log_config")["filepath"]):

    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    handler = StreamHandler(sys.stdout)
    log_format = " %(levelname)s : %(asctime)-5s %(filename)s:%(lineno)d %(funcName)-5s --> %(message)s"
    formatter = Formatter(log_format)
    handler.setFormatter(formatter)
    filehandler = FileHandler(file_name)
    filehandler.setFormatter(formatter)
    logger = getLogger(name)
    logger.addHandler(handler)
    logger.addHandler(filehandler)
    logger.setLevel(level)
    return logger

class LoggerAdap(LoggerAdapter):
    def process(self,msg,kwargs):
        return '%s' % (msg), kwargs
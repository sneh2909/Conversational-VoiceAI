import asyncio
from app.utilities import sken_logger
from tritonclient.http.aio import InferenceServerClient, InferInput
from app.utilities.constants import Constants
from transformers import AutoTokenizer
from typing import List
import time
import numpy as np
import torch
from app.utilities.singletons_factory import SkenSingleton
from IndicTransToolkit.processor import IndicProcessor

logger = sken_logger.LoggerAdap(sken_logger.get_logger(__name__),{"realtime_transcription":"nemo"})

class IndicTrans(metaclass = SkenSingleton):
    def __init__(self) -> None:
        if not hasattr(self, 'client'):
            self.model_name = Constants.fetch_constant("triton_server")["indictrans"]
            self.triton_server_url = Constants.fetch_constant("triton_server")["url"]
            self.ssl_required = Constants.fetch_constant("triton_server")["ssl_required"]
            self.concurrency = Constants.fetch_constant("triton_server")["concurrency"]
            self.client = InferenceServerClient(
                url=self.triton_server_url, 
                verbose=False, 
                conn_timeout=None,
                ssl=self.ssl_required
            )
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained(Constants.fetch_constant("tokenizers")["indictrans2-indic-en-dist-200M"], trust_remote_code=True)
        self.indic_processor = IndicProcessor(inference=True)
    
    async def get_translations(self, sentences: List[str]):
        if sentences:
            try:
                # Tokenize client-side
                tokenized_inputs = await asyncio.to_thread(self.preprocess_sentence, sentences)
                translation = await self.send_triton_request(tokenized_inputs)
                return translation[0] if translation else ""
            except Exception as err:
                logger.error(f"Error During Translation || Error={err}", exc_info=True)
                raise err

    def preprocess_sentence(self, sentences):
        batch_sentences = self.indic_processor.preprocess_batch(sentences, src_lang="hin_Deva", tgt_lang="eng_Latn")
        tokenized_inputs = self.tokenizer(batch_sentences, return_tensors="np", padding=True, truncation=True, return_attention_mask=True)
        return tokenized_inputs

    async def send_triton_request(self, tokenized_inputs):

        # Prepare Triton inputs

        triton_inputs = [
            InferInput('INPUT_IDS', tokenized_inputs['input_ids'].shape, "INT32"),
            InferInput('attention_mask', tokenized_inputs['attention_mask'].shape, "INT32")
        ]

        triton_inputs[0].set_data_from_numpy(tokenized_inputs['input_ids'].astype(np.int32))
        triton_inputs[1].set_data_from_numpy(tokenized_inputs['attention_mask'].astype(np.int32))

        # Perform inference
        start = time.time()
        logger.info("Sending request to Triton server")
        results = await self.client.infer(self.model_name, triton_inputs)
        outputs = results.as_numpy('OUTPUT_IDS')
        end = time.time()
        logger.info(f"Received response from Triton server || Time Taken={end-start}")
        with self.tokenizer.as_target_tokenizer():
            translations =self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        return translations

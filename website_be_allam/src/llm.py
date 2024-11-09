from langchain_core.prompt_values import ChatPromptValue
# from langchain_ibm import WatsonxLLM
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
import random
import os
import requests

import os

from langchain_community.llms import WatsonxLLM as _WatsonxLLM
from langchain_ibm import WatsonxEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult
from typing import List, Optional, Any
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes


from dotenv import load_dotenv
load_dotenv()

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 512,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 50,
    "top_p": 1,
}

class LLM():
    is_chatbot = False
    def invoke(self, value: ChatPromptValue):
        pass

class IBMLLM(LLM):
    def __init__(self, model_id="meta-llama/llama-3-70b-instruct", params=parameters):
        self.llm = _WatsonxLLM(
            model_id=model_id,
            url="https://eu-de.ml.cloud.ibm.com",
            project_id="4162cbfc-92d2-4fe9-b3e7-c26dac539d6c",
            params = params
        )
    def invoke(self, value):
        return self.llm.invoke(value)

class HFLLM(LLM):
    def __init__(self, model_id="meta-llama/llama-3-70b-instruct", params=parameters):
        self.llm = InferenceClient(
        'meta-llama/Llama-3.1-70B-Instruct',
        token='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',
    )
    
    def invoke(self, value):
        result = ""
        for message in self.llm.chat_completion(
            messages=[{"role": "user", "content": value}],
            seed=random.randint(1, 100000),
            max_tokens=512,
            stream=True,
        ): result += message.choices[0].delta.content
        
        return result

class HFLLM2(LLM):
    def __init__(self, model_id="meta-llama/Meta-Llama-3-70B-Instruct"):
        self.llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            huggingfacehub_api_token=os.environ['HF_TOKEN']
        )
    
    def invoke(self, value):
        return self.llm.invoke(value)
    
class OPENAILLM(LLM):
    def __init__(self, endpoint="https://api.openai.com/v1/chat/completions", model_id="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model = model_id,
            base_url=endpoint
        )
    def invoke(self, value):
        return self.llm.invoke(value)  

class Chatbot(LLM):
    is_chatbot = True
    def __init__(self,endpoint):
        self.endpoint = endpoint
    def invoke(self, value: ChatPromptValue):
        obj = {
        "message" : str(value),
        "history" : []
        }
        response = requests.post(self.endpoint, json=obj)
        return response.json()['content']

class CustomWatsonxLLM(_WatsonxLLM):
    temperature: float = 0.05
    """
    A workaround for interface incompatibility: Ragas expected all LLMs to
    have a `temperature` property whereas WatsonxLLM does not define it.
    """

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        A workaround for interface incompatibility: Ragas expected the
        `token_usage` property of the LLM result be of a particular shape.
        WatsonX returns it in a slightly different shape.
        """
        result: LLMResult = super()._generate(prompts, stop, run_manager, stream, **kwargs)
        if not result.llm_output or "token_usage" not in result.llm_output:
            return result
        usage = result.llm_output["token_usage"]
        if not isinstance(usage, dict):
            return result
        result.llm_output["token_usage"] = {
            "prompt_tokens": usage["input_token_count"],
            "completion_tokens": usage["generated_token_count"],
            "total_tokens": usage["input_token_count"] + usage["generated_token_count"],
        }
        return result
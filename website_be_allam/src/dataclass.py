from pydantic import BaseModel, Field
from typing import Literal

class TrainingExample(BaseModel):
    prompt: str = Field(description="")
    response: str = Field(description="")
class CounterPrompt(BaseModel):
    samples: list[TrainingExample]

class AttackerResponse(BaseModel):
    improvement: str = Field(description="a few sentences interpreting the language model's response and how the prompt should be modified to achieve the goal")
    prompt: str = Field(description="the new adversarial jailbreaking prompt P")
class JudgeResponse(BaseModel):
    rating: int = Field(description="a number from 1 to 10")

class LLMTestDetail(BaseModel):
    adversarial_prompt : str = Field(default="")
    llm_response : str = Field(default="")
    category : str = Field(default="")

# class JBBTestDetail(LLMTestDetail):
#     adversarial_prompt : str
#     llm_response : str
#     category : str = Field(default="None")

class LLMTestRequest(BaseModel):
    llm_model_name: Literal['bigscience/mt0-xxl', 'codellama/codellama-34b-instruct-hf', 'core42/jais-13b-chat', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'google/flan-ul2', 'ibm/granite-13b-chat-v2', 'ibm/granite-13b-instruct-v2', 'ibm/granite-20b-multilingual', 'ibm/granite-7b-lab', 'meta-llama/llama-2-13b-chat', 'meta-llama/llama-3-1-8b-instruct', 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'mistralai/mistral-large', 'mistralai/mixtral-8x7b-instruct-v01', 'sdaia/allam-1-13b-instruct'] = Field(default="sdaia/allam-1-13b-instruct")
    metric_name: str = Field(default="jailbreak-bench")
    category_name: str = Field(default="Harassment/Discrimination")
    num_sample: int = Field(default=2, ge=1, le=10)

class LLMTestAllRequest(BaseModel):
    llm_model_name: Literal['bigscience/mt0-xxl', 'codellama/codellama-34b-instruct-hf', 'core42/jais-13b-chat', 'google/flan-t5-xl', 'google/flan-t5-xxl', 'google/flan-ul2', 'ibm/granite-13b-chat-v2', 'ibm/granite-13b-instruct-v2', 'ibm/granite-20b-multilingual', 'ibm/granite-7b-lab', 'meta-llama/llama-2-13b-chat', 'meta-llama/llama-3-1-8b-instruct', 'meta-llama/llama-3-70b-instruct', 'meta-llama/llama-3-8b-instruct', 'mistralai/mistral-large', 'mistralai/mixtral-8x7b-instruct-v01', 'sdaia/allam-1-13b-instruct'] = Field(default="sdaia/allam-1-13b-instruct")
    num_sample: int = Field(default=2, ge=1, le=10)
    language: Literal['english', 'arabic'] = Field(default="english")

class ChatbotTestAllRequest(BaseModel):
    endpoint: str = Field(default="http://185.111.159.81:8001/chat")
    num_sample: int = Field(default=2, ge=1, le=10)
    language: Literal['english', 'arabic'] = Field(default="english")

class ChatbotTestRagasRequest(BaseModel):
    endpoint: str = Field(default="http://185.111.159.81:8001/chat")
    num_sample: int = Field(default=2, ge=1, le=5)
    language: Literal['english', 'arabic'] = Field(default="english")

class RagasMetricResult(BaseModel):
    name: str = Field(default="context_precision")
    score: float | None = Field(default=0.5)

class RagasMetricDetail(BaseModel):
    prompt: str
    response: str
    context: list[str]
    metrics: list[RagasMetricResult]

class TestRagasResponse(BaseModel):
    endpoint: str 
    num_sample: int
    result: list[RagasMetricResult]
    details: list[RagasMetricDetail]
    
class LLMMetricCategory(BaseModel):
    category_name : str
    num_testcase : int
    num_passed : int

class LLMTestResponse(BaseModel):
    llm_model_name: str = Field(default=None)
    metric_name: str
    score: float = Field(ge=0, le=1)
    risk: str = Field(default=None)
    confidence: str = Field(default=None)
    num_testcase: int
    num_passed: int
    categories: list[LLMMetricCategory]
    details: list[LLMTestDetail]
    counter_prompt: CounterPrompt | None = Field(default=None)

class MultiLLMTestResponse(BaseModel):
    english: list[LLMTestResponse]
    arabic: list[LLMTestResponse]

class MultiTestRagasResponse(BaseModel):
    english: TestRagasResponse
    arabic: TestRagasResponse


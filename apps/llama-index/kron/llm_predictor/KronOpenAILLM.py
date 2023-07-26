from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

from llama_index.bridge.langchain import BaseLanguageModel, BaseChatModel
from llama_index.llms.langchain import LangChainLLM
from llama_index.llms.openai import OpenAI

from llama_index.llms.base import (
    LLM,
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)

from kron.llm_predictor.openai_utils import kron_openai_modelname_to_contextsize

class KronOpenAI(OpenAI):

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=kron_openai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens or -1,
            is_chat_model=self._is_chat_model,
        )
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        #print("KronOpenAI complete called")
        response = super().complete(prompt, **kwargs)
        text = response.text
        text = text.strip()   #triples might not start at the begining of the line
        #useful triplets are before <|endoftext|>
        text = text.split("<|endoftext|>")[0]
        response.text = text
        return response
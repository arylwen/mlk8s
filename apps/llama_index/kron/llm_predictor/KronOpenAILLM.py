from llama_index.bridge.langchain import BaseLanguageModel, BaseChatModel
from llama_index.llms.langchain import LangChainLLM
from llama_index.bridge.langchain import OpenAI, ChatOpenAI

from llama_index.llms.base import LLMMetadata

from kron.llm_predictor.openai_utils import kron_openai_modelname_to_contextsize

class KronOpenAI(OpenAI):

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=kron_openai_modelname_to_contextsize(self.model),
            num_output=self.max_tokens or -1,
            is_chat_model=self._is_chat_model,
        )
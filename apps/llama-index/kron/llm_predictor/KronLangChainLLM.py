from llama_index.bridge.langchain import BaseLanguageModel, BaseChatModel
from llama_index.llms.langchain import LangChainLLM
from llama_index.bridge.langchain import OpenAI, ChatOpenAI

from llama_index.llms.base import LLMMetadata

from kron.llm_predictor.openai_utils import kron_openai_modelname_to_contextsize

def is_chat_model(llm: BaseLanguageModel) -> bool:
    return isinstance(llm, BaseChatModel)

class KronLangChainLLM(LangChainLLM):
    """Adapter for a LangChain LLM."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        super().__init__(llm)


    @property
    def metadata(self) -> LLMMetadata:
        is_chat_model_ = is_chat_model(self.llm)
        if isinstance(self.llm, OpenAI):
            return LLMMetadata(
                context_window=kron_openai_modelname_to_contextsize(self.llm.model_name),
                num_output=self.llm.max_tokens,
                is_chat_model=is_chat_model_ ,
            )
        elif isinstance(self.llm, ChatOpenAI):
            return LLMMetadata(
                context_window=kron_openai_modelname_to_contextsize(self.llm.model_name),
                num_output=self.llm.max_tokens or -1,
                is_chat_model=is_chat_model_ ,
            )
        else:
            return super().metadata()

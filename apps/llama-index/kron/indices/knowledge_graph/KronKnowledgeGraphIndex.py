import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index import KnowledgeGraphIndex 
from llama_index.data_structs.data_structs import KG
from llama_index.indices.service_context import ServiceContext
from llama_index.prompts.prompts import KnowledgeGraphPrompt
from llama_index.storage.storage_context import StorageContext
from llama_index.schema import BaseNode

class KronKnowledgeGraphIndex(KnowledgeGraphIndex):
    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[KG] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        kg_triple_extract_template: Optional[KnowledgeGraphPrompt] = None,
        max_triplets_per_chunk: int = 10,
        include_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
          super().__init__(
                           nodes, 
                           index_struct, 
                           service_context, 
                           storage_context, 
                           kg_triple_extract_template, 
                           max_triplets_per_chunk, 
                           include_embeddings, 
                           kwargs
                        )
          
    def _extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract keywords from text."""
        #response, _ = self._service_context.llm_predictor.predict(
        response = self._service_context.llm_predictor.predict(
            self.kg_triple_extract_template,
            text=text,
        )
        return self._kron_parse_triplet_response(response)

    @staticmethod
    def _kron_parse_triplet_response(response: str) -> List[Tuple[str, str, str]]:
        print("_kron_parse_triplet_response")
        knowledge_strs = response.strip().split("\n")
        results = []
        for text in knowledge_strs:
            text = text.strip()   #triples might not start at the begining of the line
            #text = text.replace('<|endoftext|>', '')
            #useful triplets are before <|endoftext|>
            text = text.split("<|endoftext|>")[0]
            if text == "" or text[0] != "(":
                # skip empty lines and non-triplets
                continue
            tokens = text[1:-1].split(",")
            if len(tokens) != 3:
                continue
            subj, pred, obj = tokens
            results.append((subj.strip(), pred.strip(), obj.strip()))
        return results
# llama index
This is a thin extension to llama_index to support a custom OpenAI compatible API, without modifying the llama_index code.

## what
I wanted to make sense of documents and discover useful information without reading 187 arXiv articles. For example, I wanted to know about benchmarks for Question Answering(QA), not Quality Assurance (QA)
|<div style="width:220px"> Writer/camel-5b-hf inference </div> | Writer/camel-5b-hf guided retrieval  |
|---|---|
|What benchmarks can we use for QA? | What benchmarks can we use for QA? |
| 1. Unit testing: Testing individual components or functions of a system. <br>2. Integration testing: Testing how different components interact with each other.<br>3. System testing: Testing the entire application or system as a whole.<br>4. Acceptance testing: Evaluating whether the system meets requirements and client expectations.<br>5. Performance testing: Testing the system's responsiveness, stability, and performance under load.<br>6. Security testing: Testing for vulnerabilities and ensuring data integrity and confidentiality.<br>7. User testing: Testing the system's usability and user experience.<br>8. System testing with real users: Testing the system with real users to gather user feedback.<br> | Response 1: Answer: We can use the following benchmarks for QA: 1) SQuAD 2) TACQA 3) ChEBI 4) BioASQ 5) IJCAI-LIMS 6) TREC-6 7) TREC-7 8) TREC-8 9) TREC-9 10) TREC-10 11) TREC-13 12) TREC-13 <br>\---------------------<br>Response 2: Answer: We can use benchmarks such as SQuAD, TACQ, and ChEBI to evaluate QA systems.<br>\---------------------<br>Response 3: Answer: We can use the following benchmarks for QA: SQuAD, TACQA, and ChEBI.<br>\---------------------<br>Response 4: Answer: We can use the BioRED test set for QA.<br>\---------------------<br>Response 5: Answer: We can use F-score, precision, recall, and Matthew's correlation coefficient (MRR) as benchmarks for QA.<br>\---------------------<br>Response 6: Answer: We can use the following benchmarks for QA: TREC, SQuAD, and TAC. <br><br>| 

## how
Kron llama_api extension configures a custom openai.api_base and adds Writer/camel-5b-hf and mosaicml/mpt-7b-instruct to the supported model list for llama_index. It uses [llama_api](/apps/llama-api/), an OpenAPI compatible adapter on top of [RAY](/config/ray/).

![system architecture](/docs/diagrams/llama-compact.drawio.png "knowledge graph system architecture")

## let's get technical
Under the hood, there are a few k8s deployments working together.

![kron k8s](/docs/diagrams/llama.drawio.svg)
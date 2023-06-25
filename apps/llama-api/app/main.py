from fastapi import FastAPI
from pydantic import BaseModel

import asyncio

import argparse
import asyncio
import json
import logging

import os
from typing import Generator, Optional, Union, Dict, List, Any

import fastapi
from fastapi import Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
#import httpx
from pydantic import BaseSettings
import shortuuid
#import tiktoken
import uvicorn

from app.protocol.openai_api_protocol import (
#    ChatCompletionRequest,
#    ChatCompletionResponse,
#    ChatCompletionResponseStreamChoice,
#    ChatCompletionStreamResponse,
#    ChatMessage,
#    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
#    DeltaMessage,
#    CompletionResponseStreamChoice,
#    CompletionStreamResponse,
#    EmbeddingsRequest,
#    EmbeddingsResponse,
#    ErrorResponse,
#    ModelCard,
#    ModelList,
#    ModelPermission,
    UsageInfo,
)

import ray
import ray.data
import pandas as pd

import os
import time
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
        CORSMiddleware,
        allow_origins="*",
        allow_credentials=False,
        allow_methods="*",
        allow_headers="*",
    )

load_dotenv()

RAY_CLIENT_URL = os.getenv('RAY_CLIENT_URL')
print(f'Using RAY at: {RAY_CLIENT_URL}')

#python versions must match on client and server: 3.9.15
ray.init(
    address=RAY_CLIENT_URL,
    namespace="kuberay",
    runtime_env={
        "pip": [
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "numpy<1.24",  
            "einops==0.6.1",
            "torch",            
        ],
        "env_vars": {
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    },
)

print(f'RAY at: {RAY_CLIENT_URL} initialized.')

ray.data.context.DatasetContext.get_current().use_streaming_executor = False

@ray.remote(num_cpus=10)
class PredictCallable:

    def __init__(self, model_id: str, revision: str = None):
        print('__init__')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
#           revision=revision,
#            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
#           device_map="auto",  # automatically makes use of all GPUs available to the Actor
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.tie_weights()
        end = time.time()
        print(f'model loaded successfully in: {end-start}s')

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        print('__call__')        

        tokenized = self.tokenizer(
            list(batch["prompt"]), return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device)

        gen_tokens = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.9,
            max_length=100,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return pd.DataFrame(
            self.tokenizer.batch_decode(gen_tokens), columns=["responses"]
        )

'''
    Does not support int values or list of lists
    TODO - raise exceptn
'''
def process_input(model_name, input):
    if isinstance(input, str):
        input = [input]
#    elif isinstance(input, list):
#        if isinstance(input[0], int):
#            decoding = tiktoken.model.encoding_for_model(model_name)
#            input = [decoding.decode(input)]
#        elif isinstance(input[0], list):
#            decoding = tiktoken.model.encoding_for_model(model_name)
#            input = [decoding.decode(text) for text in input]

    return input

@app.get('/')
async def index():
    return {"Message": "This is Index"}

#batch inference endpoint
@app.post('/v1/completions')
async def predict(request: CompletionRequest):

    #request.prompt becomes a list
    request.prompt = process_input(request.model, request.prompt)
    print(request.prompt)

    actor = PredictCallable.remote(model_id=request.model, 
                                   revision = "float16") 
    
    future = actor.__call__.remote(pd.DataFrame(request.prompt, columns=["prompt"]))
    start = time.time()
    gen = ray.get(future)
    end = time.time()
    print(f'inference time: {end-start}s')


    prediction = gen.iloc[0]

    choices = []
    choices.append(
                CompletionResponseChoice(
                    index=0,
                    text=prediction[0],
                    logprobs=0,
                    finish_reason="stop",
                )
    )

    return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo()
        )


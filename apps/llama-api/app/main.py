from fastapi import FastAPI
from pydantic import BaseModel
import json

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

from app.persistence.core import init_database
from app.persistence.core import LlmInferenceRecord
from app.persistence.data_access import LlmInferencePersistence
#from app.persistence.core import SQLALCHEMY_DATABASE_URI

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

DB_USER = os.getenv("DB_USER","postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD","")
DB_HOST = os.getenv("DB_HOST","0.0.0.0")
DB_NAME = os.getenv("DB_NAME","llama-api")
SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:"
        f"{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    )

RAY_CLIENT_URL = os.getenv('RAY_CLIENT_URL')
print(f'Using RAY at: {RAY_CLIENT_URL}')

#TODO add thread safety
__IDLE_ACTOR_DICT__ = {}
__BUSY_ACTOR_DICT__ = {}
__GPU_MODEL_LIST__ = ["tiiuae/falcon-7b-instruct", "Writer/camel-5b-hf", "mosaicml/mpt-7b-instruct", "mosaicml/mpt-30b-instruct"]

def ray_init():
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
                "importlib",           
            ],
            "env_vars": {
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            }
        },
    )

    ray.data.context.DatasetContext.get_current().use_streaming_executor = False

    '''
    actor cache: used to reduce the inference time by the load model time
        when a model is requested, add a list of actors

    '''
    __IDLE_ACTOR_DICT__ = {}
    __BUSY_ACTOR_DICT__ = {}

    print(f'RAY at: {RAY_CLIENT_URL} initialized.')

ray_init()

print(f'Using pgsql at: {SQLALCHEMY_DATABASE_URI}')
dbsession = init_database(SQLALCHEMY_DATABASE_URI)
print(f'dbsession at: {SQLALCHEMY_DATABASE_URI} started.')

'''
    max_restarts=1 - restart the actor if it dies unexpectedly 
    TODO build a LRU scheme where actors are removed if not used for a long period of time
'''
@ray.remote(num_cpus=20, max_restarts=1)
class PredictCallable:

    def __init__(self, model_id: str, revision: str = 'main'):
        print('__init__')
        import torch
        import importlib
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        # Set constants
        MODULE = importlib.import_module("transformers") # dynamic import of module class, AutoModel not good enough for text generation
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # support gpu inference if possible

        self.model_name = model_id
        #try to fit the model on GPU
        self.device = DEVICE

        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True) # load config for model
        print(config)
        if config.architectures:
            model_classname = config.architectures[0]
            try:
                model_class = getattr(MODULE, model_classname) # get model class from config
                self.model = model_class.from_pretrained(
                    model_id, 
                    revision=revision,
                    config=config, 
#                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ) # dynamically load right model class for text generation
            except Exception as e:
                print(f"****************{e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
#                    torch_dtype=config.torch_dtype, #torch.bfloat16,
#                    low_cpu_mem_usage=True,
                    device_map="auto",  # automatically makes use of all GPUs available to the Actor
                    trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
#                low_cpu_mem_usage=True,
                device_map="auto",  # automatically makes use of all GPUs available to the Actor
                trust_remote_code=True,
            )

        self.model.tie_weights()
        end = time.time()
        print(f'model loaded successfully in: {end-start}s')

    '''
        echo: True = returns the prompt with the completion
        echo: False = returns only the completion
    '''
    def __call__(self, batch: pd.DataFrame, echo=False, max_tokens=512) -> pd.DataFrame:
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
            max_length=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if echo:
            result =  self.tokenizer.batch_decode(gen_tokens)
        else:
            result = self.tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])

        return pd.DataFrame(
             result , columns=["responses"]
        )

@ray.remote(num_cpus=5, num_gpus=1, max_restarts=1)
class PredictCallableGPU:

    def __init__(self, model_id: str, revision: str = 'main'):
        print('__init__')
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch
        import importlib

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        # Set constants
        MODULE = importlib.import_module("transformers") # dynamic import of module class, AutoModel not good enough for text generation
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # support gpu inference if possible

        self.model_name = model_id
        #try to fit the model on GPU
        self.device = DEVICE

        start = time.time()
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True) # load config for model
        print(config)
        if config.architectures:
            model_classname = config.architectures[0]
            try:
                model_class = getattr(MODULE, model_classname) # get model class from config
                self.model = model_class.from_pretrained(
                    model_id, 
                    revision=revision,                    
                    config=config, 
                    trust_remote_code=True,
                    device_map="auto", 
                ) # dynamically load right model class for text generation
            except Exception as e:
                print(f"*****{e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    revision=revision,
                    torch_dtype=config.torch_dtype, #torch.bfloat16,
                    device_map="auto",  # automatically makes use of all GPUs available to the Actor
                    trust_remote_code=True,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                low_cpu_mem_usage=True,
                device_map="auto",  # automatically makes use of all GPUs available to the Actor
                trust_remote_code=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.tie_weights()
        end = time.time()

        print(f'model loaded successfully in: {end-start}s')

    '''
        echo: True = returns the prompt with the completion
        echo: False = returns only the completion
    '''
    def __call__(self, batch: pd.DataFrame, echo=False, max_tokens=512) -> pd.DataFrame:
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
            max_length=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if echo:
            result =  self.tokenizer.batch_decode(gen_tokens)
        else:
            result = self.tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])

        return pd.DataFrame(
             result , columns=["responses"]
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
    #TODO extract methos and put the database log in a try catch
    try: 
        start = time.time()
        response = await dispatch_request_to_model(request)
        end = time.time()
        persistence = LlmInferencePersistence(db_session=dbsession, user_id=request.user)
        persistence.save_predictions(
            model_id=request.model,
            request=json.dumps(str(request)),
            response=json.dumps(str(response)),
            inference_time=end-start,
        )
        return response
    except ray.exceptions.RayActorError as rae:
        # reinit RAY and retry; most probably stale connection
        ray_init()
        start = time.time()
        response = await dispatch_request_to_model(request)
        end = time.time()
        persistence = LlmInferencePersistence(db_session=dbsession, user_id=request.user)
        persistence.save_predictions(
            model_id=request.model,
            request=json.dumps(request),
            response=json.dumps(response),
            inference_time=end-start,
        )        
        return response
    except ConnectionError as ce:
        # reinit RAY and retry; most probably stale connection
        ray_init()
        start = time.time()
        response = await dispatch_request_to_model(request)
        end = time.time()
        persistence = LlmInferencePersistence(db_session=dbsession, user_id=request.user)
        persistence.save_predictions(
            model_id=request.model,
            request=json.dumps(request),
            response=json.dumps(response),
            inference_time=end-start,
        )        
        return response
    
'''
    temporary way to dispatch model to GPU; 
    TODO: calculate model size and decide
'''
async def dispatch_request_to_model(request: CompletionRequest):
    if request.model in __GPU_MODEL_LIST__:
        response = await predict_gpu(request)
    else:
        response = await predict_cpu(request)
    return response

async def predict_cpu(request: CompletionRequest):

    #request.prompt becomes a list
    request.prompt = process_input(request.model, request.prompt)
    #print(request.prompt)
    print(request)

    #use an actor from the model queue if available, otherwise create the actor (load the model)
    if __IDLE_ACTOR_DICT__.get(request.model):
        actor = __IDLE_ACTOR_DICT__[request.model].pop()
    else:
        try:
            actor = PredictCallable.remote(model_id=request.model) 
            __IDLE_ACTOR_DICT__[request.model]=[]
        except Exception as e:
            print(f"****************Exception occured in __init__: {e}")
        
    try:
        future = actor.__call__.remote(pd.DataFrame(request.prompt, 
                                                    columns=["prompt"]), 
                                                    echo=request.echo, 
                                                    max_tokens=request.max_tokens)
  
        if __BUSY_ACTOR_DICT__.get(request.model):
            __BUSY_ACTOR_DICT__[request.model][future] = actor
        else:
            __BUSY_ACTOR_DICT__[request.model]={}
            __BUSY_ACTOR_DICT__[request.model][future] = actor
    except Exception as e:
            print(f"****************Exception occured in __call__: {e}")


    start = time.time()
    gen = ray.get(future, timeout=3600)
    end = time.time()
    print(f'inference time: {end-start}s')
    print(gen.iloc[0])

    prediction = gen.iloc[0]

    choices = []
    choices.append(
                CompletionResponseChoice(
                    index=0,
                    text=prediction[0],
    #                logprobs={"tokens": [], 'top_logprobs':[]},
                    finish_reason="stop",
                )
    )

    #move actor from the busy queue to the idle queue
    actor = __BUSY_ACTOR_DICT__[request.model].pop(future)
    __IDLE_ACTOR_DICT__[request.model].append(actor)

    return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo()
        )

async def predict_gpu(request: CompletionRequest):

    #request.prompt becomes a list
    request.prompt = process_input(request.model, request.prompt)
    #print(request.prompt)
    print(request)

    #use an actor from the model queue if available, otherwise create the actor (load the model)
    if __IDLE_ACTOR_DICT__.get(request.model):
        actor = __IDLE_ACTOR_DICT__[request.model].pop()
    else:
        try:
            #must kill any other actors on the GPU to switch models
            models = __IDLE_ACTOR_DICT__.keys()
            for m in models:
                #only kill the actors on the GPU, don't affect CPU jobs
                if request.model in __GPU_MODEL_LIST__:
                    __IDLE_ACTOR_DICT__[m] = []
                    __BUSY_ACTOR_DICT__[m]  = []
            actor = PredictCallableGPU.remote(model_id=request.model) 
            __IDLE_ACTOR_DICT__[request.model]=[]
        except Exception as e:
#            print(f"****************Exception occured in __init__: {e}")
            template = "*******************An exception of type {0} occurred in __init__. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            #this will bubble up and reinit ray
            raise e
        
    try:
        future = actor.__call__.remote(pd.DataFrame(request.prompt, 
                                                    columns=["prompt"]), 
                                                    echo=request.echo, 
                                                    max_tokens=request.max_tokens)
 
        if __BUSY_ACTOR_DICT__.get(request.model):
            __BUSY_ACTOR_DICT__[request.model][future] = actor
        else:
            __BUSY_ACTOR_DICT__[request.model]={}
            __BUSY_ACTOR_DICT__[request.model][future] = actor
    except Exception as e:
 #           print(f"****************Exception occured in __call__: {e}")
            template = "*******************An exception of type {0} occurred in __call__. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            raise e

    start = time.time()
    gen = ray.get(future, timeout=1800)
    end = time.time()
    print(f'inference time: {end-start}s')
    print(gen.iloc[0])

    prediction = gen.iloc[0]

    choices = []
    choices.append(
                CompletionResponseChoice(
                    index=0,
                    text=prediction[0],
    #                logprobs={"tokens": [], 'top_logprobs':[]},
                    finish_reason="stop",
                )
    )

    #move actor from the busy queue to the idle queue
    actor = __BUSY_ACTOR_DICT__[request.model].pop(future)
    __IDLE_ACTOR_DICT__[request.model].append(actor)

    return CompletionResponse(
            model=request.model, choices=choices, usage=UsageInfo()
        )


# Python's standard library imports
import os
import sys
import json
import time
import logging
import grpc

# FastAPI related imports
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

# Pydantic related imports
#from pydantic import BaseModel, BaseSettings

# Typing related imports
from typing import Generator, Optional, Union, Dict, List, Any

# External libraries imports
import ray
import ray.data
import pandas as pd
import numpy as np
from requests.exceptions import ConnectionError

# Modules from your application
from app.persistence.core import init_database, LlmInferenceRecord
from app.persistence.data_access import LlmInferencePersistence

# Models and protocols related imports
from app.protocol.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DeltaMessage,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
    UsageInfo,
)

# External libraries related to your application
import shortuuid
import tiktoken
import uvicorn
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
if(RAY_CLIENT_URL is None):
    sys.exit('RAY_CLIENT_URL environement variable is not set. Exiting.')

print(f'Using RAY at: {RAY_CLIENT_URL}')

#TODO add thread safety
__IDLE_ACTOR_DICT__ = {}
__BUSY_ACTOR_DICT__ = {}
__GPU_MODEL_LIST__ = ["tiiuae/falcon-7b-instruct", 
                      "Writer/camel-5b-hf", 
                      "Writer/InstructPalmyra-20b", 
                      "mosaicml/mpt-7b-instruct", 
                      "mosaicml/mpt-30b-instruct",
                      "Arylwen/instruct-palmyra-20b-gptq-8",
                      "Arylwen/instruct-palmyra-20b-gptq-4",                      
                      "Arylwen/instruct-palmyra-20b-gptq-2",
                      "deepseek-ai/deepseek-coder-6.7b-instruct",
                      ]

def ray_init():
    #python versions must match on client and server: 3.9
    ray.init(
        address=RAY_CLIENT_URL,
        namespace="kuberay",
        runtime_env={
            "pip": [
                "torch==2.0.1", 
                "accelerate>=0.16.0",
                "auto-gptq",  
                "transformers>=4.34.0",
                #"git+https://github.com/huggingface/transformers.git",
                #"git+https://github.com/huggingface/optimum.git",      
                #"transformers",
                "optimum",          
                "numpy<1.24",  
                "einops==0.6.1",
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
    def __call__(self, 
                 batch: pd.DataFrame, 
                 echo=False, 
                 max_tokens=512,
                 temperature=0.9) -> pd.DataFrame:
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
            temperature=temperature,
            max_length=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        #token accounting
        prompt_tokens = [len(x) for x in input_ids]
        total_tokens = [len(x) for x in gen_tokens]
        completion_tokens = [x-y for (x,y) in zip(total_tokens, prompt_tokens)]            

        if echo:
            result =  self.tokenizer.batch_decode(gen_tokens)
        else:
            result = self.tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])

        return pd.DataFrame(
            np.transpose([result, prompt_tokens, completion_tokens, total_tokens]) , 
                        columns=["responses", 'prompt_tokens', "completion_tokens", "total_tokens"] 
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
    def __call__(self, batch: pd.DataFrame, 
                 echo=False, 
                 max_tokens=512, 
                 temperature=0.9) -> pd.DataFrame:
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
            temperature=temperature,
            max_length=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        #token accounting
        prompt_tokens = [len(x) for x in input_ids]
        total_tokens = [len(x) for x in gen_tokens]
        completion_tokens = [x-y for (x,y) in zip(total_tokens, prompt_tokens)]            

        if echo:
            result =  self.tokenizer.batch_decode(gen_tokens)
        else:
            result = self.tokenizer.batch_decode(gen_tokens[:, input_ids.shape[1]:])

        return pd.DataFrame(
             np.transpose([result, prompt_tokens, completion_tokens, total_tokens]) , 
                        columns=["responses", 'prompt_tokens', "completion_tokens", "total_tokens"]
        )
    


'''
    Does not support int values or list of lists
    TODO - raise exception
'''
def process_input(model_name, input):
    if isinstance(input, str):
        input = [input]
    elif isinstance(input, list):
        if isinstance(input[0], int):
            decoding = tiktoken.model.encoding_for_model(model_name)
            input = [decoding.decode(input)]
        elif isinstance(input[0], list):
            decoding = tiktoken.model.encoding_for_model(model_name)
            input = [decoding.decode(text) for text in input]

    return input

@app.get('/')
async def index():
    return {"Message": "This is Index"}

def toJSON(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

#batch inference endpoint
@app.post('/v1/completions')
async def predict(request: CompletionRequest):
    #TODO extract method and put the database log in a try catch
    try: 
        start = time.time()
        response = await dispatch_request_to_model(request)
        end = time.time()
        persistence = LlmInferencePersistence(db_session=dbsession, user_id=request.user)
        persistence.save_predictions(
            model_id=request.model,
            request=toJSON(request),
            response=toJSON(response),
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
            request=toJSON(request),
            response=toJSON(response),
            inference_time=end-start, 
        )        
        return response
    #todo 
    except (ConnectionError, grpc.RpcError) as ce:
        # reinit RAY and retry; most probably stale connection
        ray_init()
        start = time.time()
        response = await dispatch_request_to_model(request)
        end = time.time()
        persistence = LlmInferencePersistence(db_session=dbsession, user_id=request.user)
        persistence.save_predictions(
            model_id=request.model,
            request=toJSON(request),
            response=toJSON(response),
            inference_time=end-start,
        )        
        return response
    except Exception as unkex:
        template = "*******************An exception of type {0} occurred in __call__. Arguments:\n{1!r}"
        message = template.format(type(unkex).__name__, unkex.args)
        print(message)
        raise unkex

    
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
            template = "*******************An exception of type {0} occurred in __call__. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            raise e
        
    try:
        future = actor.__call__.remote(pd.DataFrame(request.prompt, 
                                                    columns=["prompt"]), 
                                                    temperature = request.temperature,
                                                    echo=request.echo, 
                                                    max_tokens=request.max_tokens)
  
        if __BUSY_ACTOR_DICT__.get(request.model):
            __BUSY_ACTOR_DICT__[request.model][future] = actor
        else:
            __BUSY_ACTOR_DICT__[request.model]={}
            __BUSY_ACTOR_DICT__[request.model][future] = actor
    except Exception as e:
        template = "*******************An exception of type {0} occurred in __call__. Arguments:\n{1!r}"
        message = template.format(type(e).__name__, e.args)
        print(message)
        raise e


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
            template = "*******************An exception of type {0} occurred in __init__. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            #this will bubble up and reinit ray
            raise e
        
    try:
        future = actor.__call__.remote(pd.DataFrame(request.prompt, 
                                                    columns=["prompt"]), 
                                                    temperature = request.temperature,
                                                    echo=request.echo, 
                                                    max_tokens=request.max_tokens)
 
        if __BUSY_ACTOR_DICT__.get(request.model):
            __BUSY_ACTOR_DICT__[request.model][future] = actor
        else:
            __BUSY_ACTOR_DICT__[request.model]={}
            __BUSY_ACTOR_DICT__[request.model][future] = actor
    except Exception as e:
            template = "*******************An exception of type {0} occurred in __call__. Arguments:\n{1!r}"
            message = template.format(type(e).__name__, e.args)
            print(message)
            raise e

    start = time.time()
    gen = ray.get(future, timeout=1800)
    end = time.time()
    print(f'inference time: {end-start}s')
    print(gen.iloc[0])

    #TODO - support a return list
    prediction = gen.iloc[0]

    choices = []
    choices.append(
                CompletionResponseChoice(
                    index=0,
                    text=prediction["responses"],
    #                logprobs={"tokens": [], 'top_logprobs':[]},
                    finish_reason="stop",
                )
    )

    usageInfo = UsageInfo()
    usageInfo.completion_tokens = prediction["completion_tokens"]
    usageInfo.prompt_tokens = prediction["prompt_tokens"]
    usageInfo.total_tokens = prediction["total_tokens"]

    #move actor from the busy queue to the idle queue
    actor = __BUSY_ACTOR_DICT__[request.model].pop(future)
    __IDLE_ACTOR_DICT__[request.model].append(actor)

    return CompletionResponse(
            model=request.model, choices=choices, usage=usageInfo
        )


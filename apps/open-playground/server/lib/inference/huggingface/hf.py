import os
import gc
import transformers
import psutil
import torch
import importlib
import logging
import warnings

from transformers import AutoTokenizer, AutoConfig, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
from .generator import greedy_search_generator
from .helpers import StoppingCriteriaSub

# monkey patch for transformers
transformers.generation.utils.GenerationMixin.greedy_search = greedy_search_generator
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Set constants
MODULE = importlib.import_module("transformers") # dynamic import of module class, AutoModel not good enough for text generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # support gpu inference if possible

logger = logging.getLogger(__name__)

class HFInference:
    '''
    Class for huggingface local inference
    '''
    def __init__(self, model_name: str):
        #cleanup models not in use
        gc.collect()
        torch.cuda.empty_cache()
        self.model_name = model_name
        #try to fit the model on GPU
        self.device = DEVICE
        self.model, self.tokenizer = self.load_model(model_name)

    def __del__(self):
        logger.info(f'Destructor called for:{self.model_name}')
        if self.model:
            self.model.cpu()
            del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Helper function to load model from transformers library
    def load_model(self, model_name: str) -> (PreTrainedModel, PreTrainedTokenizer):
        '''
        Load model from transformers library
        dynamically instantiates the right model class for text generation from model config architecture
        '''
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name) # load config for model
        logger.info(config)
        if config.architectures:
            model_classname = config.architectures[0]
            model_class = getattr(MODULE, model_classname) # get model class from config
            model = model_class.from_pretrained(model_name, config=config) # dynamically load right model class for text generation
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto' if DEVICE == 'cuda' else None, torch_dtype=torch.float16)

        param_size = sum(
            param.nelement() * param.element_size() for param in model.parameters()
        )
        buffer_size = sum(
            buffer.nelement() * buffer.element_size() for buffer in model.buffers()
        )
        size_all_mb = (param_size + buffer_size) / 1024**2
        logger.info('model size: {:.3f}MB'.format(size_all_mb))

        if DEVICE == 'cuda':
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
        else:
            device_memory = psutil.virtual_memory().total / 1024**2

        logger.info('device memory: {:.3f}MB'.format(device_memory))

        #20% padding
        model_fraction = 0.79
        if size_all_mb > device_memory * model_fraction: #some padding
            if self.device == 'cuda':
               #try to run on cpu if not enough memory on the GPU
               self.device = 'cpu'
               device_memory = psutil.virtual_memory().total / 1024**2 
               logger.info('Model size is too large for GPU to run inference on. Running inference on cpu: device memory: {:.3f}MB'.format(device_memory))
               if size_all_mb > device_memory * model_fraction: #some padding
                   raise Exception('Model size is too large for host to run inference on')
            else:
                raise Exception('Model size is too large for host to run inference on')

        model.to(self.device) # gpu inference if possible
        return model, tokenizer

    def generate(self, 
            prompt: str, 
            max_length: int, 
            temperature: float, 
            top_k: int, 
            top_p: float, 
            repetition_penalty: float, 
            stop_sequences: list = None,
            **kwargs
        ):
        '''
        Generate text from prompt, using monkey patched transformers.generation.utils.GenerationMixin.greedy_search
        '''
        inputs_str = prompt.strip()
        inputs = self.tokenizer(inputs_str, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        try:
            outputs = self.model.generate(inputs=input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                early_stopping=False,
                # stopping_criteria=stopping_criteria if stopping_criteria else None,
            )
        except Exception as e:
            raise Exception(f"Error generating text: {e}")

        curr_token = ""
        sentence = "<|endoftext|>"
        first_token = True
        for output in outputs:
            next_token = output
            if len(next_token.size()) > 1: continue # skip the last generated full array
            if curr := self.tokenizer.convert_ids_to_tokens(
                next_token, skip_special_tokens=True
            ):
                curr = curr[0] # string with special character potentially
                if (curr[0] == "Ġ"): # BPE tokenizer
                    curr_token = curr_token.replace("Ċ", "\n")
                    curr_token = curr.replace("Ġ", " ")
                    sentence += curr_token # we can yield here/print here
                    yield curr_token
                    # BPE
                elif (curr[0] == "▁"): # sentence piece tokenizer
                    if first_token:
                        curr_token = curr.replace("▁", "")
                        first_token = False
                    else:
                        curr_token = curr.replace("▁", " ")
                    sentence += curr_token # we can yield here/print here
                    yield curr_token
                else:
                    curr_token = curr
                    curr_token = curr_token.replace("Ċ", "\n")
                    yield curr_token
                    sentence += curr_token

                curr_token = ""

        # dispatch last token, if we can
        if curr_token != "": 
            yield curr_token # send only if non empty

        logger.info(f'[COMPLETION]: {sentence}')           

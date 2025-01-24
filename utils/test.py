import bardapi
import os
import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from accelerate import PartialState
from accelerate.utils import set_seed

import argparse
import inspect
import logging
from typing import Tuple

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

model_id="./models_hf/7B"

model_id = '/home/baebro/hojun_ws/LMM based 3D Scene Graph Generation(L3DSG)/L3DSG_New/LMM Prompting/models_hf/7B'

tokenizer = LlamaTokenizer.from_pretrained(model_id)

model =LlamaForCausalLM.from_pretrained(model_id)#.to('cuda:1')
                                                     #'#, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

for i in ['armchair','lamp','bed']:
    prompt = 'What are the typical shape, componets, color of a {}?'.format(i)
    model_input = tokenizer(prompt, return_tensors="pt")#.to("cuda:1")

    model.eval()
    # with torch.no_grad():from_pretrained
    model_response = model.generate(**model_input, max_new_tokens=100)
    x = model_response[0]
    result = tokenizer.decode(x,skip_special_tokens=True)
    print(result.split('\n'))


# -*- coding: utf-8 -*-
# !huggingface-cli login

import torch
from datetime import datetime
import torch._dynamo.config
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelForCausalLM
import logging as log
import os
from peft import  PeftModel
    
    
log_directory = './logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')
print(torch.__version__)
outfile = log_directory + "/llama32_"+ time_hash +'.log'


log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])


def create_prompt(question):
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant to help users<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        system_message = "You are a helpful assistant.Please answer the question if it is possible"
       
        prompt_template=f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        #print(prompt_template)
        return prompt_template
#--------------------------------------------------------------------------------------------------------
# Define a function to process prompts
def process_prompt(prompt, model, tokenizer, device, max_length=250):
    """Processes a prompt, generates a response, and logs the result."""
    prompt_encoded = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt")
    model.eval()
    output = model.generate(
        input_ids=prompt_encoded.input_ids.to(device),
        max_length=max_length,
        attention_mask=prompt_encoded.attention_mask.to(device)
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    log.info("-"*80)
    log.info(f"Model Question:  {prompt}")
    log.info("-"*80)
    log.info(f"Model answer:  {answer}")
    log.info("-"*80)
    
def create_prompt(question,system_message):
    """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a helpful AI assistant to help users<|eot_id|><|start_header_id|>user<|end_header_id|>

    What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    if not system_message:
        system_message = "You are a helpful assistant.Please answer the question if it is possible"
    
    prompt_template=f'''
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    '''
    #print(prompt_template)
    return prompt_template

def get_chat_response(self,output):
    # outputcont = "".join(output)
    # parts = outputcont.split("[/INST]", 1)
    return output

####################################################################################################33
#               MAIN PROGRAM
####################################################################################################33
model_name =  "meta-llama/Llama-3.2-1B-Instruct" 

## Loading the Model
############################################################################################################
# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
    bf16 = True
    fp16 = False

# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next

############################################################################################################




# Define paths
checkpoint_dir = f"/ssd/tranformer_learn/models/checkpoint-64550"
# Load quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,  # Replace with your base model name
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({"additional_special_tokens": ["<<AFC6CODE>>"]})
#base_model.resize_token_embeddings(len(tokenizer))


# Load adapters
peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)

# Load tokenizer
# Verify inference
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = peft_model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))


while True:
    print('\nPlease ask the question or press q to quit')
    question = input()
    if 'q' == question:
        print("Exiting")
        break
    print(f'Processing Message from input()')
    prompt = f"{ question}"
    system_message = "You are a helpful assistant.Please answer the question if it is possible"
    prompt_template = create_prompt(prompt,system_message)
    inputs = tokenizer(prompt_template, return_tensors="pt",truncation=True,max_length=2000)
    outputs = peft_model.generate(**inputs.to(device) ,max_new_tokens=2000)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    log.info(output)
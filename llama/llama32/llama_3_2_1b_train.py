# -*- coding: utf-8 -*-

# due to rope related vllm bug #ValueError: `rope_scaling` must be a dictionary with two fields, `type` and `factor`, got {'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
#!pip install -q vllm==0.5.3.post1
# !pip install -q --upgrade transformers vllm accelerate
# !pip install -q huggingface_hub
# !mkdir -p ./llama3.1-quantised

# !huggingface-cli login

"""# Train  LLAMA 3.2 with direct data for Fine tuning (without Instruct Tuning dataset

**This only works on GPUs like A100 which has  bfloat16 (bf16="True") in the BitsandBytes Config** and need about 40 GB GPU RAM
"""

import torch
import shutil
from transformers import  get_linear_schedule_with_warmup # for training
from datetime import datetime
import torch._dynamo.config
from torch.cuda.amp import autocast,GradScaler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')
print(torch.__version__)
import logging as log
import torch.nn as nn

import os
log_directory = './logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
outfile = log_directory + "/llama32_"+ time_hash +'.log'



log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])

####################################################################################################33
# from Karpathy and modified
# https://github.com/karpathy/nanoGPT/blob/086ebe1822791b775e951b4b562fbb7131d83cc2/train.py
def get_random_batch(len_train_data,input_ids,attention_mask,block_size=1024,
                    batch_size=12):
    """
    To get back the data
        x,y= get_random_batch(len_train_data,input_ids,attention_mask,
                        block_size=block_size,batch_size=train_batch_size)
        input_dec = tokenizer.decode(x[0,:].squeeze(), skip_special_tokens=False)
        log.info(f"Decoded x : '{input_dec}'")# correct
    """
    # random select from training data set
    ix = torch.randint(0,len_train_data-block_size , (batch_size,))
    x = torch.stack([(input_ids[i:i+block_size]) for i in ix])
    y = torch.stack([((attention_mask[i:i+block_size])) for i in ix])
    return x, y
#--------------------------------------------------------------------------------------------------------
def train_model(model, tokenizer, input_ids, attention_mask, len_train_data, 
                 num_train_epochs, train_batch_size, block_size, device, 
                 optimizer, lr_scheduler, save_epochs, test_prompt_encoded, torch_dtype,save):
    scaler = GradScaler()

    for epoch in range(num_train_epochs):
        log.info(f"Epoch {epoch+1} of {num_train_epochs}")
        epoch_loss = 0
        for i in range(0, len_train_data, block_size):
            # Get data in random per batch from input
            # not all training data may not be covered in one epoch here
            x, y = get_random_batch(len_train_data, input_ids, attention_mask,
                                    block_size=block_size, batch_size=train_batch_size)
            # attention_mask given by tokenize is array of ones= [1,1,..], that is attend to all tokens
            with autocast(dtype=torch_dtype):
                outputs = model(input_ids=x.to(device), attention_mask=y.to(device), labels=x.to(device))
                loss = outputs.loss
                epoch_loss += loss.item()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()

            # Accumulate loss for logging
            epoch_loss += loss.item()

            checkpoint_dir = f"./llama3.1_trained/llama3-{epoch+1}-{time_hash}"
        average_epch_loss = epoch_loss/num_train_epochs
        log.info(f"Epoch {epoch} complete. Loss: {average_epch_loss}")
            # if epoch % save_epochs ==0 and epoch !=0  and save:
                # model.eval()
                # test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=250,
                #                 attention_mask = test_prompt_encoded.attention_mask.to(device))
                # test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
                # log.info(f"Over-fit check answer: Epoch {epoch} {test_answer}")
                # log.info(f"Epoch {epoch} over saving  model in {checkpoint_dir}")
                # #torch.save(model, checkpoint_dir) # AttributeError: Can't pickle local object 'add_hook_to_module.<locals>.new_forward'
                #  #torch.save(model.state_dict(), checkpoint_dir)
                # model.save_pretrained(checkpoint_dir)
                # log.info(f"Model saved")
                # model.train()
                # log.info(f"Epoch {epoch} complete. Loss: {average_epch_loss} saving {checkpoint_dir}")
                # #delete the previous save epoch
                # checkpoint_dir_del = f"./llama3.1_trained/llama3-{epoch}-{time_hash}"
                # try:
                #     shutil.rmtree(checkpoint_dir_del)
                #     log.info(f"Delete previously saved {checkpoint_dir_del}")
                # except:
                #     pass

 

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

def freeze_and_unfreeze_layers(model, target_layers):
    """Freezes all layers of the model and then unfreezes specific target layers.

    Args:
        model: The model to freeze and unfreeze layers.
        target_layers: A list of layer names to unfreeze.
    """

    # Freeze all layers
    for name, param in model.named_parameters():
        param.requires_grad = False

     # Unfreeze target layers
    for name, param in model.named_parameters():
        #log.info(f"Parameter {name}")
        if any(layer in name for layer in target_layers):#; and name in ['self_attn.o','self_attn.k','self_attn.q','self_attn.v']:
            log.info(f"Unfreezing {name}")
            param.requires_grad = True

    # Verify which layers are trainable
    for name, param in model.named_parameters():
        if param.requires_grad:
            log.info(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")


            
####################################################################################################33
#               MAIN PROGRAM
####################################################################################################33
model_name = 'meta-llama'
model_name_long =  "meta-llama/Llama-3.2-1B-Instruct" #
#model_name_long =  "distilbert/distilgpt2" # for test

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
#tokenizer.pad_token = tokenizer.eos_token

input_file_path = './data/nsp_troubleshooting.txt' # a small training file to learn

with open(input_file_path, 'r') as f:
    input_text = f.read()
log.info(f"Training data {input_file_path}")
log.info(f"length of dataset in words: {len(input_text):,}") #252,023
encoding = tokenizer(input_text, truncation=False, padding=False,return_tensors='pt')
log.info(f"encoding.input_ids.shape {encoding.input_ids.shape}")
log.info(f"encoding.attention_mask.shape {encoding.attention_mask.shape}")
len_train_data = encoding.input_ids.shape[1]
log.info(f"length of dataset in tokens = {len_train_data}")

"""## Loading the Model"""
# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Going to load the model {model_name_long}")
bf16 = False
fp16 = True

major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
    bf16 = True
    fp16 = False

# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next
#device = torch.device('cuda:0')

# Load base model
if bf16:
    torch_dtype=torch.bfloat16
else:
    torch_dtype=torch.float16 # it is not possible to train in float16  #https://github.com/huggingface/transformers/issues/23165

log.info(f"Going to load the model {model_name_long} ")
# This works, this is training the qunatised model
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    #device_map=device_map,

)
#model = nn.DataParallel(model)
model = model.to(device)
log.info(f"Loaded model with torch_dtype={torch_dtype}")


# Access and print the transformer layers
# print("Transformer Layers:")
# for i, layer in enumerate(model.model.layers):
#     print(f"Layer {i}: {layer}")



"""## Now to start the training"""

# Add a test prompt to check over-fitting
test_prompt = "How to check if NFM-P database is running"
test_prompt = create_prompt(test_prompt)
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False, return_tensors="pt")
# flatten the tensor from  torch.Size([1, xx]) to  torch.Size([xxx])
input_ids=encoding.input_ids.view(-1)
attention_mask=encoding.attention_mask.view(-1)

log.info(f"tokenizer.model_max_length = {tokenizer.model_max_length}, Learning Rate =3e-5")
# Set up the training parameters
block_size = int((len_train_data-1)/1) # CUDA out of memory. Tried to allocate 43.67 GiB
block_size = 500 # for uses 24/40 gb with model loaded A100 3 bad results
train_batch_size = 1 # 38 GB/40 in A100
num_train_epochs = 20
save_epochs =num_train_epochs
# Set the optimizer and learning rate scheduler
# num_warmup_steps = 100
max_grad_norm = 1.0
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_train_steps = len_train_data // train_batch_size * num_train_epochs
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
lr_scheduler = None
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) #todo was  3e-5
log.info(f"len_train_data={len_train_data} block_size ={block_size} batch_size= {train_batch_size}")
x,y= get_random_batch(len_train_data,input_ids,attention_mask,
                block_size=block_size,batch_size=train_batch_size)
input_dec = tokenizer.decode(x[0,:].squeeze(), skip_special_tokens=False)
log.info(f"Decoded x : '{input_dec}'")# correct
log.info("-"*80)

############################################################################################################3

# Define your prompts
prompts = [
    "NSP resynchronization with network devices is not happening, what should I do?",
    "I am not able to start  NFM-P client ",
    "NE accounts are not seen in the NFM-P client GUI",
    "Cannot select some menu options or save some configurations, what could be the reason",
    "How do I onboard a NSP service",
    "Who is Bill Gates and what is his great name to fame",
]
    
target_layers = [f'.{i}' for i in range(8, 12)] # firs 10 to 20, next , 20 to 30 # last 30 to 32
freeze_and_unfreeze_layers(model, target_layers)

model.train()
train_model(model, tokenizer, input_ids, attention_mask, len_train_data, 
             num_train_epochs, train_batch_size, block_size, device, 
             optimizer, lr_scheduler, save_epochs, test_prompt_encoded, torch_dtype,False)
# Process each prompt
model.eval()
for prompt in prompts:
    process_prompt(prompt, model, tokenizer, device)

checkpoint_dir = f"./llama3.2_trained/llama3-final"
log.info(f"Training over saving  model in {checkpoint_dir}")
model.save_pretrained(checkpoint_dir)

log.info("Training over- Exiting normally")

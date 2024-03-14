 #!/usr/bin/env python.
# Module to train the model


import torch
import shutil
from utils import get_random_batch
from transformers import  get_linear_schedule_with_warmup # for training
import logging as log
from datetime import datetime
import torch._dynamo.config
from torch.cuda.amp import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

print(torch.__version__)
#torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
# torch._dynamo.config.log_level = log.DEBUG
# torch._dynamo.config.verbose = True

time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')
outfile = "./training/training_" +time_hash +".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(outfile),
        log.StreamHandler()
    ]
)

model_name = 'mistral'
model_name_long ='mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
#tokenizer.pad_token = tokenizer.eos_token

input_file_path = './output.mdx' # a very small training file to learn

with open(input_file_path, 'r') as f:
    input_text = f.read()
log.info(f"Training data {input_file_path}")
log.info(f"length of dataset in words: {len(input_text):,}") #252,023
encoding = tokenizer(input_text, truncation=False, padding=False,return_tensors='pt')
log.info(f"encoding.input_ids.shape {encoding.input_ids.shape}")
log.info(f"encoding.attention_mask.shape {encoding.attention_mask.shape}")
len_train_data = encoding.input_ids.shape[1]
log.info(f"length of dataset in tokens = {len_train_data}")
# Add a test prompt to check over-fitting
test_prompt = 'How do I get started with Network as code ?'
test_prompt = f'<s>[INST]{test_prompt}[/INST]'
#Ideal answer from gpt2 base model is something like below
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False, return_tensors="pt")
# Gandhi was born in 1867. He was the son of a farmer and a merchant. He was educated at the University of Delhi. He was a member of the Indian National Congress. He was a member
# flatten the tensor from  torch.Size([1, xx]) to  torch.Size([xxx])
input_ids=encoding.input_ids.view(-1)
attention_mask=encoding.attention_mask.view(-1)

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Going to load the model {model_name_long}")

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

bf16 = False    
fp16 = True

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        bf16 = True    
        fp16 = False
        
# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next
device = torch.device('cuda:0')

# Load base model
if bf16:
    torch_dtype=torch.bfloat16
else:
    torch_dtype=torch.float16

log.info(f"Going to load the model {model_name_long} in 4 bit Quanitsed mode {bnb_config} ")
# This works, this is training the qunatised model
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    quantization_config=bnb_config,
    device_map=device_map
)

log.info("Loaded model in 4 bit Quantised form")
# lets us try LoRA also now 

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 32 #16

# Dropout probability for LoRA layers
lora_dropout = 0.1


# Load LoRA configuration
lora_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],

)
# Loss is not decreaing with this
# log.info(f"Going to load the model {model_name_long} with LoRA  {lora_config} ")
# model = get_peft_model(model, lora_config)
# log.info("Loaded model with LoRA")

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

log.info(f"GPU Free memory after loading is {f/(1.25*10**8)} GB")

model.config.use_cache = False
model.config.pretraining_tp = 1

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) 
log.info(f"tokenizer.model_max_length = {tokenizer.model_max_length}, Learning Rate =3e-5")
# Set up the training parameters
train_batch_size = 1
block_size = int((len_train_data-1)/10)
if len_train_data > tokenizer.model_max_length:
    block_size = int(tokenizer.model_max_length/8) # tokenizer.model_max_length=1024
num_train_epochs = 100
save_epochs = 20

# Set the optimizer and learning rate scheduler
# num_warmup_steps = 100
# max_grad_norm = 1.0
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_train_steps = len_train_data // train_batch_size * num_train_epochs
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
log.info(f"len_train_data={len_train_data} block_size ={block_size} batch_size= {train_batch_size}")
model.train()
with autocast(dtype=torch.bfloat16):
    for epoch in range(num_train_epochs):
        log.info(f"Epoch {epoch+1} of {num_train_epochs}")
        epoch_loss = 0
        for i in range(0,len_train_data, block_size):
            # Get data in random per batch from input
            # not all training data may not be covered in one epoch here
            x,y= get_random_batch(len_train_data,input_ids,attention_mask,
                block_size=block_size,batch_size=train_batch_size)
            # attention_mask given by tokenize is array of ones= [1,1,..], that is attend to all tokens
            outputs = model(input_ids=x.to(device),attention_mask=y.to(device),labels=x.to(device))
            loss = outputs.loss
            epoch_loss += loss.item()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
        # Save the model checkpoint every 10th
     
        checkpoint_dir = f"./mistral-directcqlora/epoch-{epoch+1}-{time_hash}"
        average_epch_loss = epoch_loss/num_train_epochs
        if epoch % save_epochs ==0:
             #model.save_pretrained(checkpoint_dir)#ou are calling `ave_pretrained` on a 4-bit converted model. This is currently not supported
            model.eval()
            test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=250,
                            attention_mask = test_prompt_encoded.attention_mask.to(device))
            test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
            log.info(f"Over-fit check answer: Epoch {epoch} {test_answer}")
            #torch.save(model, checkpoint_dir) # AttributeError: Can't pickle local object 'add_hook_to_module.<locals>.new_forward'
            torch.save(model.state_dict(), checkpoint_dir) 
            model.train()
            log.info(f"Epoch {epoch} complete. Loss: {average_epch_loss} saving {checkpoint_dir}")
       
        log.info(f"Epoch {epoch} complete. Loss: {average_epch_loss}")
      
        #delete the previous save epoch
        checkpoint_dir = f"./mistral-directcqlora/epoch-{epoch}-{time_hash}"
        try:
            shutil.rmtree(checkpoint_dir)
        except:
            pass
        
    model.eval()
    test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=250,
                       attention_mask = test_prompt_encoded.attention_mask.to(device))
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    log.info(f"Over-fit check answer: {test_answer}")
    model.train()
    log.info(f"Training over saving fill model in {checkpoint_dir}")
    torch.save(model.state_dict(), checkpoint_dir ) 
    log.info(f"Model saved")
        



# Module to train the model
from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch
import shutil
from utils import get_batch
from transformers import  get_linear_schedule_with_warmup # for training
import logging as log
from datetime import datetime
import torch._dynamo.config
from torch.cuda.amp import autocast

print(torch.__version__)
#torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
# torch._dynamo.config.log_level = log.DEBUG
# torch._dynamo.config.verbose = True

time_hash=str(datetime.now()).strip()
outfile = "./training/training_" +time_hash +".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(outfile),
        log.StreamHandler()
    ]
)

model_name = 'llama'
model_name_long ='decapoda-research/llama-7b-hf'

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
#tokenizer.pad_token = tokenizer.eos_token

# Read the cleaned input text
# original data Canon Camera manual https://gdlp01.c-wss.com/gds/0/0300004730/02/eosrt3-eos1100d-im2-c-en.pdf
#input_file_path = './data/clean2.txt' # manually cleaned the indices part
input_file_path = './data/small_2.txt' # a very small training file to learn
# original data https://www.gutenberg.org/files/17921/17921-0.txt as is
#input_file_path = './data/17921-0-cleaned.txt'
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
test_prompt = 'Where is New York City located'
#Ideal answer from gpt2 base model is something like below
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False, return_tensors="pt")
# Gandhi was born in 1867. He was the son of a farmer and a merchant. He was educated at the University of Delhi. He was a member of the Indian National Congress. He was a member
# flatten the tensor from  torch.Size([1, xx]) to  torch.Size([xxx])
input_ids=encoding.input_ids.view(-1)
attention_mask=encoding.attention_mask.view(-1)

# # Freeze bottom n layers
# #https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d
# for parameter in model.parameters():
#      parameter.requires_grad = False
# n =8 # last four layers
# for i, m in enumerate(model.transformer.h):        
#     #Only un-freeze the last n transformer blocks
#     if i >= n:
#         for parameter in m.parameters():
#             parameter.requires_grad = True 
#         log.info(f"Un-freezed layer {i} for training")

# for parameter in model.transformer.ln_f.parameters():        
#     parameter.requires_grad = True

# for parameter in model.lm_head.parameters():        
#     parameter.requires_grad = True

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Going to load the model {model_name_long}")
# Load the llama model 
model =AutoModelForCausalLM.from_pretrained(model_name_long,torch_dtype=torch.float16).to(device)
log.info("Loaded the model")

#model.to(device)
# learning_rate = 6e-4 # ??

model.eval()
test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=50,
                    num_return_sequences=1)
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
log.info(f"Over-fit check answer: {test_answer}")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) 

# Set up the training parameters
train_batch_size = 4
block_size = len_train_data-1
if len_train_data > tokenizer.model_max_length:
    block_size = int(tokenizer.model_max_length/4) # tokenizer.model_max_length=1024
num_train_epochs = 20 #100

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
            x,y= get_batch(len_train_data,input_ids,attention_mask,device,
                block_size=block_size,batch_size=train_batch_size)
            # attention_mask given by tokenize is array of ones= [1,1,..], that is attend to all tokens
            outputs = model(input_ids=x,attention_mask=y,labels=x)
            loss = outputs.loss
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
        # Save the model checkpoint every 10th
        checkpoint_dir = f"./test-gpt2-2/{model_name}-epoch-{epoch+1}-{time_hash}"
        model.save_pretrained(checkpoint_dir)
        log.info(f"Epoch {epoch} complete. Loss: {loss.item()} saving {checkpoint_dir}")
        model.eval()
        test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=250,
                            num_return_sequences=1)
        test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
        log.info(f"Over-fit check answer: {test_answer}")
        model.train()
        #delete the previous save epoch
        checkpoint_dir = f"./test-gpt2-2//{model_name}-epoch-{epoch}-{time_hash}"
        try:
            shutil.rmtree(checkpoint_dir)
        except:
            pass



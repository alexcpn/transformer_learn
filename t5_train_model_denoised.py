# Module to train the model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import shutil
from utils import get_batch_denoised
from transformers import  get_linear_schedule_with_warmup # for training
import logging as log
from datetime import datetime
import re
import torch._dynamo.config

print(torch.__version__)
torch.set_float32_matmul_precision('high')
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

model_name = 't5-base'

tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Read the cleaned input text
# original data Canon Camera manual https://gdlp01.c-wss.com/gds/0/0300004730/02/eosrt3-eos1100d-im2-c-en.pdf
#input_file_path = './data/clean2.txt' # manually cleaned the indices part
# original data https://www.gutenberg.org/files/17921/17921-0.txt as is
input_file_path = './data/17921-0-cleaned.txt'
with open(input_file_path, 'r') as f:
    input_text = f.read()
log.info(f"Training data {input_file_path}")
log.info(f"length of dataset in words: {len(input_text):,}") #252,023
encoding = tokenizer(input_text, truncation=False, padding=True,return_tensors='pt')
log.info(f"encoding.input_ids.shape {encoding.input_ids.shape}")
log.info(f"encoding.attention_mask.shape {encoding.attention_mask.shape}")
len_train_data = encoding.input_ids.shape[1]
log.info(f"length of dataset in tokens = {len_train_data}")
# Add a test prompt to check overfitting
test_prompt = 'When was Gandhi born'
#Ideal answer from gpt2 base model is something like below
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False, return_tensors="pt")
# Gandhi was born in 1867. He was the son of a farmer and a merchant. He was educated at the University of Delhi. He was a member of the Indian National Congress. He was a member
# flatten the tensor from  torch.Size([1, xx]) to  torch.Size([xxx])
input_ids=encoding.input_ids.view(-1)
attention_mask=encoding.attention_mask.view(-1)
# Note , if we give truncation as False then the token sequence length goes more than model_max_length
# Token indices sequence length is longer than the specified maximum sequence length for this
#  model (23552 > 1024). Running this sequence through the model will result in indexing errors
# However we are not running through the model; We will add it to an array and train with block_size

# Load the T5 model 
model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Freeze bottom n layers
# #https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d


# Freeze the top 4 layers of the T5 model
for param in model.encoder.block[:4].parameters():
    param.requires_grad = False

for parameter in model.parameters():
     parameter.requires_grad = False
n =8 # last four layers
for i, m in enumerate(model.encoder.block):        
    #Only un-freeze the last n transformer blocks
    if i >= n:
        for parameter in m.parameters():
            parameter.requires_grad = True 
        log.info(f"Un-freezed layer {i} for training")

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # We could optimize training using Pytorch 2.0
# # It neds cuda 11.6+ ,  it crashes as of now
# if int(re.search(r'\d+', torch.__version__).group()) >= 2:
#     # for pytorch 2.0
#     model =torch.compile(model)
#     log.info(f"Compiled the model for speed up")

model.to(device)

model.eval()
test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=50,
                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
log.info(f"Over-fit check answer: {test_answer}")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) 

# Set up the training parameters
train_batch_size = 8
block_size = len_train_data-1
if len_train_data > tokenizer.model_max_length:
    block_size = int(tokenizer.model_max_length/8) # tokenizer.model_max_length=1024
num_train_epochs = 20

# Set the optimizer and learning rate scheduler
# num_warmup_steps = 100
# max_grad_norm = 1.0
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_train_steps = len_train_data // train_batch_size * num_train_epochs
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
log.info(f"len_train_data={len_train_data} block_size ={block_size} batch_size= {train_batch_size}")
model.train()
for epoch in range(num_train_epochs):
    log.info(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    for i in range(0,len_train_data, block_size):
        # Get data in random per batch from input
        # not all training data may not be covered in one epoch here
        x, l, y= get_batch_denoised(len_train_data,input_ids,
             attention_mask,device,block_size,train_batch_size,
             len(tokenizer),tokenizer.eos_token_id)
        #print(f"input_ids={x.shape}, labels={l.shape}, attention_mask={y.shape}")
        outputs = model(input_ids=x,attention_mask=y,labels=l)
        loss = outputs.loss
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
    # Save the model checkpoint every 10th
    checkpoint_dir = f"./test-t5/{model_name}-epoch-{epoch+1}-{time_hash}"
    model.save_pretrained(checkpoint_dir)
    log.info(f"Epoch {epoch} complete. Loss: {loss.item()} saving {checkpoint_dir}")
    model.eval()
    # Check if the model has overfitted
    test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=250,
                        num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    log.info(f"Over-fit check answer: {test_answer}")
    model.train()
    # delete the previous save epoch
    # checkpoint_dir = f"./model/{model_name}-epoch-{epoch}-{time_hash}"
    # try:
    #     shutil.rmtree(checkpoint_dir)
    # except:
    #     pass



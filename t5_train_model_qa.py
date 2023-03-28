# Module to train the model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import shutil
from utils import get_batch_for_qa
from transformers import  get_linear_schedule_with_warmup # for training
import logging as log
from datetime import datetime
import pandas as pd
import torch._dynamo.config
import numpy as np

print(torch.__version__)
torch.set_float32_matmul_precision('low')
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
#model_name = 'google/t5-small-ssm-nq'
log.info(f"Model Name {model_name}")
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv('./data/qa_dataset2.csv',index_col=False,sep='|')# see create_qa_Set.py
len_train_data =len(df.index)
print(f"Total NER dataset rows={len_train_data}")
#model_name = 'google/t5-small-ssm-nq'
tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.pad_token = tokenizer.eos_token

# Add a test prompt to check over-fitting
test_prompt = 'describe: Conditions of Impaired Mobility of Joints' # for t5-base
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False, return_tensors="pt")

# Load the T5 model 
model_name = './test5-t5-good/t5-base-epoch-100-2023-03-25 20:35:35.638512'
model = T5ForConditionalGeneration.from_pretrained(model_name)
log.info(f"Loaded base model_name {model_name}")
model_name = 't5-base-trained-medical'
# Freeze bottom n layers
#https://towardsdatascience.com/conditional-text-generation-by-fine-tuning-gpt-2-11c1a9fc639d


# Freeze all the layers of the T5 model
# for param in model.encoder.block.parameters():
#     param.requires_grad = False

# # Unfreeze the last total
# n =8 # last four layers  for t5-base 12 layer
# #n =22 # last four layers for t5-large 24 layers
# for i, m in enumerate(model.encoder.block):        
#     #Only un-freeze the last n transformer blocks
#     if i >= n:
#         for parameter in m.parameters():
#             parameter.requires_grad = True 
#         log.info(f"Un-freezed layer {i} for training")

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
test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),
                   min_new_tokens=50, max_new_tokens=250,num_beams=10, no_repeat_ngram_size=1)
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
log.info(f"Over-fit check answer: {test_answer}")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) 

# Set up the training parameters
train_batch_size =1
num_train_epochs = 50

# Set the optimizer and learning rate scheduler
# num_warmup_steps = 100
# max_grad_norm = 1.0
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_train_steps = len_train_data // train_batch_size * num_train_epochs
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
log.info(f"len_train_data={len_train_data}  batch_size= {train_batch_size}")
model.train()
for epoch in range(num_train_epochs):
    log.info(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    for i in range(0,len_train_data, train_batch_size):
        # Get data in random per batch from input
        # not all training data may not be covered in one epoch here
        x,y,m= get_batch_for_qa(df,device,tokenizer,batch_size=train_batch_size)
        print(f"index={i} input_ids={x.shape}, labels={y.shape},attention_mask={m.shape}")
        #index=221 input_ids=torch.Size([1, 9]), labels=torch.Size([1, 151]),attention_mask=torch.Size([1, 9])
        loss = 99
        while loss > 0.5: # reduce loss in each batch and proceed
            outputs = model(input_ids=x,attention_mask=m,labels=y)
            loss = outputs.loss
            print(f"In batch loss={loss}")
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad()
    #Save the model checkpoint
    checkpoint_dir = f"./test10-t5/{model_name}-epoch-{epoch+1}-{time_hash}"
    model.save_pretrained(checkpoint_dir)
    log.info(f"Epoch {epoch} complete. Loss: {loss.item()} saving {checkpoint_dir}")
    model.eval()
    #Check if the model has over-fitted
    test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),
                   min_new_tokens=50, max_new_tokens=250,num_beams=10, no_repeat_ngram_size=1)
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    log.info(f"Over-fit check answer: {test_answer}")
    model.train()
    #delete the previous save epoch
    checkpoint_dir = f"./test10-t5/{model_name}-epoch-{epoch}-{time_hash}"
    try:
        shutil.rmtree(checkpoint_dir)
    except:
        pass




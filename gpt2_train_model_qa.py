# Module to train the model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import shutil
from utils import get_batch_for_qa_gpt
from transformers import  get_linear_schedule_with_warmup # for training
import logging as log
from datetime import datetime
import pandas as pd
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

model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# + model.resize_token_embeddings(len(tokenizer))

#tokenizer.pad_token = tokenizer.pad_token
df = pd.read_csv('./data/qa_small_2.csv',index_col=False,sep='|')# see create_qa_Set.py
len_train_data =len(df.index)
print(f"Total NER dataset rows={len_train_data}")

# Add a test prompt to check over-fitting
test_prompt = 'Where is New York City'
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False, return_tensors="pt")

# Load the base model
model_name = './test-gpt2-2/gpt2-epoch-20-2023-04-15 16:31:57.913960'
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
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

# # We could optimize training using Pytorch 2.0
# # It neds cuda 11.6+ ,  it crashes as of now
# if int(re.search(r'\d+', torch.__version__).group()) >= 2:
#     # for pytorch 2.0
#     model =torch.compile(model)
#     log.info(f"Compiled the model for speed up")

model.to(device)
# learning_rate = 6e-4 # ??

model.eval()
test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),
                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
log.info(f"Over-fit check answer: {test_answer}")

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) 

# Set up the training parameters
train_batch_size = 2
num_train_epochs = 20

# Set the optimizer and learning rate scheduler
num_warmup_steps = 10
max_grad_norm = 1.0
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_train_steps = len_train_data // train_batch_size * num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
log.info(f"len_train_data={len_train_data} batch_size= {train_batch_size}")
model.train()
for epoch in range(num_train_epochs):
    log.info(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    for i in range(0,len_train_data, train_batch_size):
        # Get data in random per batch from input
        # not all training data may not be covered in one epoch here
        x,y,m= get_batch_for_qa_gpt(df,device,tokenizer,batch_size=train_batch_size)
        print(f"index={i} input_ids={x.shape}, labels={y.shape},attention_mask={m.shape}")
        outputs = model(input_ids=x,attention_mask=m,labels=y)
        # uncomment to check if get_batch_for_qa_gpt is returning correctly
        # for i in range(train_batch_size):
        #     input_dec = tokenizer.decode(x[i,:].squeeze(), skip_special_tokens=False)
        #     log.info(f"Decoded check: {i} {input_dec}")# correct
        #     # 2023-04-18 13:03:58,002 [INFO] Decoded check: 0 Which is the second largest city in the Unites States?<|endoftext|>Los Angels is the second largest city in New York [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
        #     # 2023-04-18 13:03:58,002 [INFO] Decoded check: 1 New York population [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] <|endoftext|>The population of New York City as of 2020 is 8,804,190 distributed over 300.46 square miles (778.2 km2)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    # Save the model checkpoint every 10th
    checkpoint_dir = f"./test-gpt2-4/{model_name}-epoch-{epoch+1}-{time_hash}"
    model.save_pretrained(checkpoint_dir)
    log.info(f"Epoch {epoch} complete. Loss: {loss.item()} saving {checkpoint_dir}")
    model.eval()
    test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),max_length=250,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True, 
                        top_k=50, 
                        top_p=0.95, 
                        num_return_sequences=3)
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    log.info(f"Over-fit check answer: {test_answer}")
    model.train()
    #delete the previous save epoch
    checkpoint_dir = f"./test-gpt2-4/{model_name}-epoch-{epoch}-{time_hash}"
    try:
        shutil.rmtree(checkpoint_dir)
    except:
        pass



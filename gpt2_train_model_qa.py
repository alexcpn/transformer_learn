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

tokenizer = GPT2Tokenizer.from_pretrained('gpt2',padding_side='left')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

df = pd.read_csv('./data/small_3_qa2.csv',index_col=False,sep='|')

train_df=df.sample(frac=0.8,random_state=200)
test_df=df.drop(train_df.index)
len_test_data = len(test_df.index)
len_train_data = len(train_df.index)

log.info(f"Total QA dataset rows={len_train_data}")
log.info(f"len_train_data={len_train_data}")
log.info(f"len_train_data={len_test_data}")

# Load the model to fine tune

model_name = 'small3qa'
checkpoint_dir ='./small3-gpt2-5/gpt2-epoch-100-2023-04-19 13:57:47.849368'
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
model.resize_token_embeddings(len(tokenizer))

# ---------------- Freeze bottom n layers (Optional)---------------------------
# Idea from https://bit.ly/3NdbpvR

for parameter in model.parameters():
    parameter.requires_grad = False
n = 8  # last four layers
for i, m in enumerate(model.transformer.h):
    # Only un-freeze the last n transformer blocks
    if i >= n:
        for parameter in m.parameters():
            parameter.requires_grad = True
        log.info(f"Un-freezed layer {i} for training")

for parameter in model.transformer.ln_f.parameters():
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():
    parameter.requires_grad = True

# -------Use CUDA/GPU if Available---------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------- Add a test prompt to check over-fitting while training---------
test_prompt = 'What is granulation tissue?'
test_prompt_encoded = tokenizer(test_prompt, truncation=True, padding=True, return_tensors="pt")

test_output = model.generate(input_ids = test_prompt_encoded.input_ids.to(device),
                    num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
log.info(f"Over-fit check answer before training: {test_answer}")

# -----------Initialize optimizer and learning rate----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Set up the training parameters
train_batch_size = 16
test_batch_size =train_batch_size
num_train_epochs = 200
num_train_steps = len_train_data // train_batch_size * num_train_epochs

log.info(f"num_train_epochs={num_train_epochs} train_batch_size= {train_batch_size}")
log.info(f"num_train_steps={num_train_steps} test_batch_size= {test_batch_size}")

# Set the optimizer and learning rate scheduler
# num_warmup_steps = 10
# max_grad_norm = 1.0
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

# -----------------Start Training with saving after each epoch-----------------

model.train()
for epoch in range(num_train_epochs):
    log.info(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    count =0
    for i in range(0,len_train_data, train_batch_size):
        count += 1
        # Get data in random per batch from input
        # not all training data may not be covered in one epoch here
        x,y,m= get_batch_for_qa_gpt(train_df,device,tokenizer,batch_size=train_batch_size)
        log.debug(f"index={i} input_ids={x.shape}, labels={y.shape},attention_mask={m.shape}")
        outputs = model(input_ids=x,attention_mask=m,labels=y)
        # uncomment to check if get_batch_for_qa_gpt is returning correctly
        # for i in range(train_batch_size):
        #     input_dec = tokenizer.decode(x[i,:].squeeze(), skip_special_tokens=False)
        #     log.info(f"Decoded check: {i} {input_dec}")# correct
        #[INFO] Decoded check: 3 [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] Which tissue is used as a graft to fill defects in the dura mater? The fascia lata of the thigh is widely used as a graft to fill defects in the dura mater. [PAD]
        loss = outputs.loss
        loss.backward()
        epoch_loss += loss.item()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
    avg_epoch_loss = epoch_loss /count
    log.info(f"Train Epoch {epoch} complete.Avg Loss: {avg_epoch_loss}")
    checkpoint_dir = f"./small3-qa/{model_name}-epoch-{epoch+1}-{time_hash}"
    model.save_pretrained(checkpoint_dir)
    log.info(f"Saved Model at {checkpoint_dir}")
    model.eval()  # Set model to Eval loop
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
    checkpoint_dir = f"./small3-qa/{model_name}-epoch-{epoch}-{time_hash}"
    try:
        shutil.rmtree(checkpoint_dir)
    except:
        pass

     # -----------Calculate Validation loss-------------------------------------

    validation_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len_test_data, test_batch_size):
            count += 1
            x,y,m= get_batch_for_qa_gpt(test_df,device,tokenizer,batch_size=test_batch_size)
            log.debug(f"index={i} input_ids={x.shape}, labels={y.shape},attention_mask={m.shape}")
            outputs = model(input_ids=x, attention_mask=m, labels=y)
            validation_loss += outputs.loss.item()
    log.info(f"Train and Validation Epoch {epoch} complete. Averge Loss: {avg_epoch_loss} " +
                 f"Avg Validation Loss  {validation_loss/count}")


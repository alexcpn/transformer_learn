# Module to train the model
from transformers import T5ForConditionalGeneration ,T5Tokenizer
import torch
import shutil
from utils import  get_batch,shift_tokens_right
from transformers import  get_linear_schedule_with_warmup # for training
import logging as log
from datetime import datetime

time_hash = str(datetime.now()).strip()
outfile = "./logs/training_" + time_hash + ".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(outfile),
        log.StreamHandler()
    ]
)

#model_name = 't5-small'
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Read the cleaned input text
input_file_path = './data/small_3.txt'
with open(input_file_path, 'r') as f:
    input_text = f.read()
log.info(f"length of dataset in words: {len(input_text):,}") #252,023

encoding = tokenizer(input_text, truncation=False, padding=True,return_tensors='pt')
log.info(f"encoding.input_ids.shape {encoding.input_ids.shape}")
#encoding.input_ids.shape torch.Size([1, 48735])

log.info(f"encoding.attention_mask.shape {encoding.attention_mask.shape}")
len_train_data = encoding.input_ids.shape[1]
log.info(f"len_train_data = {len_train_data}")

 # flatten the tensor from  torch.Size([1, 48735]) to  torch.Size([48735])
input_ids=encoding.input_ids.view(-1)
attention_mask=encoding.attention_mask.view(-1)

# Note , if we give truncation as False then the token sequence length goes more than model_max_length
# Token indices sequence length is longer than the specified maximum sequence length for this
#  model (23552 > 1024). Running this sequence through the model will result in indexing errors
# However we are not running through the model; We will add it to an array and train with block_size

block_size = int(tokenizer.model_max_length/8) # tokenizer.model_max_length=1024

# Load the T5 model 
#model = T5ForConditionalGeneration.from_pretrained(model_name,device_map="auto", torch_dtype=torch.float16)
model = T5ForConditionalGeneration.from_pretrained(model_name,device_map="auto")

num_encoder_layers = len(model.encoder.block)  
num_decoder_layers = len(model.decoder.block)  

# freeze everything
for param in model.parameters():
     param.requires_grad = False

# Un-Freeze lower 4 layers of encoder (lower is unfreezed)
for i in range(0,num_encoder_layers-8,1):
    for param in model.encoder.block[i].parameters():
        param.requires_grad = True

# Un-Freeze lower 4 layers of decoder (lower is unfreezed)
for i in range(0,num_decoder_layers-8,1):
    for param in model.decoder.block[i].parameters():
        param.requires_grad = True

for name, param in model.named_parameters():
    if param.requires_grad:
        log.info(f"{name},{param.requires_grad}")

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# learning_rate = 6e-4 # ??
#optimizer = torch.optim.Adam(model.parameters(), lr=3e-5) - use get_linear_schedule_with_warmup

# Set up the training parameters
train_batch_size = 4
num_train_epochs = 100

# -----------Initialize optimizer and learning rate----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_train_steps = len_train_data // train_batch_size * num_train_epochs
# num_warmup_steps = 100
max_grad_norm = 1.0
#lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
#num_train_epochs =100

log.info(f"Batch Size {train_batch_size} Block Size {block_size} Epochs {num_train_epochs} Steps= {num_train_steps}")
model.train()

#scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_train_epochs):
    log.info(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    for i in range(0, len_train_data, block_size*train_batch_size):
        # Get data in random per batch from input
        # not all training data may not be covered in one epoch here
        x, y = get_batch(len_train_data, input_ids,attention_mask,
                            i, block_size=block_size,
                            batch_size=train_batch_size)
    
        # attention_mask given by tokenize is array of ones= [1,1,..],
        # that is attend to all tokens
        if device.type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device) 

        # for FP 16 training
        #with torch.autocast(device_type=device.type): #(device_type='cuda', dtype=torch.float16):
        #pad_token_id = model.config.pad_token_id
        #sx = shift_tokens_right(x,pad_token_id)

        # for i in range(train_batch_size): # there may not be a full batch
        #      input_dec = tokenizer.decode(x[i,:].squeeze(), skip_special_tokens=False)
        #      log.info(f"Decoded x :{i} '{input_dec}'")# correct
        #      input_dec = tokenizer.decode(sx[i,:].squeeze(), skip_special_tokens=False)
        #      log.info(f"Sifted x(=sx):{i} '{input_dec}'")# correct

        outputs = model(input_ids=x, attention_mask=y, labels=x)
        loss = outputs.loss
        epoch_loss += loss.item()

        #scaler.scale(loss).backward() # for FP 16 training
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # for FP 16 training especially
        #scaler.step(optimizer) # for FP 16 training
        optimizer.step()
        # lr_scheduler.step()
        # Updates the scale for next iteration.
        #scaler.update() # for FP 16 training
        optimizer.zero_grad()
    avg_epoch_loss = epoch_loss /(len_train_data/(block_size*train_batch_size))
    log.info(f"Train Epoch {epoch} complete.Avg Loss: {avg_epoch_loss}")
    if avg_epoch_loss < .006:
        log.info("Average epoch loss is very low,stopping training")
        break

    # Save the model checkpoint every 10th
    checkpoint_dir = f"./models/flan-t5-2/{model_name}-epoch-{epoch+1}-{time_hash}"
    model.save_pretrained(checkpoint_dir)
    log.info(f"Model saved at {checkpoint_dir}")
    # delete the previous save epoch, except middle point
    checkpoint_dir = f"./models/flan-t5-2/{model_name}-epoch-{epoch}-{time_hash}"
    try:
        shutil.rmtree(checkpoint_dir)
    except:
        pass



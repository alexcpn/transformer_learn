# Module to train the model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import shutil
from utils import get_batch
import logging as log
from datetime import datetime
import torch._dynamo.config

# -----------Optimization and debugging for Torch------------------------------

print(torch.__version__)
# torch.set_float32_matmul_precision('high')
# torch._dynamo.config.log_level = log.DEBUG
# torch._dynamo.config.verbose = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---------------Initialise Logging--------------------------------------------

time_hash = str(datetime.now()).strip()
outfile = "./training/training_" + time_hash + ".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(outfile),
        log.StreamHandler()
    ]
)

# ------------Initialise Tokenizer---------------------------------------------

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name,padding_side='left')
# tokenizer.pad_token = tokenizer.eos_token # wrong
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# --------------- Read the Pre-processed Input text data ----------------------

input_file_path = './data/small_3.txt'
with open(input_file_path, 'r') as f:
    input_text = f.read()

log.info(f"Training data {input_file_path}")
log.info(f"length of dataset in words: {len(input_text)}")

# ------------- Tokenize the text----------------------------------------------

encoding = tokenizer(input_text, truncation=False, padding=True,
             return_tensors='pt')

# Note , if we give truncation as False then the token sequence length goes more
# than model_max_length Token indices sequence length is longer than the specified
# maximum sequence length for this model (23552 > 1024). Running this sequence
# through the model will result in indexing errors. However we are not running 
# through  the model; We will add it to an array and train with block_size

log.info(f"encoding.input_ids.shape {encoding.input_ids.shape}")
log.info(f"encoding.attention_mask.shape {encoding.attention_mask.shape}")
len_data = encoding.input_ids.shape[1]
log.info(f"length of dataset in tokens = {len_data}")


# flatten the tensor from  torch.Size([1, xx]) to  torch.Size([xxx])

input_ids = encoding.input_ids.view(-1)
attention_mask = encoding.attention_mask.view(-1)

# ------------ Split the dataset into training and validation sets-------------

split_index = int(len_data * 0.8)
train_input_ids = input_ids[:split_index]
test_input_ids = input_ids[split_index:]

train_attention_mask = attention_mask[:split_index]
test_attention_mask = attention_mask[split_index:]

len_train_data = train_input_ids.shape[0]
len_test_data = test_input_ids.shape[0]

log.debug(f"train_input_ids.shape {train_input_ids.shape}")
log.debug(f"test_input_ids.shape {test_input_ids.shape}")
log.debug(f"train_attention_mask.shape {train_attention_mask.shape}")
log.debug(f"test_attention_mask.shape {test_attention_mask.shape}")

log.info(f"length of train dataset in tokens = {len_train_data}")
log.info(f"length of test dataset in tokens = {len_test_data}")

# ---------------- Load the  model --------------------------------------------

model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=50257)
# see https://stackoverflow.com/a/76045898/429476
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

# ----------Use Pytorch 2.0 if possible to optimize----------------------------

# # We could optimize training using Pytorch 2.0
# # It neds cuda 11.6+ ,  it crashes as of now
# if int(re.search(r'\d+', torch.__version__).group()) >= 2:
#     # for pytorch 2.0
#     model =torch.compile(model)
#     log.info(f"Compiled the model for speed up")

# -------- Load model and later data to device (GPU if available)--------------
model.to(device)
model.eval()

# ------------- Add a test prompt to check over-fitting while training---------

test_prompt = 'Formation of Granulation Tissue'
prompt_encoded = tokenizer(test_prompt, truncation=True, padding=False,
                           return_tensors="pt")
test_output = model.generate(input_ids=prompt_encoded.input_ids.to(device),
                             max_length=50,
                             num_return_sequences=1)  # todo check the params here
test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
log.info(f"Over-fit check answer before training: {test_answer}")

# -----------Initialize optimizer and learning rate----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------ Set up the training parameters----------------------------------

num_train_epochs = 100  # 100 increase this if data is small
train_batch_size = 4  # reduce this if you run out of memory
train_block_size = len_train_data-1  # this is how much text in each batch
if len_train_data > tokenizer.model_max_length:
    train_block_size = int(tokenizer.model_max_length/4)  # tokenizer.model_max_length=1024
num_train_steps = len_train_data // train_batch_size * num_train_epochs
log.info(f"len_train_data={len_train_data} train_block_size ={train_block_size}"
        +f" train_batch_size= {train_batch_size}")

test_batch_size = 4
test_block_size = train_block_size
num_test_steps = len_test_data // test_batch_size * num_train_epochs
log.info(f"len_test_data={len_train_data} test_block_size ={test_block_size}"+
                    f" test_batch_size= {test_batch_size}")

# We are not using the below for now as I have not understood it
# Set the optimizer and learning rate scheduler
# num_warmup_steps = 100
# max_grad_norm = 1.0
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
# lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
#            num_train_steps)

# -----------------Start Training with saving after each epoch-----------------

for epoch in range(num_train_epochs):
    # Flip/ set the model to training
    model.train()
    log.info(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    count =0
    # For each Epoch go randomly through parts of the data based on block size
    for i in range(0, len_train_data, train_block_size):
        count += 1
        # Get data in random per batch from input
        # not all training data may not be covered in one epoch here
        x, y = get_batch(len_train_data, train_input_ids, train_attention_mask,
                         device, block_size=train_block_size,
                         batch_size=train_batch_size)
        # attention_mask given by tokenize is array of ones= [1,1,..],
        # that is attend to all tokens
        outputs = model(input_ids=x, attention_mask=y, labels=x)
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
    avg_epoch_loss = epoch_loss /count
    log.info(f"Train Epoch {epoch} complete.Avg Loss: {avg_epoch_loss}")
    # -----------Test to check the model learning------------------------------

    model.eval()  # Set model to Eval loop
    test_output = model.generate(input_ids=prompt_encoded.input_ids.to(device),
                                 max_length=100,
                                 num_return_sequences=1)
    test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
    log.info(f"Over-fit check answer: {test_answer}")

    # --------- Save current Epoch and Delete Previous Epoch-------------------
    checkpoint_dir = f"./small3-gpt2-6/{model_name}-epoch-{epoch+1}-{time_hash}"
    model.save_pretrained(checkpoint_dir)
    log.info(f"Saved Model at {checkpoint_dir}")
    try:
        checkpoint_dir = f"./small3-gpt2-6/{model_name}-epoch-{epoch}-{time_hash}"
        shutil.rmtree(checkpoint_dir)
    except:
        pass

    # -----------Calculate Validation loss-------------------------------------

    validation_loss = 0
    count = 0
    with torch.no_grad():
        for i in range(0, len_test_data, test_block_size):
            count += 1
            x, y = get_batch(len_test_data, test_input_ids, test_attention_mask, device,
                             block_size=test_block_size, batch_size=test_batch_size)
            outputs = model(input_ids=x, attention_mask=y, labels=x)
            validation_loss += outputs.loss.item()
    log.info(f"Train and Validation Epoch {epoch} complete. Averge Loss: {avg_epoch_loss} " +
                 f"Avg Validation Loss  {validation_loss/count}")


#from transformers import GPT2Tokenizer, GPT2Model

from transformers import T5Tokenizer
import torch
from utils import get_batch
#from utils import printTokenizerDetails

model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
#printTokenizerDetails(tokenizer) # model_max_length: 1024 # vocab_size: 50257

# Read the cleaned input text
# original data Canon Camera manual https://gdlp01.c-wss.com/gds/0/0300004730/02/eosrt3-eos1100d-im2-c-en.pdf
input_file_path = './data/clean1.txt'
with open(input_file_path, 'r') as f:
    input_text = f.read()
print(f"length of dataset in words: {len(input_text):,}")

encoding = tokenizer(input_text, truncation=False, padding=True,return_tensors='pt')
print(len(input_text))
print(f"encoding.input_ids.shape {encoding.input_ids.shape}")
#encoding.input_ids.shape torch.Size([1, 48735])

print(f"encoding.attention_mask.shape {encoding.attention_mask.shape}")
len_train_data = encoding.input_ids.shape[1]
print(f"len_train_data = {len_train_data}")

 # flatten the tensor from  torch.Size([1, 48735]) to  torch.Size([48735])
input_ids=encoding.input_ids.view(-1)
attention_mask=encoding.attention_mask.view(-1) # All one's attend to everything

# Note , if we give truncation as False then the token sequence length goes more than model_max_length
# Token indices sequence length is longer than the specified maximum sequence length for this
#  model (23552 > 1024). Running this sequence through the model will result in indexing errors
# However we are not running through the model; We will add it to an array and train with block_size

block_size = tokenizer.model_max_length # 1024

num_train_epochs =1
for epoch in range(num_train_epochs):
    print(f"Epoch {epoch+1} of {num_train_epochs}")
    epoch_loss = 0
    for i in range(0,len_train_data, block_size):
        # do the batch size manipulation here
        x,y= get_batch(len_train_data,input_ids,attention_mask,'cpu',block_size=200,batch_size=1)
        #x.shape=torch.Size([batch_size, 1024]) y.shape=torch.Size([batch_size, 1024])
        x_out = tokenizer.decode(x.squeeze())
        # print(f"'{x_out}'")
        # print("------------------")
        # print(f"'{y}'")
        print(f"x.shape={x.shape} y.shape={y.shape}")
        #outputs = model(input_ids=x,attention_mask=y,labels=x)
        print(f"Epoch {epoch} complete. Loss:")
    # Save the model checkpoint every 10th

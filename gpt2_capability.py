"""
  To check gpt2 capability
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Ref https://huggingface.co/gpt2


def gpt_gen(tokenizer, model, prompt,device):

    encoded_input = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt")
    # try using beam search to get better results 
    # see https://huggingface.co/blog/how-to-generate
    outputs = model.generate(input_ids = encoded_input.input_ids.to(device),max_length=50,num_return_sequences=1)

    print(100 * '-')    
    print(f"Prompt {prompt}")
    print(10 * '-') 
    print("Response ",tokenizer.decode(outputs[0],  skip_special_tokens=True))    

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    print(f"Gpu device {torch.cuda.get_device_name(deviceid)}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(tokenizer.model_max_length) # 1024
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

model = model.to(device)
prompt = "translate English to French:what do the 3 dots mean in math?"  # «Que signifient les 3 points en mathématiques?»
gpt_gen(tokenizer, model, prompt,device)

prompt = " Who invented the Telephone"  # Charles Schwab with t5-base # the Telephone with t5-large
gpt_gen(tokenizer, model, prompt,device)

prompt = " Who invented X-Ray"  
gpt_gen(tokenizer, model, prompt,device)

prompt = " What is the color of sky"  # blue with t5-base and t5-large
gpt_gen(tokenizer, model, prompt,device)

prompt = " When was Franklin D Roosevelt born"  # 1932
gpt_gen(tokenizer, model, prompt,device)

prompt = " When was Gandhi born"  # 1932
gpt_gen(tokenizer, model, prompt,device)

prompt = "I enjoy walking with"
gpt_gen(tokenizer, model, prompt,device)

# GPT-2 respose
'''
----------------------------------------------------------------------------------------------------
Prompt translate English to French:what do the 3 dots mean in math?
----------
Response  translate English to French:what do the 3 dots mean in math?

The 3 dots are the number of digits in a number.

The 3 dots are the number of letters in a number.

The 3 dots are the
----------------------------------------------------------------------------------------------------
Prompt  Who invented the Telephone
----------
Response   Who invented the Telephone?

The Telephone was invented by the inventor, William H. H. Haldeman, in 1859. It was a telephone that was used by the United States Army in World War I. The telephone was a telephone
----------------------------------------------------------------------------------------------------
Prompt  Who invented X-Ray
----------
Response   Who invented X-Ray?

The X-Ray machine was invented by Dr. John C. Wright, who was a physicist at the University of California, Berkeley. Wright was a pioneer in the development of the X-Ray machine, which
----------------------------------------------------------------------------------------------------
Prompt  What is the color of sky
----------
Response   What is the color of sky?

The color of sky is a combination of light and dark. The light is the color of the sky. The dark is the color of the sky.

The color of sky is a combination of light
----------------------------------------------------------------------------------------------------
Prompt  When was Franklin D Roosevelt born
----------
Response   When was Franklin D Roosevelt born?

The answer is that he was born on January 1, 1844, in New York City, New York. He was born on January 1, 1844, in New York City, New York.

----------------------------------------------------------------------------------------------------
Prompt  When was Gandhi born
----------
Response   When was Gandhi born?

Gandhi was born in 1867. He was the son of a farmer and a merchant. He was educated at the University of Delhi. He was a member of the Indian National Congress. He was a member
----------------------------------------------------------------------------------------------------
Prompt I enjoy walking with
----------
Response  I enjoy walking with my friends, but I'm not a big fan of the idea of having to go to the gym. I'm not a big fan of the idea of having to go to the gym. I'm not a big fan of the
'''
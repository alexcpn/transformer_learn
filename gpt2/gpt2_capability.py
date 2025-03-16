"""
  To check gpt2 capability
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# Ref https://huggingface.co/gpt2


def gpt_gen(tokenizer, model, prompt,device):

    encoded_input = tokenizer(prompt, truncation=True, padding="longest", return_tensors="pt")
    
    # try using beam search to get better results 
    # see https://huggingface.co/blog/how-to-generate
    outputs = model.generate(input_ids = encoded_input.input_ids.to(device),max_new_tokens=250)

    #print(100 * '-')    
    #print(f"Prompt {prompt}")
    print(10 * '-') 
    print("Response ",tokenizer.batch_decode(outputs,  skip_special_tokens=True))    

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    print(f"Gpu device {torch.cuda.get_device_name(deviceid)}")

model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.model_max_length) # 1024
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

model = model.to(device)
prompt = "translate English to French:what do the 3 dots mean in math?"  # «Que signifient les 3 points en mathématiques?»
gpt_gen(tokenizer, model, prompt,device)

prompt = " Who invented the Telephone"  # Charles Schwab with t5-base # the Telephone with t5-large
gpt_gen(tokenizer, model, prompt,device)

prompt = " Who invented X-Ray"  
gpt_gen(tokenizer, model, prompt,device)

prompt = " What is the color of sky"  # blue with t5-base and t5-large
gpt_gen(tokenizer, model, prompt,device)

prompt = '''context-"The discovery of such collections of organisms on post-mortem examination may lead to erroneous
 conclusions being drawn as to the cause of death.  Results of Bacterial Growth.Some organisms, such as those of tetanus and erysipelas, and certain of the pyogenic bacteria,
show little tendency to pass far beyond the point at which they gain an entrance to the body.
Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis although frequently remaining localised at the seat of inoculation, tend to pass to distant parts,
lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects.
In the human subject, multiplication in the blood-stream does not occur to any great extent.In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci 
or of streptococci may be obtained from the blood." where does bacteria multiply the most'''
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

'''
With GPT large

---------
Response  ['translate English to French:what do the 3 dots mean in math?\n\nThe answer is that the dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots are the letters of the alphabet.\n\nThe dots']
----------
Response  [" Who invented the Telephone?\n\nThe answer is: no one.\n\nThe first telephone was invented in 1879 by a man named Alexander Graham Bell. He was a young man who had been working on a project to develop a telephone system that would be more reliable and more efficient than the existing telegraph system. He was also working on a new type of telephone that would be more powerful and more reliable than the existing telegraph system.\n\nBell's invention was a telephone that could be used to send and receive messages. It was a device that could be used to send and receive messages. It was a device that could be used to send and receive messages.\n\nBell's invention was a telephone that could be used to send and receive messages. It was a device that could be used to send and receive messages.\n\nBell's invention was a telephone that could be used to send and receive messages. It was a device that could be used to send and receive messages.\n\nBell's invention was a telephone that could be used to send and receive messages. It was a device that could be used to send and receive messages.\n\nBell's invention was a telephone that could be used to send and receive messages. It was a device that"]
----------
Response  [' Who invented X-Ray Vision?\n\nThe X-Ray Vision was invented by Dr. Robert E. Lee, a scientist who worked for the U.S. Army. He was working on a new type of radar that could be used to detect enemy aircraft. He was also working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was working on a new type of radar that could be used to detect enemy aircraft. He was']
----------
Response  [' What is the color of sky?\n\nA. The color of sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is blue.\n\nQ. What is the color of the sky?\n\nA. The color of the sky is']
----------
Response  ['context-"The discovery of such collections of organisms on post-mortem examination may lead to erroneous\n conclusions being drawn as to the cause of death.  Results of Bacterial Growth.Some organisms, such as those of tetanus and erysipelas, and certain of the pyogenic bacteria,\nshow little tendency to pass far beyond the point at which they gain an entrance to the body.\nOthers, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis although frequently remaining localised at the seat of inoculation, tend to pass to distant parts,\nlodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects.\nIn the human subject, multiplication in the blood-stream does not occur to any great extent.In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci \nor of streptococci may be obtained from the blood." where does bacteria multiply the most?\n"The most important factor in the multiplication of bacteria is the presence of a suitable medium for the growth of the bacteria.\nThe most suitable medium is the blood.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the most suitable medium for the growth of bacteria.\nThe blood is the']
'''
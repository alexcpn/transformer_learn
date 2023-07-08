"""
  To check Bloom capability
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
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

model_name = "bigscience/bloom-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.model_max_length) # 1024
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

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

#
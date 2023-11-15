
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import  PeftModel
from datetime import datetime

import sys

if len(sys.argv) > 1:
    argument = sys.argv[1]
    print(f"Argument received: {argument}")
    trained_model_adapter = argument
else:
    print("No argument provided.")
    trained_model_adapter = "/home/nokiaop/alex/epoch5mistral"
    
print(f"Loading QLORA Mistral Trained Adapter files from {trained_model_adapter}")    

formatted_time = datetime.now().strftime('%d:%m:%y::%H:%M')

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# #ValueError: Cannot merge LORA layers when the model is loaded in 8-bit mode
# We cannot load in 4 bit or 8 bit and load the trained QLORA

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load the entire model on the GPU 0


device_map = {"": 0}

# Loading from a saved state_dict, 
#  nokiaop@namsrv01:~/alex$ python eval_ft_mistral_lora.py mistral-direct/epoch-99-2023-11-14\ 05\:51\:34.180042
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, 
    quantization_config=bnb_config,
    device_map=device_map
)

state_dict = torch.load(trained_model_adapter)
model.load_state_dict(state_dict)


# Uncomment below for Loading the QLora trained model
#model = AutoModelForCausalLM.from_pretrained(
#            model_name, torch_dtype=torch.bfloat16, load_in_4bit=False, device_map=device_map)
# model_to_merge = PeftModel.from_pretrained(model, trained_model_adapter) #14 GB GPU T4
# model = model_to_merge.merge_and_unload()


model.eval()

# prompt =f""" How do I start with Network As Code to increase QoS?  """
# prompt_template=f'''<s>[INST]
# {prompt}[/INST]]'''

# inputs = tokenizer(prompt_template, return_tensors="pt")
# outputs = model.generate(**inputs.to('cuda') ,max_new_tokens=2000)
# output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# #print(output)
# print("-"*80)
# outputcont = "".join(output)
# parts = outputcont.split("[/INST]", 1)
# print(parts[1])

outfilename = 'epoch5_mistral_out' + formatted_time +'.txt'
while True:
    lines = []
    print("Enter your text (Type 'E' on a new line to Enter or exit):")

    while True:
        line = input()
        if line == 'E':
            break
        lines.append(line)

    question = '\n'.join(lines)

    if not question:
        print("Exiting")
        break

    # print(f'Processing Message: {question}')
    # prompt = f"{question}"
    # prompt_template=f'''<s>[INST]{sample_prompt}[/INST]
    # {sample_response}</s>
    # [INST] In the context of Network as Code, answer: {prompt}[/INST]
    # '''
        
    print(f'Processing Message:')
    if '[INST]' in question:
        prompt_template=question
        print(question)
    else:
        prompt_template = f'<s>[INST]{question}[/INST]'
        print(prompt_template)
        
    inputs = tokenizer(prompt_template, return_tensors="pt")
    outputs = model.generate(**inputs.to('cuda') ,repetition_penalty=1.2,max_new_tokens=1000)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #print(output)
    print("-"*80)
    outputcont = "".join(output)
    parts = outputcont.split("[/INST]", 1)
    print(parts[1])
    with open(outfilename, "a", encoding='utf-8') as myfile:
        myfile.write(prompt_template)
        myfile.write(parts[1])
        myfile.write("-"*80)
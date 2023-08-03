import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,

)
from peft import  PeftModel,PeftConfig
# lets use the best model in town to check if we get any better
# for NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import pprint

pp = pprint.PrettyPrinter(width=80)

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Load the entire model on the GPU 0
device_map = {"": 0}

model_name = "meta-llama/Llama-2-7b-chat-hf"
pretrained_model_name = "./models/llam2-7b-4bit-small3-1400/results/checkpoint-1400"

from transformers import AutoTokenizer

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Load in 4 bit
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
                                             device_map=device_map)

# # Load via PEFT and merge lora layers
# config = PeftConfig.from_pretrained(pretrained_model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
# model = PeftModel.from_pretrained(model, pretrained_model_name,torch_dtype=torch.float16)
# model = model.merge_and_unload() # ValueError: Cannot merge LORA layers when the model is loaded in 8-bit mode
# model.eval()


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training



prompt = "Describe Professor Thiersch of Leipsic method of grafting"
system_message = "You are a helpful assistant. From your training data please answer"
prompt_template=f'''[INST] <<SYS>>
{system_message}
<</SYS>>

{prompt} [/INST]'''

inputs = tokenizer(prompt_template, return_tensors="pt")
outputs = model.generate(**inputs.to('cuda') ,max_new_tokens=500)
output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#Output of model trained for 100
outputcont = "".join(output)
parts = outputcont.split("[/INST]", 1)
print(prompt)
print(system_message)
pp.pprint(parts[1])
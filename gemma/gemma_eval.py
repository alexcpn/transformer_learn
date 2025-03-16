

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import  PeftModel
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("training_logs.txt")])

# === Configuration ===
model_name = "google/gemma-3-1b-pt"
device_map = {"": 0}  # Load model on GPU 0
output_dir = "./results"

# Quantization Configuration (8-bit)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load Model & Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map=device_map, torch_dtype=torch.bfloat16) 

model_to_merge = PeftModel.from_pretrained(model, output_dir) #14 GB GPU T4
model = model_to_merge.merge_and_unload()

"""# Test Model directly"""

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Ensure all model weights are in `fp16` after training
model.eval()

#<s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
#https://www.promptingguide.ai/models/mistral-7b

tokenizer.padding_side = 'right'
prompt =f" Who was Jarkell ?"

# From https://ai.google.dev/gemma/docs/core/pytorch_gemma
# Chat templates
USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn><eos>\n"

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

    # # else submit the question
    # formatted_prompt = (
    #     USER_CHAT_TEMPLATE.format(
    #         prompt=question
    #     )
    #     # + MODEL_CHAT_TEMPLATE.format(prompt='California.')
    #     # + USER_CHAT_TEMPLATE.format(prompt='What can I do in California?')
    #     + '<start_of_turn>model\n'
    # )

    # print(f"Formatted prompt={formatted_prompt}")
    print("-"*80)
    inputs = tokenizer(question, return_tensors="pt")

    outputs = model.generate(**inputs.to("cuda") ,max_new_tokens=200)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"Infer output {output}")
    #Output of model trained for 100



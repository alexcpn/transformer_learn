# -*- coding: utf-8 -*-

# !huggingface-cli login

"""# Train  LLAMA 3.2 with direct data for Fine tuning (without Instruct Tuning dataset

**This only works on GPUs like A100 which has  bfloat16 (bf16="True") in the BitsandBytes Config** and need about 40 GB GPU RAM
"""

import torch
from datetime import datetime
import torch._dynamo.config
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments,AutoModelForCausalLM

time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')
print(torch.__version__)
import logging as log
import os
log_directory = './logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
outfile = log_directory + "/llama32_"+ time_hash +'.log'


log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])

####################################################################################################33

 

def create_prompt(question):
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant to help users<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        system_message = "You are a helpful assistant.Please answer the question if it is possible"
       
        prompt_template=f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        #print(prompt_template)
        return prompt_template
#--------------------------------------------------------------------------------------------------------
# Define a function to process prompts
def process_prompt(prompt, model, tokenizer, device, max_length=250):
    """Processes a prompt, generates a response, and logs the result."""
    prompt_encoded = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt")
    model.eval()
    output = model.generate(
        input_ids=prompt_encoded.input_ids.to(device),
        max_length=max_length,
        attention_mask=prompt_encoded.attention_mask.to(device)
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    log.info("-"*80)
    log.info(f"Model Question:  {prompt}")
    log.info("-"*80)
    log.info(f"Model answer:  {answer}")
    log.info("-"*80)

def freeze_all__and_unfreeze_layers(model, target_layers):
    """Freezes all layers of the model and then unfreezes specific target layers.

    Args:
        model: The model to freeze and unfreeze layers.
        target_layers: A list of layer names to unfreeze.
    """

    # Freeze all layers
    for name, param in model.named_parameters():
        param.requires_grad = False

     # Unfreeze target layers
    for name, param in model.named_parameters():
        #log.info(f"Parameter {name}")
        #if any(layer in name for layer in target_layers)
        if any(layer in name for layer in target_layers)and 'input_layernorm' in name:
            log.info(f"Unfreezing {name}")
            param.requires_grad = True

    # Verify which layers are trainable
    for name, param in model.named_parameters():
        if param.requires_grad:
            log.info(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")


            
####################################################################################################33
#               MAIN PROGRAM
####################################################################################################33
model_name = 'meta-llama'
model_name_long =  "meta-llama/Llama-3.2-1B-Instruct" #
#model_name_long =  "distilbert/distilgpt2" # for test

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
#tokenizer.pad_token = tokenizer.eos_token

train_path = './data/nsp_troubleshooting.txt' # a small training file to learn

## Loading the Model

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Going to load the model {model_name_long}")
bf16 = False
fp16 = True

major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
    bf16 = True
    fp16 = False

# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next
#device = torch.device('cuda:0')

# Load base model
if bf16:
    torch_dtype=torch.bfloat16
else:
    torch_dtype=torch.float16 # it is not possible to train in float16  #https://github.com/huggingface/transformers/issues/23165

log.info(f"Going to load the model {model_name_long} ")
# This works, this is training the qunatised model
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    device_map=device_map,

)
log.info(f"Loaded model with torch_dtype={torch_dtype}")


# Access and print the transformer layers
# print("Transformer Layers:")
# for i, layer in enumerate(model.model.layers):
#     print(f"Layer {i}: {layer}")



## Now to start the training

############################################################################################################3

# Define your prompts
prompts = [
    "NSP resynchronization with network devices is not happening, what should I do?",
    "I am not able to start  NFM-P client ",
    "NE accounts are not seen in the NFM-P client GUI",
    "Cannot select some menu options or save some configurations, what could be the reason",
    "How do I onboard a NSP service",
    "Who is Bill Gates and what is his great name to fame",
]
    
target_layers = [f'.{i}' for i in range(1, 16)] # firs 10 to 20, next , 20 to 30 # last 30 to 32
freeze_all__and_unfreeze_layers(model, target_layers)

model.train()

def load_dataset(path,tokenizer):
    dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=path,
          block_size=250)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return dataset,data_collator

train_dataset,data_collator = load_dataset(train_path,tokenizer)

#model.add_adapter(bnb_config)
model.train()
training_args = TrainingArguments(
    output_dir="./models/", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=10, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    warmup_steps=50,  # Reduced number of warmup steps for learning rate scheduler
    learning_rate=5e-4,  # Learning rate for training
    weight_decay=0.01,  # Weight decay to regularize training
    gradient_accumulation_steps=4,
    prediction_loss_only=True,
    logging_strategy="epoch",
    logging_steps=100,
    save_total_limit=1,  # do not save the model
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    #eval_dataset=test_dataset,
)
trainer.train()

# Process each prompt
model.eval()
for prompt in prompts:
    process_prompt(prompt, model, tokenizer, device)

checkpoint_dir = f"./llama3.2_trained/llama3-final"
log.info(f"Training over saving  model in {checkpoint_dir}")
model.save_pretrained(checkpoint_dir)

log.info("Training over- Exiting normally")

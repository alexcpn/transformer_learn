
import torch
from transformers import (
    AutoModelForCausalLM,
)
import logging as log
import numpy as np
from scipy.stats import norm

model_name = 'meta-llama'
model_name_long =  "meta-llama/Llama-3.2-1B-Instruct" #

print(torch.__version__)
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
torch_dtype=torch.float16 

log.info(f"Going to load the model {model_name_long} ")
# This works, this is training the qunatised model
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    #device_map=device_map,

)
#model = nn.DataParallel(model)
model = model.to(device)
log.info(f"Loaded model with torch_dtype={torch_dtype}")
#print(model)


# Freeze all attention layers
for layer in model.model.layers:  # Access each decoder layer
    # Freeze attention submodules
    for param in layer.self_attn.parameters():
        param.requires_grad = False

# Verify the parameters frozen
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Check that only non-attention layers are trainable
trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
print(f"Trainable parameters: {trainable_params}")



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

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
import torch.nn as nn
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
# lets use the best model in town to check if we get any better
# for NotImplementedError: A UTF-8 locale is required. Got ANSI_X3.4-1968
import locale
locale.getpreferredencoding = lambda: "UTF-8"
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("training_logs.txt")])

# === Configuration ===
model_name = "google/gemma-3-1b-pt"
device_map = {"": 0}  # Load model on GPU 0
output_dir = "./results"

# Training Parameters
train_params = {
    "num_train_epochs": 10,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "optim": "paged_adamw_32bit",
    "lr_scheduler_type": "constant",
    "save_steps": 1000,
    "logging_steps": 10,
    "fp16": True,
    "bf16": False,
    "gradient_checkpointing": True
}

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16, lora_dropout=0.1, r=64, bias="none",
    task_type="CAUSAL_LM", target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
)

# Load Dataset
train_dataset = load_dataset("text", data_files="decoders/gemma/gistfile1.txt")["train"]

# Quantization Configuration (8-bit)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load Model & Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map=device_map, torch_dtype=torch.float16
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Train Model
trainer = SFTTrainer(
    model=model, train_dataset=train_dataset, peft_config=peft_config,
    tokenizer=tokenizer, args=TrainingArguments(output_dir=output_dir, **train_params)
)
trainer.train()
trainer.save_model() #T4 8 GB GPU RAM

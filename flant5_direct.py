from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers
import torch
from transformers import TextDataset,DataCollatorForLanguageModeling

def load_dataset(path,tokenizer):
    dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=path,
          block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return dataset,data_collator

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/flan-t5-base"
# Load the Flan-T5 model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#freeze decoder block
num_encoder_layers = len(model.encoder.block)  
num_decoder_layers = len(model.decoder.block)  


# # Freeze upper 3 layers of encoder (lower is unfreezed)
# for i in range(num_encoder_layers-1,num_encoder_layers-4,-1):
#     for param in model.encoder.block[i].parameters():
#         param.requires_grad = False

# Freeze all layers of decoder
# for i in range(num_decoder_layers):
#     for param in model.decoder.block[i].parameters():
#         param.requires_grad = False

# OR

# freeze everything
for param in model.parameters():
     param.requires_grad = False

# Un-Freeze lower 4 layers of encoder (lower is unfreezed)
for i in range(0,num_encoder_layers-11,1):
    for param in model.encoder.block[i].parameters():
        param.requires_grad = True
 for name, param in model.named_parameters():
    print(name,param.requires_grad)


#optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  

model.to(device)
train_path ="./data/small_3.txt"
model.train()

train_dataset,data_collator = load_dataset(train_path,tokenizer)
training_args = transformers.TrainingArguments(
    output_dir="./models/flan-t5", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=8, # batch size for training
    # per_device_eval_batch_size=64,  # batch size for evaluation
    # eval_steps = 400, # Number of update steps between two evaluations.
    # save_steps=800, # after # steps model is saved 
    #warmup_steps=200,# number of warmup steps for learning rate scheduler
    # prediction_loss_only=True,
    #optim='adamw_torch'
    )

# Create a Trainer instance
trainer = transformers.Trainer(
  model=model,
  data_collator=data_collator,
  train_dataset=train_dataset,
  args=training_args,
)

# Fine-tune the model
trainer.train()
trainer.save_model()
"""
# Adapted from https://colab.research.google.com/drive/16VQg4EuMdoYxMzbU85L-S_DPTxbtCTFc#scrollTo=wpwb0BeyDWo8
"""
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments,AutoModelForSeq2SeqLM
#The class `AutoModelWithLMHead` is deprecated and will be removed in a future version.
#  Please use `AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for
#  masked language models and `AutoModelForSeq2SeqLM` for encoder-decoder models.

def load_dataset(path,tokenizer):
    dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=path,
          block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return dataset,data_collator

train_path = './data/small_3.txt'
model_name = "google/flan-t5-base"
model_name = "deepset/bert-base-cased-squad2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset,data_collator = load_dataset(train_path,tokenizer)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Freeze all the layers of the T5 model
# for param in model.encoder.block.parameters():
#     param.requires_grad = False

# #Unfreeze the bottom n layer
# n =8 # last n layers  for t5-model
# for i, m in enumerate(model.encoder.block):        
#     #Only un-freeze the last n transformer blocks
#     if i < n:
#         for parameter in m.parameters():
#             parameter.requires_grad = True 
#         print(f"Un-freezed layer {i} for training")

training_args = TrainingArguments(
    output_dir="./models/flan-t5-base-direct/", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs
    per_device_train_batch_size=3, # batch size for training
    per_device_eval_batch_size=3,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=1000, # after # steps model is saved 
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    #eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model()
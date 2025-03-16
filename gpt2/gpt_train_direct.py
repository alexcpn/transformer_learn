from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments,AutoModelForCausalLM


train_path ="./data/small_3.txt"

def load_dataset(path,tokenizer):
    dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=path,
          block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return dataset,data_collator

tokenizer = AutoTokenizer.from_pretrained("gpt2")
train_dataset,data_collator = load_dataset(train_path,tokenizer)


model = AutoModelForCausalLM.from_pretrained("gpt2")


training_args = TrainingArguments(
    output_dir="../models/gpt2-small3-v2", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=100, # number of training epochs
    per_device_train_batch_size=12, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved 
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


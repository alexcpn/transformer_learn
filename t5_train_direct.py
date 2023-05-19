from transformers import T5ForConditionalGeneration, T5Tokenizer,\
     T5Config, TextDataset, DataCollatorForSeq2Seq,\
      Seq2SeqTrainingArguments, Seq2SeqTrainer,Trainer

# Load pretrained T5 model and tokenizer
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

train_path ="./data/small_3.txt"


# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_steps=100,
    logging_dir='./logs',
)

train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
    
# Create data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./t5_base_direct')

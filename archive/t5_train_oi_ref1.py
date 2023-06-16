import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

# Initialize the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base', output_hidden_states=True)
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load and preprocess the training data
train_data = [...] # Your training data
train_encodings = tokenizer.batch_encode_plus(train_data, padding=True, truncation=True)

# Set up the training parameters
train_batch_size = 8
num_train_epochs = 10
num_warmup_steps = 100
max_grad_norm = 1.0

# Set the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-4)
num_train_steps = len(train_data) // train_batch_size * num_train_epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

# Train the model
for epoch in range(num_train_epochs):
    print(f"Epoch {epoch+1} of {num_train_epochs}")
    model.train()
    epoch_loss = 0
    for i in range(0, len(train_data), train_batch_size):
        batch_encodings = {k: [v[j] for j in range(i, min(i+train_batch_size, len(train_data)))] for k, v in train_encodings.items()}
        inputs = {k: torch.tensor(v) for k, v in batch_encodings.items()}
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['input_ids'])
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    print(f"Epoch loss: {epoch_loss}")

    # Save the model checkpoint
    checkpoint_dir = f"t5-base-finetuned-epoch-{epoch+1}"
    model.save_pretrained(checkpoint_dir)

# Generate text using the fine-tuned model
model.eval()
prompt = "What is the meaning of life?"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
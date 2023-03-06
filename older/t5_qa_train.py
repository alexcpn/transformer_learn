from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Define the training data
train_data = [
    "Answer the following question: What is the capital of France?",
    "Answer the following question: Who invented the telephone?",
    "Answer the following question: What is the largest planet in our solar system?",
    "Answer the following question: Who wrote the novel 'To Kill a Mockingbird'?",
    "Answer the following question: What is the highest mountain in the world?",
    "Answer the following question: Who discovered penicillin?",
]

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

model.train()
for epoch in range(5):
    for question in train_data:
        input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} complete. Loss: {loss.item()}")

model.eval()
# Use the fine-tuned model to answer a question
question = "Who invented Telephone?"
prompt = f"question: {question}"
inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
outputs = model.generate(input_ids=inputs, max_length=132, num_beams=4, early_stopping=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)  # Output: "Rome"

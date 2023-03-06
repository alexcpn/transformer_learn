from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the saved checkpoint directory
checkpoint_dir ='./training/gpt2-epoch-50-7:54-4-6'
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2',model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token

# Use the fine-tuned model to answer a question
question = "If you shoot at night when "
prompt = f"{question}"
encoding = tokenizer(prompt, return_tensors='pt').to(device)
print(encoding['input_ids'])
print(encoding['attention_mask'])
outputs = model.generate(input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            max_length=132, num_beams=4, early_stopping=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)  # Output: "Rome"

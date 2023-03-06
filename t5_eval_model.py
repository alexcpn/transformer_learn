from transformers import T5ForConditionalGeneration ,T5Tokenizer
import torch

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the saved checkpoint directory
checkpoint_dir ='./training/t5-small-epoch-5'
model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
model.to(device)
model.eval()

tokenizer = T5Tokenizer.from_pretrained('t5-small',model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token

# Use the fine-tuned model to answer a question
question = "clean the equipment"
prompt = f"{question}"
inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
outputs = model.generate(input_ids=inputs.to(device), max_length=132, num_beams=4, early_stopping=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)  # Output: "Rome"

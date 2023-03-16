"""
Evaluate the trained model
"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import logging as log
from datetime import datetime

outfile = "./training/output_" + str(datetime.now()) +".log"
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log.FileHandler(outfile)
        #log.StreamHandler() # only to file
    ]
)

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the saved checkpoint directory
checkpoint_dir ='./training/gpt2-epoch-50-7:54-4-6'
checkpoint_dir ='./training/gpt2-epoch-50 2023-03-07 11:13:46.62'  # # Unfreeze 6: Loss: 0.06946726143360138
#checkpoint_dir ='./training/gpt2-epoch-200-2023-03-08 20:00:21.576489' # Unfreeze 4: Loss: 0.0677826926112175
#checkpoint_dir ='./training/gpt2-epoch-1000-2023-03-08 22:19:56.069333'# tain all layers: Loss: 0.040
checkpoint_dir ='./test-gpt2/gpt2-epoch-50-2023-03-15 18:28:40.386832' #.74
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',model_max_length=1024)
#tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(checkpoint_dir,pad_token_id=tokenizer.eos_token_id)
model.to(device)
model.eval()


# Use the fine-tuned model to answer a question
#question = "How to shoot at night?"
log.info(f"Model name {checkpoint_dir}")
print(f"Model name {checkpoint_dir}")
from termcolor import  cprint
while True:
      cprint('\nPlease ask the question or press q to quit', 'green', attrs=['blink'])
      question = input()
      if 'q' == question:
            print("Exiting")
            break
      print(f'Processing Message from input()')
      prompt = f"{ question}"
      #encoding = tokenizer(prompt, return_tensors='pt').to(device)
      # print(encoding['input_ids'])
      # print(encoding['attention_mask'])
      # outputs = model.generate(input_ids=encoding['input_ids'],
      #             attention_mask=encoding['attention_mask'],
      #             max_length=132, num_beams=4, early_stopping=True)
      # making same as capability 
      encoded_input = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt")
      outputs = model.generate(input_ids = encoded_input.input_ids.to(device),max_length=250,
                              num_return_sequences=1)

      answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
      cprint(f"Question: {question}\n", 'green')
      cprint(f"Generated\n", 'cyan') 
      cprint(f"{answer}\n", 'blue') 
      log.info(f"Question: {question}")
      log.info(f"Generated: {answer}") 
      



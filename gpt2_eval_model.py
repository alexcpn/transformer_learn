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
checkpoint_dir ='./test-gpt2-2/gpt2-epoch-100-2023-03-22 22:25:27.449184' #.74 - base model
checkpoint_dir ='test-gpt2-3/gpt2-epoch-100-2023-03-22 22:25:27.449184-epoch-50-2023-03-28 13:21:49.496735'# .02 tuned on qa
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
      outputs = model.generate(input_ids = encoded_input.input_ids.to(device),
                                 min_new_tokens=200,max_new_tokens=250, 
                                 #num_beams=1,# 1 means no beam search
                                 #early_stopping=True,
                                 #num_beam_groups=1, #1 default
                                 #temperature=1, # 1 default
                                 no_repeat_ngram_size=1)

      answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
      cprint(f"Question: {question}\n", 'green')
      cprint(f"Generated\n", 'cyan') 
      cprint(f"{answer}\n", 'blue') 
      log.info(f"Question: {question}")
      log.info(f"Generated: {answer}") 
      



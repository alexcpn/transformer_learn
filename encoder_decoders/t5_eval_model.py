"""
Evaluate the trained model
"""
from transformers import AutoModelForSeq2SeqLM ,T5Tokenizer
import torch
import logging as log
from datetime import datetime

outfile = "./logs/output_" + str(datetime.now()) +".log"
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
#checkpoint_dir = 'models/flan-t5-base-direct/checkpoint-1500' trained with regulare training code
checkpoint_dir = './models/flan-t5b-nd/google/flan-t5-base-epoch-11-2023-05-23 17:01:30.339737'
checkpoint_dir ='./models/test/flan_t5_unit'
model_name = 'google/flan-t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name,model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir,pad_token_id=tokenizer.eos_token_id)
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
      #outputs = model.generate(input_ids=inputs.to(device), max_length=132, num_beams=4, early_stopping=True)
      #outputs = model.generate(input_ids = encoded_input.input_ids.to(device),
      #                                min_new_tokens=100,max_new_tokens=250, 
      #                                #num_beams=1,# 1 means no beam search
      #                                #early_stopping=True,
      #                                #num_beam_groups=1, #1 default
      #                                #temperature=1, # 1 default
      #                                no_repeat_ngram_size=1)
      outputs = model.generate(**encoded_input.to(device),max_new_tokens=250)
      answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
      cprint(f"Question: {question}\n", 'green')
      cprint(f"Generated\n", 'cyan') 
      cprint(f"{answer}\n", 'blue') 
      log.info(f"Question: {question}")
      log.info(f"Generated: {answer}") 
      



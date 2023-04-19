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
checkpoint_dir ='./test-gpt2-4/gpt2-epoch-50-2023-04-18 20:32:35.343858'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2',model_max_length=1024,padding_side='left')
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # note we have used this while training
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir,pad_token_id=50257)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
model.eval()

#  To get the token id
# answer = tokenizer.decode(torch.tensor([50256]), skip_special_tokens=False)
# print(f"answer 50256 {answer}") 
# #answer 50256 <|endoftext|>
# encoded_input = tokenizer(answer, truncation=True, padding=False, return_tensors="pt")
# print(f"encoded_input {encoded_input}")
# #encoded_input {'input_ids': tensor([[50256]]), 'attention_mask': tensor([[1]])}
# encoded_input = tokenizer('[PAD]', truncation=True, padding=False, return_tensors="pt") - works when tokenizer
# is intialised with special tokens
# encoded_input {'input_ids': tensor([[50257]]), 'attention_mask': tensor([[1]])}

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
      prompt = f"{question}"
      encoded_input = tokenizer(prompt,truncation=True,padding=True, return_tensors='pt').to(device)
      outputs = model.generate(input_ids = encoded_input.input_ids.to(device),
                        attention_mask = encoded_input.attention_mask.to(device),
                                 min_new_tokens=200,
                                 max_new_tokens=250,
                                 #num_beams=1,# 1 means no beam search
                                 #early_stopping=True,
                                 #num_beam_groups=1, #1 default
                                 #temperature=1, # 1 default
                                 repetition_penalty=1.5,
                                 no_repeat_ngram_size=1)

      answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
      cprint(f"Question: {question}\n", 'green')
      cprint(f"Generated\n", 'cyan') 
      cprint(f"{answer}\n", 'blue') 
      log.info(f"Question: {question}")
      log.info(f"Generated: {answer}") 
      # the below is better 
      test_output = model.generate(input_ids=encoded_input.input_ids.to(device), 
                    attention_mask = encoded_input.attention_mask.to(device),
                    max_length=250,
                    num_return_sequences=1)
      test_answer = tokenizer.decode(test_output[0], skip_special_tokens=True)
      log.info(f"Generated 2: {test_answer}")
      cprint(f"Generated 2\n", 'cyan') 
      cprint(f"{test_answer}\n", 'blue') 

      



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
checkpoint_dir ='./training/gpt2-epoch-50 2023-03-07 11:13:46.62'  #Loss: 0.06946726143360138
#checkpoint_dir ='./training/gpt2-epoch-100' #Epoch 49 complete. Loss: 0.07775882631540298

tokenizer = GPT2Tokenizer.from_pretrained('gpt2',model_max_length=1024)
tokenizer.pad_token = tokenizer.eos_token

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
      prompt = f"{question}"
      encoding = tokenizer(prompt, return_tensors='pt').to(device)
      # print(encoding['input_ids'])
      # print(encoding['attention_mask'])
      outputs = model.generate(input_ids=encoding['input_ids'],
                  attention_mask=encoding['attention_mask'],
                  max_length=132, num_beams=4, early_stopping=True)
      answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
      cprint(f"Question: {question}\n", 'green')
      cprint(f"Generated\n", 'cyan') 
      cprint(f"{answer}\n", 'blue') 
      log.info(f"Question: {question}")
      log.info(f"Generated: {answer}") 
      



''' 
If you shoot at night when 
generated
--------------------
If you shoot at night when  is dark the shutter speed becomes slow automatically
 shooting so that both the subject and background are properly exposed
  If you do not want slow shutter speed to be set Flash sync speed in Av mode in
   Custom Functions to or The flash pops up by itself In the and modes the flash will 
   automatically when flash is necessary The flash does not fire If you shoot continuously 
   with the flash at short intervals the flash might stop operating to protect the flash unit
    can not set flash exposure compensation with External flash func setting If flash exposure
     compensation has been set on an external Speedlite Flash exp comp can not be set in the]
      External flash func setting screen Also if you set
orginal
--------
If you shoot at night when the background is dark the shutter speed
becomes slow automatically shooting so that both the
subject and background are properly exposed If you do not want
slow shutter speed to be set set Flash sync speed in Av mode
in Custom Functions to or
The flash pops up by itself
In the and modes the flash will
automatically when flash is necessary
The flash does not fire
If you shoot continuously with the flash at short intervals the
flash might stop operating to protect the flash unit
can not set flash exposure compensation with External
flash func setting
If flash exposure compensation has been set on an external Speedlite
Flash exp comp can not be set in the External flash func
---------------
How to shoot at night?

How to shoot at night? The Night Portrait mode blurs the background to make the human subject stand out It also makes skin tones and the hair look softer than with the Full Auto mode Shooting Tips The further the distance between the subject and background the better The further the distance between the subject and background the more blurred the background will look The subject will also stand out better in front of plain dark background Use telephoto lens If you have zoom lens use the telephoto end to fill the frame with the subject from the waist up Move in closer if necessary Focus the face Check that the AF point covering the face flashes in red If you hold down the shutter button you
ale

with 50 iteration and clean2

How to shoot at night? The Night Portrait mode blurs the background to make the human subject stand out It also makes skin tones and the hair look softer than with the Full Auto mode Shooting Tips The further the distance between the subject and background the better The further the distance between the subject and background the more blurred the background will look The subject will also stand out better in front of plain dark background Use telephoto lens If you have zoom lens use the telephoto end to fill the frame with the subject from the waist up Move in closer if necessary Focus the face Check that the AF point covering the face flashes in red If you hold down the shutter button you

with 100 iterations and clean2

How to shoot at night? When the background is dark the shutter speed becomes slow automatically shooting so that both the subject and background are properly exposed If you do not want slow shutter speed to be set set Flash sync speed in Av mode in Custom Functions to or The flash pops up by itself In the and modes the flash will automatically when flash is necessary The flash does not fire If you shoot continuously with the flash at short intervals the flash might stop operating to protect the flash unit can not set flash exposure compensation with External flash func setting If flash exposure compensation has been set on an external Speedlite Flash exp comp can not be set in the External flash func setting screen Also

'''

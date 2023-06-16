"""
  To test T5 capability
"""
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
def t5_qa(tokenizer, model, prompt,device):
    encoded_input = tokenizer(prompt, max_length=1024,
                                 truncation=True, return_tensors="pt")
    # print("input_ids",input_ids)
    outputs = model.generate(input_ids = encoded_input.input_ids.to(device),max_length=500, no_repeat_ngram_size=1)
                            # attention_mask = encoded_input.attention_mask.to(device))
    print(100 * '-')    
    print(f"Prompt {prompt}")
    print(10 * '-') 
    print("Response ",tokenizer.decode(outputs[0],  skip_special_tokens=True))

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    print(f"Gpu device {torch.cuda.get_device_name(deviceid)}")

tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("t5-base")

model = model.to(device)
prompt = "translate English to French:'what do the 3 dots mean in math?'"  # «Que signifient les 3 points en mathématiques?»
t5_qa(tokenizer, model, prompt,device)

prompt = "question: 'Who invented the Telephone'"  # Charles Schwab with t5-base # the Telephone with t5-large
t5_qa(tokenizer, model, prompt,device)

prompt = "question: 'Who invented X-Ray'"  
t5_qa(tokenizer, model, prompt,device)

prompt = "question: 'What is the color of sky'"  # blue with t5-base and t5-large
t5_qa(tokenizer, model, prompt,device)

prompt = "question: 'When was Franklin D Roosevelt born'"  # 1932
t5_qa(tokenizer, model, prompt,device)

prompt = "question: 'When was Gandhi born'"  # 1932
t5_qa(tokenizer, model, prompt,device)
# T5 base is trained also on MulitRC
# So using a training example https://cogcomp.seas.upenn.edu/multirc/explore/train/category-wiki_articles.html

context = '''As Philip marched south, his opponents blocked him near Chaeronea,\
     Boeotia. During the ensuing Battle of Chaeronea, Philip commanded the right\
     wing and Alexander the left, accompanied by a group of Philip's trusted generals.\
     According to the ancient sources, the two sides fought bitterly for some time.\
     Philip deliberately commanded his troops to retreat, counting on the untested\
     Athenian hoplites to follow, thus breaking their line. Alexander was the first to\
     break the Theban lines, followed by Philip's generals. Having damaged the enemy's cohesion,\
     Philip ordered his troops to press forward and quickly routed them. With the Athenians lost,\
     the Thebans were surrounded. Left to fight alone, they were defeated.After the victory at\
     Chaeronea, Philip and Alexander marched unopposed into the Peloponnese, \
     welcomed by all cities; however, when they reached Sparta, they were refused, \
     but did not resort to war.[35] At Corinth, Philip established a "Hellenic Alliance" \
     (modelled on the old anti-Persian alliance of the Greco-Persian Wars), which included \
    most Greek city-states except Sparta. Philip was then named Hegemon \
    (often translated as "Supreme Commander") of this league (known by modern scholars\
     as the League of Corinth), and announced his plans to attack the Persian Empire.[36][37]
'''
question = 'Why was Sparta not part of the "Hellenic Alliance"?'
choices = ['It did not welcome Philip','They did not love in Hellen',
 'They were not invited']
answer_idx = 0
prompt = f"question: {question} context: {context}"
t5_qa(tokenizer, model, prompt,device) # refused  #todo check which prompt is used

prompt = 'I enjoy walking with'
t5_qa(tokenizer, model, prompt,device)

''' t5-large
----------------------------------------------------------------------------------------------------
Prompt:translate English to French:'what do the 3 dots mean in math?'
«Que signifient les trois points en mathématiques?»
----------------------------------------------------------------------------------------------------
Prompt:question: 'Who invented the Telephone'
'Who
----------------------------------------------------------------------------------------------------
Prompt:question: 'Who invented X-Ray'
'Who invented X-Ray'
----------------------------------------------------------------------------------------------------
Prompt:question: 'What is the color of sky'
blue
----------------------------------------------------------------------------------------------------
Prompt:question: 'When was Franklin D Roosevelt born'
1932
----------------------------------------------------------------------------------------------------
Prompt:question: 'When was Gandhi born'
when
----------------------------------------------------------------------------------------------------
Prompt:question: Why was Sparta not part of the "Hellenic Alliance"? context: As Philip marched south, his opponents blocked him near Chaeronea,     Boeotia. During the ensuing Battle of Chaeronea, Philip commanded the right     wing and Alexander the left, accompanied by a group of Philip's trusted generals.     According to the ancient sources, the two sides fought bitterly for some time.     Philip deliberately commanded his troops to retreat, counting on the untested     Athenian hoplites to follow, thus breaking their line. Alexander was the first to     break the Theban lines, followed by Philip's generals. Having damaged the enemy's cohesion,     Philip ordered his troops to press forward and quickly routed them. With the Athenians lost,     the Thebans were surrounded. Left to fight alone, they were defeated.After the victory at     Chaeronea, Philip and Alexander marched unopposed into the Peloponnese,      welcomed by all cities; however, when they reached Sparta, they were refused,      but did not resort to war.[35] At Corinth, Philip established a "Hellenic Alliance"      (modelled on the old anti-Persian alliance of the Greco-Persian Wars), which included     most Greek city-states except Sparta. Philip was then named Hegemon     (often translated as "Supreme Commander") of this league (known by modern scholars     as the League of Corinth), and announced his plans to attack the Persian Empire.[36][37]

they were refused
----------------------------------------------------------------------------------------------------
Prompt:I enjoy walking with
you
'''
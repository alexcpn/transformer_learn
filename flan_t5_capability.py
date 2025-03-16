from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoConfig
import torch

# Fine-tune the model on the training data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_name = "./models/flan-t5-2/google/flan-t5-base-epoch-45-2023-05-30 18:19:52.327485" # encoder+decoder
model_name = "./models/flan-t5-2/google/flan-t5-base-epoch-100-2023-05-31 20:25:06.073126"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,device_map="auto")#, torch_dtype=torch.float16)
config = AutoConfig.from_pretrained(model_name)
#print("Maximum sequence length this model can handle: ", config)
model.eval()
model.to(device)


para = '''The blue whale is the largest animal, but the 
     largest animal on land is the Elephant and after that the Giraffe'''

# inputs = tokenizer(f"From the paragraph {para}:,infer what is the largest " +
#     "animal on land after the Elephant", return_tensors="pt")
# outputs = model.generate(**inputs,max_new_tokens=100)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True)) #['Giraffe']

# inputs = tokenizer(f"From the paragraph {para}:,infer what is the largest " +
#     "animal on land", return_tensors="pt")
# outputs = model.generate(**inputs,max_new_tokens=100)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))#['Elephant']

# inputs = tokenizer(f"From the paragraph {para}:,infer what is the largest " +
#     "animal", return_tensors="pt")
# outputs = model.generate(**inputs.to(device),max_new_tokens=100)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))#['The blue whale']

para ='''The discovery of such collections of organisms on post-mortem examination may lead to erroneous
 conclusions being drawn as to the cause of death.  Results of Bacterial Growth.
Some organisms, such as those of tetanus and erysipelas, and certain of the pyogenic bacteria,
show little tendency to pass far beyond the point at which they gain an entrance to the body
Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis 
although frequently remaining localised at the seat of inoculation, tend to pass to distant parts,
lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects.
In the human subject, multiplication in the blood-stream does not occur to any great extent.
In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci 
or of streptococci may be obtained from the blood.'''
infer_list =[]
infer = 'summarize' #['Observe the presence of a large number of organisms in the blood-stream.']
infer_list.append(infer)
infer = 'where do tubercle bacillus go' #'capillaries of joints, bones, kidney, or lungs']
infer_list.append(infer)
infer = 'name some acute pyogenic infections'#['osteomyelitis, cellulitis, etc'] # large ['Bacteria in the Blood']
infer_list.append(infer)
infer = 'a title to this' #['Bacteria'] # flan-t5-large ['Bacteria in the Blood']
infer_list.append(infer)
infer = 'what is the difference between tetanus and tubercle bacillus organism'
infer_list.append(infer)
#['show little tendency to pass far beyond the point at which they gain an entrance to the body
#  Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis 
# although frequently remaining localised at the seat of inoculation, tend to pass to distant parts,
#  lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious
#  effects']
infer = 'where does bacteria multiply the least' # blood stream
infer_list.append(infer)

for infer in infer_list:
     print("--------Inference-------",infer)
     inputs = tokenizer(f"Given the context {para}: {infer}", return_tensors="pt")
     outputs = model.generate(**inputs.to(device),max_new_tokens=100)
     print("----- Context based-------------------")
     print(tokenizer.batch_decode(outputs, skip_special_tokens=True)) 
     inputs = tokenizer(f"{infer}", return_tensors="pt")
     outputs = model.generate(**inputs.to(device),max_new_tokens=100)
     print("----- Training data based-------------------")
     print(tokenizer.batch_decode(outputs, skip_special_tokens=True)) 

# training based, encoder -decoder 4 bottom layers unfreezed and trained [t5_train_model.py](t5_train_model.py)

# --------Inference------- summarize
# ----- Context based-------------------
# [', and certain of the pyogenic bacteria, show little tendency to pass far beyond the point at which they gain an entrance to the body Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis although frequently remaining localised at the seat of inoculation, tend to pass to distant parts, lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects']
# ----- Training data based-------------------
# ['- The symbiotic relationship between the symbiotic relationship and the symbiotic relationship is a key factor in the success of the symbiotic relationship.']
# --------Inference------- where do tubercle bacillus go
# ----- Context based-------------------
# ['inoculation, tend to pass to distant parts, lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects. In the human subject, multiplication in the blood-stream does not occur to any great extent. In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci or of streptococci may be obtained']
# ----- Training data based-------------------
# ['tubercle bacillus goes to the lungs']
# --------Inference------- name some acute pyogenic infections
# ----- Context based-------------------
# [', and certain of the pyogenic bacteria, show little tendency to pass far beyond the point at which they gain an entrance to the body Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis although frequently remaining localised at the seat of inoculation, tend to pass to distant parts, lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects']
# ----- Training data based-------------------
# ['pyogenic infections']
# --------Inference------- a title to this
# ----- Context based-------------------
# ['Bacterial Growth. Some organisms, such as those of tetanus and erysipelas, and certain of the pyogenic bacteria, show little tendency to pass far beyond the point at which they gain an entrance to the body Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis although frequently remaining localised at the seat of inoculation, tend to pass to']
# ----- Training data based-------------------
# ["'Still a Good Day'"]
# --------Inference------- what is the difference between tetanus and tubercle bacillus organism
# ----- Context based-------------------
# [', and certain of the pyogenic bacteria, show little tendency to pass far beyond the point at which they gain an entrance to the body Others, on the contrary for example, the tubercle bacillus and the organism of acute osteomyelitis although frequently remaining localised at the seat of inoculation, tend to pass to distant parts, lodging in the capillaries of joints, bones, kidney, or lungs, and there producing their deleterious effects']
# ----- Training data based-------------------
# ['tubercle bacillus organism']
# --------Inference------- where does bacteria multiply the least
# ----- Context based-------------------
# ['in the blood-stream does not occur to any great extent. In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci or of streptococci may be obtained from the blood. In some general acute pyogenic infections, such as osteomyelitis, cellulitis, etc., pure cultures of staphylococci or of streptococci may be']
# ----- Training data based-------------------
# ['in the lungs']
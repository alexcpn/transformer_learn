import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import logging as log
import os
from datetime import datetime
import re

time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')


# Function to vectorize the user query
def vectorize_query(query,model):
    return model.encode(query)

# Function to search for the most relevant summary
def search_summary(query,sentence_model):
    # Vectorize the user question
    query_vector = vectorize_query(query,sentence_model)
    
    # Perform similarity search using LanceDB and specify the vector column
    results = table.search(query_vector,vector_column_name="vectorS").limit(3).to_list()
    return results


####################################################################################################33

def create_prompt(question,system_message=None):
        """
        Create Prompt as per LLAMA3.2 p
        """
        if not system_message:
            system_message = "You are a helpful assistant"
        prompt_template=f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {question}<|eot_id|><|start_header_id|>assistant1231231222<|end_header_id|>
        '''
        #print(prompt_template)
        return prompt_template

#--------------------------------------------------------------------------------------------------------
# Define a function to process prompts
#--------------------------------------------------------------------------------------------------------

def process_prompt(prompt, model, tokenizer, device, temperature=0.1,max_length=500):
    """Processes a prompt, generates a response, and logs the result."""
    prompt_encoded = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt")
    model.eval()
    output = model.generate(
        input_ids=prompt_encoded.input_ids.to(device),
        max_new_tokens=max_length,
        attention_mask=prompt_encoded.attention_mask.to(device),
        temperature=temperature # More determnistic
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
        # Split the sentence at the word "assistant"
    parts = answer.split("assistant1231231222", 1)

    # Check if "assistant" is in the sentence and get the words after it
    if len(parts) > 1:
        words_after_assistant = parts[1].strip()
        return  words_after_assistant
    else:
        print("The word 'assistant' was not found in the sentence.")
        return "NONE"

####################################################################################################33
#               MAIN PROGRAM
####################################################################################################33
log_directory = './logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
outfile = log_directory + "/llama32_query"+ time_hash +'.log'

log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.INFO, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])
# Connect to LanceDB (assuming you have already created the database and table)
db = lancedb.connect('./data/my_lancedb')
table = db.open_table('summaries')

# Load the same pre-trained transformer model for vectorization
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Intialize the model
model_name = 'meta-llama'
model_name_long =  "meta-llama/Llama-3.2-1B-Instruct" #

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
#tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Going to load the model {model_name_long}")
bf16 = False
fp16 = True

major, _ = torch.cuda.get_device_capability()
if major >= 8:
    print("=" * 80)
    log.info("Your GPU supports bfloat16: accelerate training with bf16=True")
    print("=" * 80)
    bf16 = True
    fp16 = False

# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next
#device = torch.device('cuda:0')

# Load base model
if bf16:
    torch_dtype=torch.bfloat16
else:
    torch_dtype=torch.float16 

log.info(f"Going to load the model {model_name_long} ")

model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    device_map=device_map,

)

log.info(f"Loaded model with torch_dtype={torch_dtype}")



# Example usage
user_question = "Not able to manage new devices"
results = search_summary(user_question,sentence_model)
summary_list=[]
if not results:
   log.error("No relevant summary found.")
   exit(-1)

for idx, result in enumerate(results):
    # print(f"Result {idx + 1}:")
    # print(f"Page Number: {result['page_number']}")
    # print(f"Summary: {result['summary']}")
    summary_list.append(str(result['page_number'])+"# " +result['summary'])
    # print(f"Keywords: {result['keywords']}")
    # print(f"Original Text: {result['original_content']}")
    # print('-' * 80)
        
question = f"""From the given list of summary [{str(summary_list)}] rank which summary would possibly have \
the possible answer to the question {user_question}. Return only that summary from the list"""
log.info(question)
prompt = create_prompt(question,"You are a helpful assistant")
response = process_prompt(prompt, model, tokenizer, device,temperature=0.1)
log.info(f"Selected Summary '{response}'")


# Regular expression to find the page number followed by #
match = re.search(r'(\d+)#', response)

if not match:
    log.error("Not able to extract the page_numbers")
    exit(0)

# Extract the matched numbers
page_number = match.group(1)
log.info(f"Page number: {page_number}")
for idx, result in enumerate(results):
    if int(page_number) == result['page_number']:
        page =result['original_content']
        question = f"""Can you answer the question or query or provide more deatils query:'{user_question}' \
        Using the context below
        context:'{page}'
        """
        log.info(question)
        prompt = create_prompt(question,"You are a helpful assistant that will go through the given query and given context and think in steps and then try to answer the query \
                               with the information in the context")
        response = process_prompt(prompt, model, tokenizer, device,temperature=0.01) #less freedom to hallicunate
        log.info(response)
  
        
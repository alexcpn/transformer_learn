import pandas as pd
import fitz  # PyMuPDF
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import logging as log
import os
from datetime import datetime
import lancedb
from sentence_transformers import SentenceTransformer
import json
import pyarrow as pa
import numpy as np
import re

time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')

log_directory = './logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
outfile = log_directory + "/llama32_insert"+ time_hash +'.log'

log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.ERROR, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])

####################################################################################################33

def create_prompt(question):
        """
        Create Prompt as per LLAMA3.2 p
        """
        system_message = "You are a helpful assistant for summarizing text and result in JSON format"
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

def process_prompt(prompt, model, tokenizer, device, max_length=500):
    """Processes a prompt, generates a response, and logs the result."""
    prompt_encoded = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt")
    model.eval()
    output = model.generate(
        input_ids=prompt_encoded.input_ids.to(device),
        max_new_tokens=max_length,
        attention_mask=prompt_encoded.attention_mask.to(device),
        temperature=0.1 # More determnistic
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


def extract_outermost_braces(text):
    # Use regex to find the outermost curly braces and their contents
    match = re.search(r'\{(.*)\}', text, re.DOTALL)
    if match:
        # Return the content inside the outermost braces
        return match.group(1)
    return None


####################################################################################################33
#               MAIN PROGRAM
####################################################################################################33
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

# Read the data file
file_path = './data/nsp_troubleshooting.txt'


dict_pages={}
# Open the PDF file
with fitz.open(file_path) as pdf_document:
    # Read each page and print the text
    print("\nPages Content:")
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)  # Load each page
        page_text = page.get_text()  # Extract text from the page
        dict_pages[page_number]=page_text
        #print(f"\n--- Page {page_number + 1} ---")
        print(f"processec pdf page {page_number + 1} ---")
        if page_number== 30:
            print(page_text.encode('utf-8', 'replace').decode('utf-8'))
            print("-"*80)

print(f'Generated  text summary')

# Load a pre-trained transformer model for vectorization (e.g., using sentence-transformers)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# Connect to LanceDB (creates an on-disk database at specified path)
db = lancedb.connect('./data/my_lancedb')

# Define schema using PyArrow
schema = pa.schema([
    pa.field("page_number", pa.int64()),
    pa.field("original_content", pa.string()),
    pa.field("summary", pa.string()),
    pa.field("keywords", pa.string()),
    pa.field("vectorS", pa.list_(pa.float32(),384)),#all-MiniLM-L6-v2: This model has an embedding size of 384.
    pa.field("vectorK", pa.list_(pa.float32(),384)),
])

# Create or connect to a table with the schema
table = db.create_table('summaries', schema=schema, mode='overwrite')

# Use the Small LLAMA 3.2 1B model to create summary
for page_number, text in dict_pages.items():
    question = f"""For the given passage, provide a long summary about it, incorporating all the main keywords in the passage.
    Format should be  in JSON format like below: 
    {{
        "summary": <text summary> example "Some Summray text",
        "keywords": <a comma separated list of main keywords and acronyms that appear in the passage> exmple ["keyword1","keyword2"],
    }}
    Make sure that JSON feilds have double quotes example instead of 'summary' correct is "summary" and  use the closing and ending delimiters 
    Passage: {text}"""
    prompt = create_prompt(question)
    response = process_prompt(prompt, model, tokenizer, device)
    try:
        summary_json =json.loads(response)
    except json.decoder.JSONDecodeError as e:
        exception_msg= str(e)
        question = f""" Correct the following JSON {response} which has {exception_msg} to proper JSON format.Output only JSON.
        Format should be  in JSON format like below: 
        {{
            "summary": <text summary> example "Some Summray text",
            "keywords": <a comma separated list of keywords and acronyms  that appear in the passage> exmple ["keyword1","keyword2"],
        }}"""
        log.warning(f"{exception_msg} for {response}")
        prompt = create_prompt(question)
        response = process_prompt(prompt, model, tokenizer, device)
        log.warning(f"Corrected '{response}'")
        # see if the error can be corrected
        try:
            summary_json =json.loads(response)
        except:
            log.error(f"'{exception_msg}' for '{response}' for '{text}' ")
            continue

    keywords = ''.join(summary_json['keywords'])
    # Vectorize the summary text
    vectorS = sentence_model.encode(summary_json['summary'])
    vectorK = sentence_model.encode(keywords)
    # Store the summary vector with metadata
    print(int(page_number))
    # print(summary_json['summary'])
    # print(summary_json['keywords'])
    table.add([{
        "page_number": int(page_number),
        "original_content": text,
        "summary": summary_json['summary'],
        "keywords": keywords,
        "vectorS": vectorS,  # The embedding/vector representation of the summary
        "vectorK": vectorK  # The embedding/vector representation of the summary
    }])

print("Summary stored successfully in LanceDB.")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import torch
import logging as log
from torch.nn.functional import normalize
import numpy as np

# does not fit into 8 GB RAN

log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.StreamHandler()
                ])


def create_prompt(question,system_message=None):
        """
            https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
        """
        if not system_message:
            system_message = "You are a helpful assistant.Please answer the question if it is possible" #this is the default one
       
        prompt_template=f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        #print(prompt_template)
        return prompt_template
#-------------------------------

def gpt_gen(tokenizer, model, prompt,device):

    encoded_input = tokenizer(prompt, truncation=True,  return_tensors="pt")
    outputs = model.generate(input_ids = encoded_input.input_ids.to(device),attention_mask=encoded_input.attention_mask.to(device),max_new_tokens=2000)
    output = tokenizer.batch_decode(outputs,  skip_special_tokens=True)
    formatted_output = "\n".join(output)
    log.info(10 * '-') 
    log.info(f"Response {formatted_output}") 

def get_embedding(outputs, layer_idx=-1, use_cls_token=True):
    # Check if hidden_states are available
    if outputs.hidden_states is not None:
        # Extract hidden states (the outputs will contain hidden states from all layers)
        hidden_states = outputs.hidden_states  # A tuple with hidden states from all layers
        layer_hidden_states = hidden_states[layer_idx]  
        print(f"layer_hidden_states={layer_hidden_states.shape}")
        # Choose the hidden states from the desired layer (e.g., the last layer)
        sentence_embedding = torch.mean(layer_hidden_states, dim=1)
    return sentence_embedding

def cosine_similarity(embedding1, embedding2):
    # Convert embeddings to numpy arrays
    print(f"embedding1={embedding1.shape}")
    print(f"embedding2={embedding2.shape}")
    embedding1 = embedding1.to(torch.float32).cpu().numpy()
    embedding2 = embedding2.to(torch.float32).cpu().numpy()
    # Normalize the embeddings
    embedding1_norm = embedding1 / np.linalg.norm(embedding1, axis=1, keepdims=True)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2, axis=1, keepdims=True)
    # Compute cosine similarity
    similarity = np.dot(embedding1_norm, embedding2_norm.T)
    return similarity.item()  # Extract scalar value
#################### -------------------------------------------------------------------------------#############################
#################### -------------------------------------------------------------------------------#############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")
    
bf16 = False
fp16 = True
model_name_long ="meta-llama/Llama-3.2-1B-Instruct" # loads in 6 GB GPU RAM directly


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

# Load the model configuration and set output_hidden_states=True
config = AutoConfig.from_pretrained(model_name_long)
config.output_hidden_states = True  # Enable output of hidden states

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    config=config,
    torch_dtype=torch_dtype,
    device_map=device_map,
    #quantization_config=bnb_config,
    trust_remote_code=True

)
# Prepare input
text = "I went to the bank"
inputs = tokenizer(text, return_tensors="pt")
inputs.to(device)
# Run inference to get hidden states
with torch.no_grad():
    outputs = model(**inputs)

embedding1 = get_embedding(outputs,layer_idx=-1)

text = "I swam to the bank"
inputs = tokenizer(text, return_tensors="pt")
inputs.to(device)
# Run inference to get hidden states
with torch.no_grad():
    outputs = model(**inputs)

embedding2 = get_embedding(outputs,layer_idx=-1)

# Compute cosine similarity
similarity = cosine_similarity(embedding1, embedding2)
print(f"Cosine Similarity: {similarity}")



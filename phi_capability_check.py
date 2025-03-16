from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import logging as log

# does not fit into 8 GB RAN

log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.StreamHandler()
                ])


def gpt_gen(tokenizer, model, prompt,device):

    encoded_input = tokenizer(prompt, truncation=True, padding="longest", return_tensors="pt")
    
    # try using beam search to get better results 
    # see https://huggingface.co/blog/how-to-generate
    outputs = model.generate(input_ids = encoded_input.input_ids.to(device),max_new_tokens=250)

    #print(100 * '-')    
    #print(f"Prompt {prompt}")
    log.info(10 * '-') 
    log.info("Response ",tokenizer.batch_decode(outputs,  skip_special_tokens=True)) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    deviceid = torch.cuda.current_device()
    log.info(f"Gpu device {torch.cuda.get_device_name(deviceid)}")
    
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

 # Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = torch_dtype
compute_dtype = torch_dtype
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Quantization type (fp4 or nf4)
bnb_8bit_quant_type = "nf8"

bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type=bnb_8bit_quant_type,
        bnb_8bit_compute_dtype=compute_dtype,
        bnb_8bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True,
)
model_name_long ="microsoft/phi-2" #2.7 B
# This works, this is training the qunatised model

tokenizer = AutoTokenizer.from_pretrained(model_name_long)
model = AutoModelForCausalLM.from_pretrained(
    model_name_long,
    torch_dtype=torch_dtype,
    device_map=device_map,
    quantization_config=bnb_config,
    trust_remote_code=True

)

prompt = "translate English to French:what do the 3 dots mean in math?"  # «Que signifient les 3 points en mathématiques?»
gpt_gen(tokenizer, model, prompt,device)

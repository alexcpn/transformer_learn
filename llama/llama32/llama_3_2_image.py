import requests
import torch
import sys
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor,  BitsAndBytesConfig
from accelerate import infer_auto_device_map

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Proper device setup

# Activate 4-bit precision base model loading
use_4bit = True
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
# Load tokenizer and model with QLoRA configuration

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)


# Load the entire model on the GPU 0
device_map = {"": 0} # lets load on the next
device_map = "auto"

# Load the model without weights
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
# Compute the device map
device_map = infer_auto_device_map(
    model,
    max_memory={"cpu": "12GB", "cuda:0": "5GB"},
    no_split_module_classes=["MllamaDecoderLayer"],
)
print(device_map)
sys.exit(0)
# Load the model
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map=device_map,
)
# model = model.to(device) 

# Load the processor
processor = AutoProcessor.from_pretrained(model_id)

# Load an image from the web
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Create a message list that includes an image and a text prompt
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "If I had to write a haiku for this one, it would be: "}
    ]}
]

# Prepare inputs using the processor
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(device)

# Generate output from the model
output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
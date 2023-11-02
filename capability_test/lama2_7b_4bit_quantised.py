#nvidia-smi -L
#GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU (UUID: GPU-3d4037fa-1de1-b359-c959-bfb3d9ecbe50)
# 
# $ nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Fri_Jan__6_16:45:21_PST_2023
# Cuda compilation tools, release 12.0, V12.0.140
# Build cuda_12.0.r12.0/compiler.32267302_0
# 
# $ nvidia-smi
# Wed Nov  1 17:18:56 2023       
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
# | N/A   69C    P0    26W /  80W |     56MiB /  6144MiB |      9%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
                                                                               
# +-----------------------------------------------------------------------------+
# | Processes:                                                                  |
# |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
# |        ID   ID                                                   Usage      |
# |=============================================================================|
# |    0   N/A  N/A      5319      G   /usr/lib/xorg/Xorg                 55MiB |


import torch
import traceback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
   AutoConfig,

)
from accelerate import infer_auto_device_map ,init_empty_weights

usercontext ='''

Given the Context below

"""

Cholesterol is a waxy substance found in your blood. Your body needs cholesterol to build healthy cells, but high levels of cholesterol can increase your risk of heart disease.

With high cholesterol, you can develop fatty deposits in your blood vessels. Eventually, these deposits grow, making it difficult for enough blood to flow through your arteries. Sometimes, those deposits can break suddenly and form a clot that causes a heart attack or stroke.

High cholesterol can be inherited, but it's often the result of unhealthy lifestyle choices, which make it preventable and treatable. A healthy diet, regular exercise and sometimes medication can help reduce high cholesterol.

Answer the following question if possible
What is the role of cholestrol in the body
'''
class LLAMa2Q:
    def __init__(self,model_name,q4bitA=True) -> None:

        if  not torch.cuda.is_available() :
            print("Running 4 bit quantised model is possible only on GPU")
            raise Exception("Cannot Proceed without GPU")

        ################################################################################
        # bitsandbytes parameters
        ################################################################################

        # Activate 4-bit precision base model loading
        use_4bit = True

        # Compute dtype for 4-bit base models
        bnb_4bit_compute_dtype = "float16"

        # Quantization type (fp4 or nf4)
        bnb_4bit_quant_type = "nf4"
        # Quantization type (fp4 or nf4)
        bnb_8bit_quant_type = "nf8"

        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False

        # Load the entire model on the GPU 0
        device_map = {"": 0}
        self.device = torch.device("cuda")


        model_name =model_name

        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        if q4bitA:
          bnb_config = BitsAndBytesConfig(
              load_in_4bit=True,
              bnb_4bit_quant_type=bnb_4bit_quant_type,
              bnb_4bit_compute_dtype=compute_dtype,
              bnb_4bit_use_double_quant=use_nested_quant,
          )
        else:
          # Load in 8 bit
          bnb_config = BitsAndBytesConfig(
              load_in_8bit=True,
              bnb_8bit_quant_type=bnb_8bit_quant_type,
              bnb_8bit_compute_dtype=compute_dtype,
              bnb_8bit_use_double_quant=use_nested_quant,
              llm_int8_enable_fp32_cpu_offload=True,
        )

        # # # Load in 8 bit
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,torch_dtype=torch.bfloat16,
        #                                             device_map=device_map)

        # # Load in 4 bit
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
                                                    device_map=device_map,use_auth_token=True)

        #torch.save(self.model.state_dict(), "4bit")
        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print('model_max_length:', self.tokenizer.model_max_length)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    def create_prompt(self,question,system_message):
        prompt = f"{ question}"
        if not system_message:
            system_message = "You are a helpful assistant.Please answer the question if it is possible"
        prompt_template=f'''[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {prompt} [/INST]'''
        #print(prompt_template)
        return prompt_template

    def generate_ouputput(self,prompt_template):
        inputs = self.tokenizer(prompt_template, return_tensors="pt",truncation=True,max_length=2000)
        outputs = self.model.generate(**inputs.to(self.device) ,max_new_tokens=2000)
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output

    def get_chat_response(self,output):
        outputcont = "".join(output)
        parts = outputcont.split("[/INST]", 1)
        return parts[1]


if __name__ == "__main__":
    print("Going to load the llama2 7b model ...")
    llam2_4bit = LLAMa2Q(model_name="meta-llama/Llama-2-7b-chat-hf",q4bitA=True)
    print("Loaded LLama2 7b model")

    prompt = f"{ usercontext}"
    system_message = "You are a helpful assistant.Please answer the question if it is possible from the context"
    prompt_template=f'''[INST] <<SYS>>
    {system_message}
    <</SYS>>

    {prompt} [/INST]'''

    prompt_template = llam2_4bit.create_prompt(prompt,system_message)
    output = llam2_4bit.generate_ouputput(prompt_template)
    response =llam2_4bit.get_chat_response(output)
    print(response)
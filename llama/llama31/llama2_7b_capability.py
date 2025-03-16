import torch
import traceback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
   AutoConfig,

)
from accelerate import infer_auto_device_map ,init_empty_weights
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import pprint
from termcolor import  cprint

pp = pprint.PrettyPrinter(width=80)


class LLAMa2Quantised:

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
                                                    device_map=device_map)

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

if __name__ == '__main__':

    print("Going to load the llama2 7b model ...")
    llam2_4bit = LLAMa2Quantised(model_name="meta-llama/Llama-2-7b-chat-hf",q4bitA=True)
    print("Loaded LLama2 7b model")

    while True:
        cprint('\nPlease ask the question or press q to quit', 'green', attrs=['blink'])
        question = input()
        if 'q' == question:
            print("Exiting")
            break
        print(f'Processing Message from input()')
        prompt = f"{ question}"
        system_message = "You are a helpful assistant.Please answer the question if it is possible"
        prompt_template=f'''[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {prompt} [/INST]'''

        prompt_template = llam2_4bit.create_prompt(prompt,system_message)
        output = llam2_4bit.generate_ouputput(prompt_template)
        response =llam2_4bit.get_chat_response(output)
        print(response)
      




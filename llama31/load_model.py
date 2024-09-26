import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
  
)
from datetime import datetime
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import pprint
from termcolor import  cprint

pp = pprint.PrettyPrinter(width=80)
time_hash=str(datetime.now()).strip()
time_hash = time_hash.replace(' ', '-')
print(torch.__version__)

outfile = "./logs/llama31_"+ time_hash +'.log'
from importlib import reload  # Not needed in Python 2
import logging as log
reload(log)
log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', 
                level=log.DEBUG, datefmt='%I:%M:%S',
                handlers=[
                    log.FileHandler(outfile),
                    log.StreamHandler()
                ])


class LLAMa2Quantised:

    def __init__(self,pre_trained_model_path,model_name,q4bitA=True) -> None:

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
        pre_trained_model_path =pre_trained_model_path
        model_name= model_name

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
        device = torch.device('cuda:0')

        # Load base model
        if bf16:
            torch_dtype=torch.bfloat16
        else:
            torch_dtype=torch.float16
            
        # # # Load in 8 bit
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,torch_dtype=torch.bfloat16,
        #                                             device_map=device_map)
        config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        # # Load in 4 bit

        self.model = AutoModelForCausalLM.from_pretrained(pre_trained_model_path,config=config,
                                                     torch_dtype=torch_dtype,
                                                    quantization_config=bnb_config,
                                                    device_map=device_map)
        
        #model.load_state_dict(torch.load("path_to_your_weights_file/pytorch_model.bin"))

        #torch.save(self.model.state_dict(), "4bit")
        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print('model_max_length:', self.tokenizer.model_max_length)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    def create_prompt(self,question,system_message):
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant to help users<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        if not system_message:
            system_message = "You are a helpful assistant.Please answer the question if it is possible"
       
        prompt_template=f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

        {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        '''
        #print(prompt_template)
        return prompt_template

    def generate_ouputput(self,prompt_template):
        inputs = self.tokenizer(prompt_template, return_tensors="pt",truncation=True,max_length=2000)
        outputs = self.model.generate(**inputs.to(self.device) ,max_new_tokens=2000)
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output

    def get_chat_response(self,output):
        # outputcont = "".join(output)
        # parts = outputcont.split("[/INST]", 1)
        return output

if __name__ == '__main__':
    
    model_name =  "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pretrained_model_path='/home/alex/llama3.1_trained//llama3-final/'
    log.info(f"Going to load the {pretrained_model_path} for {model_name}...")
    llam2_4bit = LLAMa2Quantised(pretrained_model_path,model_name,q4bitA=True)
    log.info("Loaded model")

    while True:
        cprint('\nPlease ask the question or press q to quit', 'green', attrs=['blink'])
        question = input()
        if 'q' == question:
            print("Exiting")
            break
        print(f'Processing Message from input()')
        prompt = f"{ question}"
        system_message = "You are a helpful assistant.Please answer the question if it is possible"
        prompt_template = llam2_4bit.create_prompt(prompt,system_message)
        output = llam2_4bit.generate_ouputput(prompt_template)
        response =llam2_4bit.get_chat_response(output)
        log.info(response)
      



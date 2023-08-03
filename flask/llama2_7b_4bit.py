import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,

)

class LLAMa2_7b_4bitQ:
    def __init__(self) -> None:

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

        # Activate nested quantization for 4-bit base models (double quantization)
        use_nested_quant = False

        # Load the entire model on the GPU 0
        device_map = {"": 0}
        self.device = torch.device("cuda")


        model_name = "meta-llama/Llama-2-7b-chat-hf"

        # Load tokenizer and model with QLoRA configuration
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

        # Check GPU compatibility with bfloat16
       
        # if compute_dtype == torch.float16 and use_4bit:
        #     major, _ = torch.cuda.get_device_capability()
        #     if major >= 8:
        #         print("=" * 80)
        #         print("Your GPU supports bfloat16: accelerate training with bf16=True")
        #         print("=" * 80)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

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
        return prompt_template

    def generate_ouputput(self,prompt_template):
        inputs = self.tokenizer(prompt_template, return_tensors="pt",truncation=True)
        outputs = self.model.generate(**inputs.to(self.device) ,max_new_tokens=2000)
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output

    def get_chat_response(self,output):
        outputcont = "".join(output)
        parts = outputcont.split("[/INST]", 1)
        print(parts[1])
        return parts[1]
        




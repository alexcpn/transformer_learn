import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
   AutoConfig,

)

from accelerate import infer_auto_device_map ,init_empty_weights

def parse_response_out (response):
  return  re.findall(pattern, response, re.DOTALL)

# def sliding_window(filename, window_size=200, overlap=50):
#     with open(filename, 'r', encoding='utf-8') as file:
#         text = file.read()
#         words = text.split()

#         step = window_size - overlap
#         chunks = []
#         for i in range(0, len(words) - window_size + 1, step):
#             chunk = ' '.join(words[i:i + window_size])
#             chunk = remove_broken_sentences(chunk)
#             chunks.append(chunk)

#         return chunks

def sliding_window(filename, window_size=200, overlap=50):
    """
        This is for text with code snippets 
    """
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chunks = []
    current_chunk = []
    in_code_block = False
    char_count = 0

    for line in lines:
        current_chunk.append(line)
        char_count += len(line)
        
        # Check if we are entering or leaving a code block
        if line.strip() == '```':
            in_code_block = not in_code_block

        # If the current chunk size is larger than window_size and not in a code block,
        # then finalize the current chunk
        if char_count >= window_size and not in_code_block:
            chunks.append(''.join(current_chunk))
            
            # Move the window: determine how many lines to keep for overlap
            # and adjust char_count accordingly
            keep_for_next_chunk = current_chunk[-overlap:]
            char_count = sum(len(line) for line in keep_for_next_chunk)
            current_chunk = keep_for_next_chunk

    # Add the last chunk in case it didn't end at the window boundary or is still in a code block
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks


def remove_broken_sentences(chunk):
    # Split the chunk into sentences
    sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())

    # Check and remove the first sentence if it doesn't start with a capital letter or it's empty
    if sentences and (not sentences[0] or not sentences[0][0].isupper()):
        sentences = sentences[1:]

    # Check and remove the last sentence if it's empty or doesn't end with a proper punctuation
    if sentences and (not sentences[-1] or not sentences[-1][-1] in '.!?'):
        sentences = sentences[:-1]

    return ' '.join(sentences)
    
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
        self.model =torch.compile(self.model)
        print(f"Model compiled")
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

class MistralQ:
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
        self.model =torch.compile(self.model)
        print(f"Model compiled")
        #torch.save(self.model.state_dict(), "4bit")
        # Load LLaMA tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print('model_max_length:', self.tokenizer.model_max_length)
        out =self.tokenizer("<s>", return_tensors="pt",truncation=True,max_length=2000)
        print(f"tokenizer BOS={out.input_ids}")
        out =self.tokenizer.decode(out.input_ids[0,:],skip_special_tokens=False)
        print(f"self.tokenizer.decode BOS <s> ={out}")
        
        out =self.tokenizer("</s>", return_tensors="pt",truncation=True,max_length=2000)
        out =self.tokenizer.decode(out.input_ids[0,:],skip_special_tokens=False)
        print(f"self.tokenizer.decode EOS </s> ={out}")
        
        print(f"tokenizer self.tokenizer.eos_token={self.tokenizer.eos_token}")
        print(f"tokenizer self.tokenizer.bos_token={self.tokenizer.bos_token}")
        
        #tokenizer BOS=tensor([[1, 1]])
        #tokenizer EOS=tensor([[1, 2]])
        #tokenizer self.tokenizer.eos_token=</s>
        #tokenizer self.tokenizer.bos_token=<s>
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def create_prompt(self,prompt):
        """
            Showing the formating - as per https://docs.mistral.ai/llm/mistral-instruct-v0.1#chat-template
            <s>[INST] Instruction [/INST] Model answer</s>[INST] Follow-up instruction [/INST]
            https://www.promptfoo.dev/docs/guides/mistral-vs-llama/
            
            prompt="Who is Alpacona"
            tokenizer BOS=tensor([[1, 1]])
            self.tokenizer.decode BOS <s> =<s><s>
            self.tokenizer.decode EOS </s> =<s></s>
            tokenizer self.tokenizer.eos_token=</s>
            tokenizer self.tokenizer.bos_token=<s>
            Loaded Mistral 7b model
            <s>[INST]
                    Who is Alpacona</s>
                    [/INST]

            Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

                    1. Alpacona is a Brazilian singer and actress. She was born on July 1, 1990, in São Paulo, Brazil.
            2. She began her career as a singer in 2008, when she participated in the reality show "Popstars 10" and finished in second place.
            3. Since then, she has released several albums and singles, including "Olhos de Gelo" (2010), "O Que Eu Quero" (2012), and "O Que Eu Quero" (2014).
            4. She has also appeared in several TV shows and movies, including "A Família Serrada" (2011), "O Que Eu Quero" (2012), and "O Que Eu Quero" (2014).
            5. She is known for her catchy pop songs and her energetic performances on stage.
            """
    
        prompt_template=f'''<s>[INST] 
        {prompt}
        [/INST]'''
        #print(prompt_template)
        return prompt_template

    def generate_ouputput(self,prompt_template):
        #inputs = self.tokenizer(self.tokenizer.eos_token + prompt_template, return_tensors="pt",truncation=True,max_length=2000)
        inputs = self.tokenizer(prompt_template, return_tensors="pt",truncation=True,max_length=2000)
        outputs = self.model.generate(**inputs.to(self.device) ,max_new_tokens=2000)
        output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output

    def get_chat_response(self,output):
        outputcont = "".join(output)
        parts = outputcont.split("[/INST]", 1)
        return parts[1]

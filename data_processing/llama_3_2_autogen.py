import os
from autogen import AssistantAgent, UserProxyAgent
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load the model and tokenizer
model_name =  "meta-llama/Llama-3.2-1B-Instruct" #
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# 2. Define the LocalLLM class
class LocalLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters()).device

    def generate(self, prompt, max_tokens=512, temperature=0.7, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + max_tokens,
            temperature=temperature,
            do_sample=True,
            **kwargs
        )
        generated_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )[0]
        return generated_text[len(prompt):].strip()

# 3. Define the prompt formatting function
def format_prompt(system_message, user_message):
    prompt = f"""A chat between a user and an assistant.

### System:
{system_message}

### User:
{user_message}

### Assistant:
"""
    return prompt

# 4. Define the LocalAssistantAgent
class LocalAssistantAgent(AssistantAgent):
    def __init__(self, *args, local_llm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_llm = local_llm

    def generate_reply(self, prompt, **kwargs):
        if self.local_llm:
            formatted_prompt = format_prompt(self.system_message, prompt)
            return self.local_llm.generate(formatted_prompt, **kwargs)
        else:
            return super().generate_reply(prompt, **kwargs)

# 5. Instantiate the local LLM and assistant
local_llm = LocalLLM(model=model, tokenizer=tokenizer)

assistant = LocalAssistantAgent(
    name="assistant",
    llm_model="local",
    system_message="You are a helpful assistant.",
    local_llm=local_llm,
)

# 6. Create the user agent
user = UserProxyAgent(name="user")

# 7. User sends a message
user_message = "What is the capital of France?"
assistant.receive(user_message, sender=user)

# 8. Get the assistant's response
response = assistant.last_message
print("Assistant:", response.content)

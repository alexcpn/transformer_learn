from vllm import LLM, SamplingParams

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"

prompts = [
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=model_name,
  quantization="awq",
    dtype="half",
    max_model_len=512,
   gpu_memory_utilization=.9)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
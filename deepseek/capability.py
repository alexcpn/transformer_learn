from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dfd",
)

completion = client.chat.completions.create(
  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  messages=[
    {"role": "user", "content": "what is the Jacobian matrix"}
  ]
)

print(completion.choices[0].message)

# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# vllm serve --dtype bfloat16 --max_model_len 1024 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# vllm serve --dtype bfloat16 --max_model_len 1024 deepseek-ai/DDeepSeek-R1-Distill-Llama-8B
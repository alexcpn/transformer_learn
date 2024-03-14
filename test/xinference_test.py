
import openai

messages=[
    {
        "role": "user",
        "content": "Who are you?"
    }
]

client = openai.Client(api_key="empty", base_url=f"http://0.0.0.0:9997/v1")
client.chat.completions.create(
    model="mistral-instruct-v0.2",
    messages=messages,
)
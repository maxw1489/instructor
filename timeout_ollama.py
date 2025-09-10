from ollama import Client
import time

client = Client(
    host="http://10.10.10.115:11434",
    timeout=3,
)
model = "mistral-small:24b-9k"

start = time.perf_counter()
try:
    response = client.chat(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a perfect AI agent!",
        },
        {
            "role": "user",
            "content": f"Tell me a joke with 200 words",
        },
    ],
    options={"temperature": 0.3},
    )
    print(response.message.content)
except Exception as e:
    print(f"Error occurred: {e}")
    resp = None
print(f"Total time taken: {start - time.perf_counter():.2f} seconds")
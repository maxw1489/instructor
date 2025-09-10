import logging
from pprint import pprint
logging.basicConfig(level=logging.DEBUG)
from openai import OpenAI
from pydantic import BaseModel
import instructor


class Character(BaseModel):
    story: str


client = instructor.from_openai(
    OpenAI(
        base_url="http://10.10.10.115:11434/v1",
        api_key="ollama",  # required, but unused
        max_retries=0,  # Disable OpenAI client retries, let instructor handle them
    ),
    mode=instructor.Mode.JSON,
)

import time

start_time = time.time()
try:
    resp, comp = client.chat.completions.create_with_completion(
        model="mistral-small:24b-9k",
        messages=[
            {
                "role": "user",
                "content": f"Tell me a longer story living in Germany. Use a minimume of 50 sentences!!!",
            }
        ],
        response_model=Character,
        max_retries=10,
        timeout=5.0,  # Total timeout across all retry attempts
    )
except Exception as e:
    print(f"Error occurred: {e}")
    resp = None
end_time = time.time()
execution_time = end_time - start_time
print("-"*100)
print(f"Function execution time: {execution_time:.2f} seconds")
print("-"*100)
pprint(resp)
import instructor
import openai
import logging
import sys
from pydantic import BaseModel, field_validator

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Enable instructor debug logging
instructor_logger = logging.getLogger("instructor")
instructor_logger.setLevel(logging.DEBUG)

# Enable OpenAI debug logging
openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.DEBUG)

# Configuration
OLLAMA_BASE_URL = "http://10.10.10.115:11434/v1"
MODEL_NAME = "mistral-small:24b-9k"

# Initialize client with JSON mode (since function calling doesn't work well with Ollama)
client = instructor.from_openai(
    openai.OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)


# class User(BaseModel):
#     name: str
#     age: int

#     @field_validator("age")
#     def validate_age(cls, v):
#         print(f"🔍 VALIDATOR: Checking age '{v}'")
#         if v < 0:
#             print(
#                 f"❌ VALIDATION FAILED: Age {v} is negative - triggering retry"
#             )
#             raise ValueError(f"Age {v} must be positive!")
#         print(f"✅ VALIDATION PASSED: Age {v} is valid")
#         return v

class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def validate_name(cls, v):
        print(f"🔍 VALIDATOR: Checking name '{v}'")
        if not v.islower():
            print(
                f"❌ VALIDATION FAILED: Name '{v}' is not lowercase - triggering retry"
            )
            raise ValueError(f"Name '{v}' must be lowercase!")
        print(f"✅ VALIDATION PASSED: Name '{v}' is valid")
        return v


print("🧪 Testing age validation with intentionally negative age...")
print("Expected: Should retry until age becomes positive")
print("=" * 60)

try:
    # Instructor automatically retries when validation fails
    user = client.chat.completions.create(
        model=MODEL_NAME,
        response_model=User,
        messages=[{"role": "user", "content": "Arbiat is 20 years old"}],
        max_retries=3,
    )

    print(f"\n✅ SUCCESS: {user.name} is {user.age} years old")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    print("This suggests retry mechanism isn't working properly")

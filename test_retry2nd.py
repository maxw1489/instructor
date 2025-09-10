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


# Response model with validator
class NameDeriver(BaseModel):
    name: str

    @field_validator("name")
    def name_must_be_lowercase(cls, v):
        print(f"🔍 VALIDATOR: Checking name '{v}'")
        if not v.islower():
            print(
                f"❌ VALIDATION FAILED: '{v}' is not lowercase - triggering retry"
            )
            raise ValueError(f"Name '{v}' must be lowercase!")
        print(f"✅ VALIDATION PASSED: '{v}' is lowercase")
        return v


# Initialize client (using default mode - function calling)
client = instructor.from_openai(
    openai.OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)


def test_name_validation():
    try:
        print("\n📤 Sending request to instructor...")
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": "Jon is 26 years old.",
                }
            ],
            response_model=NameDeriver,
            max_retries=3,
        )

        print(f"\n✅ SUCCESS: Final result = '{result.name}'")
        print(f"✅ Validation passed on final attempt!")
        return result

    except Exception as e:
        print(f"\n❌ FAILED after all retries: {str(e)[:200]}")
        print(f"❌ This suggests the retry mechanism isn't working properly")
        return None


if __name__ == "__main__":
    result = test_name_validation()

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from query_llm import call_openai_llm, LLMRequest

if __name__ == "__main__":
    request = LLMRequest(input_text="Hello world")
    response = call_openai_llm(request)
    print("Response:", response)

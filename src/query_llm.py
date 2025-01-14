from typing import Optional
from dataclasses import dataclass
from openai import OpenAI  

@dataclass
class LLMRequest:
    """
    Collects parameters needed for an LLM call.
    """
    input_text: Optional[str] = None
    input_file_path: Optional[str] = None
    # Path to a file for logging the prompt we are about to send.
    log_input_file: Optional[str] = None
    # Path to a file for logging the LLM response.
    log_output_file: Optional[str] = None
    model: str = "gpt-4o-mini"

def call_openai_llm(request: LLMRequest) -> str:
    """
    Generic function to perform an LLM call, with optional logging of input and output.
    
    :param request: LLMRequest object with all needed parameters.
    :return: The response text from the LLM.
    """

    # If request.input_file_path is provided, read that file into request.input_text.
    if request.input_file_path and not request.input_text:
        try:
            with open(request.input_file_path, "r", encoding="utf-8") as f:
                request.input_text = f.read()
        except IOError as e:
            print(f"Error reading from {request.input_file_path}: {e}")
            request.input_text = ""

    # Safety fallback if we still have no text:
    if not request.input_text:
        print("Warning: No input text provided to call_openai_llm(). Proceeding with empty prompt.")
        request.input_text = ""

    # Optionally log the input prompt to a file
    if request.log_input_file:
        try:
            with open(request.log_input_file, "w", encoding="utf-8") as f:
                f.write(request.input_text)
        except IOError as e:
            print(f"Error writing to {request.log_input_file}: {e}")

    client = OpenAI()
    completion = client.chat.completions.create(
        model=request.model,
        store=True,
        messages=[
            {"role": "user", "content": request.input_text}
        ]
    )

    response_text = completion.choices[0].message.content.strip()

    # Optionally log the output to a file
    if request.log_output_file:
        try:
            with open(request.log_output_file, "w", encoding="utf-8") as f:
                f.write(response_text)
        except IOError as e:
            print(f"Error writing to {request.log_output_file}: {e}")

    return response_text
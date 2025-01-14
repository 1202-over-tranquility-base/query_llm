import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import pytest
from unittest.mock import patch, MagicMock
from query_llm import LLMRequest, call_openai_llm


@pytest.fixture
def mock_openai_response():
    """Fixture to provide a mock OpenAI response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked response"
    return mock_response

@patch("query_llm.OpenAI")
def test_call_openai_llm(mock_openai, mock_openai_response):
    """
    Unit test for call_openai_llm function.
    """
    # Arrange: Set up the mock and the input request
    mock_openai.return_value.chat.completions.create.return_value = mock_openai_response
    request = LLMRequest(input_text="Test prompt")

    # Act: Call the function under test
    response = call_openai_llm(request)

    # Assert: Check that the response is as expected
    assert response == "Mocked response"
    mock_openai.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        store=True,
        messages=[{"role": "user", "content": "Test prompt"}]
    )
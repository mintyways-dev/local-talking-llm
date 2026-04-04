"""Integration tests for MiniMax LLM provider.

These tests require a valid MINIMAX_API_KEY environment variable.
They are skipped automatically if the key is not set.
"""

import os
import sys
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY")
SKIP_REASON = "MINIMAX_API_KEY not set — skipping MiniMax integration tests"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def _import_create_llm(self):
        from unittest.mock import patch, MagicMock

        mock_nltk = MagicMock()
        mock_nltk.__spec__ = MagicMock()  # transformers checks __spec__ to detect nltk

        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
            "numpy": MagicMock(),
            "torch": MagicMock(),
            "torchaudio": MagicMock(),
            "nltk": mock_nltk,
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(),
            "tts": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            with patch("sys.argv", ["app.py"]):
                if "app" in sys.modules:
                    del sys.modules["app"]
                from app import create_llm
        return create_llm

    def test_minimax_basic_response(self):
        """Test that MiniMax returns a valid response to a simple prompt."""
        create_llm = self._import_create_llm()
        llm = create_llm("minimax", api_key=MINIMAX_API_KEY, temperature=0.7)
        response = llm.invoke("Say hello in exactly three words.")
        # ChatOpenAI returns an AIMessage
        content = response.content if hasattr(response, "content") else str(response)
        self.assertIsInstance(content, str)
        self.assertGreater(len(content.strip()), 0)

    def test_minimax_m27_highspeed(self):
        """Test that MiniMax-M2.7-highspeed model works."""
        create_llm = self._import_create_llm()
        llm = create_llm(
            "minimax",
            model="MiniMax-M2.7-highspeed",
            api_key=MINIMAX_API_KEY,
            temperature=0.5,
        )
        response = llm.invoke("What is 2+2? Answer with just the number.")
        content = response.content if hasattr(response, "content") else str(response)
        self.assertIn("4", content)

    def test_minimax_with_langchain_chain(self):
        """Test MiniMax works in a LangChain LCEL chain with StrOutputParser."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        create_llm = self._import_create_llm()
        llm = create_llm("minimax", api_key=MINIMAX_API_KEY, temperature=0.5)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Reply in one sentence."),
            ("human", "{input}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"input": "What color is the sky?"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)


if __name__ == "__main__":
    unittest.main()

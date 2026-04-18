"""Unit tests for TTS line-prefix sanitization behavior."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import tts module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestTtsPrefixSanitization(unittest.TestCase):
    """Tests for stripping line-start speaker labels before playback TTS."""

    def _import_tts_module(self):
        mock_modules = {
            "nltk": MagicMock(),
            "torch": MagicMock(),
            "numpy": MagicMock(),
            "torchaudio": MagicMock(),
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(),
        }

        with patch.dict("sys.modules", mock_modules):
            if "tts" in sys.modules:
                del sys.modules["tts"]
            import tts

        return tts

    def test_prefix_filter_list_is_array(self):
        tts = self._import_tts_module()
        self.assertEqual(tts.TTS_LINE_PREFIX_FILTERS, ["AI:", "Assistant:", "Bot:"])

    def test_strip_prefixes_at_line_start_with_indentation(self):
        tts = self._import_tts_module()
        text = "AI: Hello\n  Assistant: Hi there\nBot: Great\nI said AI: keep this"
        expected = "Hello\n  Hi there\nGreat\nI said AI: keep this"
        self.assertEqual(tts.strip_tts_line_prefixes(text), expected)

    def test_strip_stacked_prefixes(self):
        tts = self._import_tts_module()
        text = "AI: Assistant: Bot: Let us begin"
        self.assertEqual(tts.strip_tts_line_prefixes(text), "Let us begin")

    def test_does_not_strip_case_mismatch(self):
        tts = self._import_tts_module()
        text = "ai: keep this"
        self.assertEqual(tts.strip_tts_line_prefixes(text), text)

    def test_synthesize_uses_sanitized_text(self):
        tts = self._import_tts_module()

        service = tts.TextToSpeechService.__new__(tts.TextToSpeechService)
        service.sample_rate = 24000
        service.model = MagicMock()

        wav = MagicMock()
        wav.squeeze.return_value.cpu.return_value.numpy.return_value = [0.0]
        service.model.generate.return_value = wav

        service.synthesize("AI: Hello")
        called_text = service.model.generate.call_args[0][0]
        self.assertEqual(called_text, "Hello")

    def test_save_voice_sample_keeps_original_text(self):
        tts = self._import_tts_module()

        service = tts.TextToSpeechService.__new__(tts.TextToSpeechService)
        service.sample_rate = 24000
        service.model = MagicMock()
        service.model.generate.return_value = MagicMock()

        with patch.object(tts.ta, "save") as mock_save:
            service.save_voice_sample("AI: Keep this", "out.wav")

        called_text = service.model.generate.call_args[0][0]
        self.assertEqual(called_text, "AI: Keep this")
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Unit tests for voice activation helpers in app.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestVoiceActivationHelpers(unittest.TestCase):
    """Tests for activation threshold and state machine helpers."""

    def _import_helpers(self):
        mock_nltk = MagicMock()
        mock_nltk.__spec__ = MagicMock()

        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
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
                import app

        return app

    def test_derive_energy_threshold_uses_noise_stats(self):
        app = self._import_helpers()
        noise_levels = [0.01, 0.02, 0.015, 0.03]
        metrics = app.derive_energy_threshold(noise_levels)

        expected_mean = float(np.mean(noise_levels))
        expected_std = float(np.std(noise_levels))
        expected_threshold = max(0.01, expected_mean + 3.0 * expected_std)

        self.assertAlmostEqual(metrics["noise_rms_mean"], expected_mean)
        self.assertAlmostEqual(metrics["noise_rms_std"], expected_std)
        self.assertAlmostEqual(metrics["energy_threshold"], expected_threshold)

    def test_derive_energy_threshold_uses_minimum_on_empty(self):
        app = self._import_helpers()
        metrics = app.derive_energy_threshold([])
        self.assertEqual(metrics["energy_threshold"], 0.01)

    def test_should_start_from_history(self):
        app = self._import_helpers()
        self.assertTrue(app.should_start_from_history([True, False, True], 2))
        self.assertFalse(app.should_start_from_history([True, False, False], 2))

    def test_should_stop_recording_on_silence(self):
        app = self._import_helpers()
        should_stop = app.should_stop_recording(
            silence_frames=40,
            total_frames=50,
            frame_ms=20,
            silence_ms=800,
            max_utterance_ms=20000,
        )
        self.assertTrue(should_stop)

    def test_should_stop_recording_on_max_utterance(self):
        app = self._import_helpers()
        should_stop = app.should_stop_recording(
            silence_frames=0,
            total_frames=1000,
            frame_ms=20,
            silence_ms=800,
            max_utterance_ms=20000,
        )
        self.assertTrue(should_stop)

    def test_frame_rms_nonzero(self):
        app = self._import_helpers()
        samples = np.array([1000, -1000, 1000, -1000], dtype=np.int16)
        rms = app.frame_rms(samples.tobytes())
        self.assertGreater(rms, 0.0)


class TestCalibrationConfig(unittest.TestCase):
    """Tests for calibration config persistence helpers."""

    def _import_helpers(self):
        mock_nltk = MagicMock()
        mock_nltk.__spec__ = MagicMock()

        mock_modules = {
            "whisper": MagicMock(),
            "sounddevice": MagicMock(),
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
                import app

        return app

    def test_calibration_save_load_roundtrip(self):
        app = self._import_helpers()
        payload = {
            "energy_threshold": 0.1234,
            "noise_rms_mean": 0.0123,
            "noise_rms_std": 0.0031,
            "frame_ms": 20,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "calibration.json")
            app.save_calibration(config_path, payload)
            loaded = app.load_calibration(config_path)

        self.assertEqual(loaded, payload)

    def test_calibration_load_missing_returns_empty(self):
        app = self._import_helpers()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "missing.json")
            loaded = app.load_calibration(config_path)
        self.assertEqual(loaded, {})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import tempfile
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from modules.asr.whisper_asr import WhisperASR


class _DummyModel:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def transcribe(self, audio_path: str, **kwargs):
        self.kwargs = kwargs
        word = SimpleNamespace(start=0.0, end=0.5, word="hi", probability=0.8)
        segment = SimpleNamespace(text=" hi", words=[word], avg_logprob=-0.2)
        return iter([segment]), {"dummy": True}


class WhisperAsrKwargsTest(TestCase):
    def setUp(self) -> None:
        self.dummy_model = _DummyModel()
        patcher = mock.patch(
            "modules.asr.whisper_asr.load_model",
            return_value=self.dummy_model,
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.asr = WhisperASR(model_name="tiny", gpu=False, lang="zh")

    def _make_tiny_wav(self) -> str:
        fd, path_str = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        path = Path(path_str)
        frames = array("h", [0] * 400)
        with wave.open(path_str, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(frames.tobytes())
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path_str

    def test_transcribe_normalizes_legacy_kwargs(self) -> None:
        wav_path = self._make_tiny_wav()

        text, avg_conf, words = self.asr.transcribe(
            wav_path,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            unknown_option="drop-me",
        )

        self.assertEqual("hi", text)
        self.assertAlmostEqual(0.8, avg_conf)
        self.assertIsInstance(words, list)
        self.assertTrue(words)
        kwargs = self.dummy_model.kwargs
        self.assertIsNotNone(kwargs)
        if kwargs is None:  # defensive guard for type checkers
            self.fail("Dummy model did not receive kwargs.")
        self.assertIn("log_prob_threshold", kwargs)
        self.assertNotIn("logprob_threshold", kwargs)
        self.assertNotIn("unknown_option", kwargs)
        self.assertEqual(0.6, kwargs["no_speech_threshold"])
        self.assertTrue(kwargs["word_timestamps"])

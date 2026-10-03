"""Chunked long-audio transcription checks (fake model, no weights needed)."""
import os
import sys
import types
import unittest

import numpy as np

# app.config imports pydantic, unneeded for chunking logic.
try:
    import pydantic  # noqa: F401
except ImportError:
    _stub = types.ModuleType("pydantic")
    _stub.BaseModel = object
    sys.modules["pydantic"] = _stub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import model as M

SR = M.TARGET_SAMPLE_RATE


class Seg:
    def __init__(self, t0_ms, t1_ms, text):
        self.t0_ms, self.t1_ms, self.text = t0_ms, t1_ms, text


class Word:
    def __init__(self, seg_index, text, t0_ms, t1_ms):
        self.seg_index, self.text, self.t0_ms, self.t1_ms = seg_index, text, t0_ms, t1_ms


class Res:
    def __init__(self, text, language="fi", dur_ms=0, words=None):
        self.text, self.language = text, language
        self.segments = [Seg(0, dur_ms, text)] if text else []
        self.words = words or []


class FakeSession:
    def __init__(self, run):
        self._run, self.calls = run, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, pcm, language=None, timestamps="segment"):
        self.calls.append({"n": len(pcm), "language": language, "timestamps": timestamps})
        return self._run(np.asarray(pcm), language, timestamps)


class FakeModel:
    def __init__(self, run):
        self.session_obj = FakeSession(run)

    def session(self):
        return self.session_obj


def noisy(n, seed=0, amp=0.5):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float64) * amp).astype(np.float32)


class ChunkingTest(unittest.TestCase):
    def setUp(self):
        self._old_model = M.stt_engine._model
        self._old_chunk = M.settings.chunk_seconds

    def tearDown(self):
        M.stt_engine._model = self._old_model
        M.settings.chunk_seconds = self._old_chunk

    def run_engine(self, pcm, run, chunk=30, words=False):
        M.settings.chunk_seconds = chunk
        M.stt_engine._model = FakeModel(run)
        out = M.stt_engine.transcribe(pcm, word_timestamps=words)
        return out, M.stt_engine._model.session_obj.calls

    def test_short_single_pass(self):
        pcm = noisy(10 * SR)
        out, calls = self.run_engine(pcm, lambda c, l, t: Res("hei maailma", dur_ms=len(c) * 1000 // SR))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["language"])
        self.assertEqual(out["text"], "hei maailma")
        self.assertEqual((out["segments"][0]["start"], out["segments"][0]["end"]), (0.0, 10.0))

    def test_long_splits_offsets_and_language_pin(self):
        pcm = np.concatenate([noisy(65 * SR, seed=1)])  # loud throughout
        def run(c, lang, ts):
            self.assertLessEqual(len(c), 30 * SR + 1)
            return Res(f"[{len(c)}]", language="fi", dur_ms=len(c) * 1000 // SR)
        out, calls = self.run_engine(pcm, run)
        self.assertGreater(len(calls), 1)
        self.assertIsNone(calls[0]["language"])
        self.assertTrue(all(c["language"] == "fi" for c in calls[1:]))
        segs = out["segments"]
        self.assertEqual(segs[0]["start"], 0.0)
        for a, b in zip(segs, segs[1:]):
            self.assertAlmostEqual(a["end"], b["start"], places=2)
        self.assertAlmostEqual(segs[-1]["end"], 65.0, places=2)
        self.assertEqual(out["text"], " ".join(f"[{c['n']}]" for c in calls))
        self.assertEqual(out["language"], "fi")

    def test_silence_skipped(self):
        pcm = np.concatenate([noisy(5 * SR, seed=2), np.zeros(40 * SR, dtype=np.float32)])
        out, calls = self.run_engine(
            pcm, lambda c, l, t: Res("puhetta", dur_ms=len(c) * 1000 // SR), chunk=30)
        self.assertEqual(len(calls), 1)  # silent 40s window never decoded
        self.assertEqual(out["text"], "puhetta")

    def test_failing_chunk_splits_and_recovers(self):
        pcm = noisy(40 * SR, seed=3)
        state = {"failed": False, "ok": 0}
        def run(c, lang, ts):
            if len(c) > 6 * SR and not state["failed"]:
                state["failed"] = True
                raise RuntimeError("OutputTruncated")
            state["ok"] += 1
            return Res("ok", language="fi", dur_ms=len(c) * 1000 // SR)
        out, calls = self.run_engine(pcm, run, chunk=30)
        self.assertTrue(state["failed"])
        self.assertEqual(out["text"], " ".join(["ok"] * state["ok"]))
        self.assertAlmostEqual(out["segments"][-1]["end"], 40.0, places=2)

    def test_tiny_tail_merged(self):
        pcm = np.concatenate([noisy(int(29.5 * SR), seed=4),
                              np.zeros(SR, dtype=np.float32),
                              noisy(int(1.5 * SR), seed=5)])
        out, calls = self.run_engine(pcm, lambda c, l, t: Res("x", dur_ms=len(c) * 1000 // SR))
        self.assertEqual(len(calls), 1)  # ~2.5s tail folds into previous window
        self.assertAlmostEqual(out["segments"][0]["end"], 32.0, places=2)

    def test_word_timestamps_remapped(self):
        pcm = noisy(35 * SR, seed=5)
        def run(c, lang, ts):
            n = len(c)
            return Res("a b", language="fi", dur_ms=n * 1000 // SR,
                       words=[Word(0, "a", 0, 500), Word(0, "b", 500, n * 1000 // SR)])
        out, _ = self.run_engine(pcm, run, words=True)
        words = [w for s in out["segments"] for w in s.get("words", [])]
        self.assertEqual([w["word"] for w in words], ["a", "b"] * len(out["segments"]))
        self.assertAlmostEqual(words[2]["start"], out["segments"][1]["start"] + 0.0, places=2)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from illusion_agent.clients import MockClient
from illusion_agent.colab import run_pilot
from illusion_agent.core import sha256


class Client(MockClient):
    metadata = {"model": "fake", "revision": "unit-test"}


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "image.png").write_bytes(b"fixture")
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps({"id": "a", "original_id": "a", "image": "image.png",
            "sha256": sha256(b"fixture"), "split": "debug", "cues": [{"id": "c0", "text": ""}]}) + "\n")
        self.out = self.root / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_skips_completed_cases(self):
        first = run_pilot(Client(), self.manifest, self.out, modes=("direct", "evidence"), max_calls=1)
        second = run_pilot(Client(), self.manifest, self.out, modes=("direct", "evidence"), max_calls=4)
        third = run_pilot(Client(), self.manifest, self.out, modes=("direct", "evidence"), max_calls=4)
        self.assertEqual(second["completed_cases"], 2)
        self.assertEqual(third["attempted_calls"], 4)
        self.assertEqual(second, third)

    def test_resume_rejects_changed_prompt_config(self):
        run_pilot(Client(), self.manifest, self.out, modes=("direct",))
        with self.assertRaises(ValueError):
            run_pilot(Client(), self.manifest, self.out, modes=("evidence",))

    def test_interrupted_attempt_is_counted(self):
        class Interrupted(Client):
            def complete(self, messages, seed):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_pilot(Interrupted(), self.manifest, self.out, modes=("direct",), max_calls=1)
        summary = run_pilot(Client(), self.manifest, self.out, modes=("direct",), max_calls=1)
        self.assertEqual(summary["attempted_calls"], 1)
        self.assertEqual(summary["completed_cases"], 0)
        final = run_pilot(Client(), self.manifest, self.out, modes=("direct",), max_calls=2)
        self.assertEqual(final["completed_cases"], 1)
        self.assertEqual(final["attempted_calls"], 2)

    def test_errors_are_kept_and_not_automatically_retried(self):
        class Broken(Client):
            def complete(self, messages, seed): return "bad json", {}
        first = run_pilot(Broken(), self.manifest, self.out, modes=("direct",), max_calls=1)
        second = run_pilot(Client(), self.manifest, self.out, modes=("direct",), max_calls=1)
        self.assertEqual(first["errors"], 1)
        self.assertEqual(first, second)

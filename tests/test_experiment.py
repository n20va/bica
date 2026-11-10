import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from illusion_agent.cli import main
from illusion_agent.clients import MockClient, OllamaClient
from illusion_agent.core import load_manifest, parse_response, run_case, sha256

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "SECRET_LABEL.png"
        self.image.write_bytes(PNG)
        self.row = {"id": "SECRET_ID", "original_id": "SECRET_ORIGIN", "image": self.image.name,
                    "sha256": sha256(PNG), "split": "debug", "cues": [{"id": "c0", "text": ""}]}
        self.manifest = self.root / "manifest.jsonl"
        self.save([self.row])

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, rows):
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def invoke(self, *args):
        with patch("sys.argv", ["bica", *map(str, args)]):
            main()

    def test_gold_fields_are_rejected(self):
        self.save([{**self.row, "answer": "SECRET_ANSWER"}])
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_hash_detects_modified_image(self):
        self.image.write_bytes(PNG + b"changed")
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_original_and_hash_split_leakage(self):
        self.save([self.row, {**self.row, "id": "different", "original_id": "other", "split": "test"}])
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_truth_label_cannot_be_sent_in_cue(self):
        self.row["cues"][0]["truth"] = True
        self.save([self.row])
        with self.assertRaises(ValueError):
            load_manifest(self.manifest)

    def test_three_calls_and_no_metadata_in_model_context(self):
        row = load_manifest(self.manifest)[0]
        received = []
        class Recording(MockClient):
            def complete(self, messages, seed):
                received.append(json.loads(json.dumps(messages)))
                return super().complete(messages, seed)
        run_case(Recording(), row, row["cues"][0], "evidence", 42, lambda _: None)
        self.assertEqual(len(received), 3)
        self.assertEqual(len(received[-1]), 6)
        self.assertNotIn("SECRET", json.dumps(received))
        self.assertEqual(received[0][1]["images"], [base64.b64encode(PNG).decode()])

    def test_no_cross_case_history(self):
        row = load_manifest(self.manifest)[0]
        starts = []
        class Recording(MockClient):
            def complete(self, messages, seed):
                starts.append(len(messages))
                return super().complete(messages, seed)
        for _ in range(2):
            run_case(Recording(), row, row["cues"][0], "evidence", 42, lambda _: None)
        self.assertEqual(starts, [2, 4, 6, 2, 4, 6])

    def test_supported_answer_needs_evidence(self):
        obj = {"interpretations": [{"label": "vase", "status": "supported", "evidence": [], "against": []}],
               "cue_verdict": "not_applicable"}
        with self.assertRaises(ValueError):
            parse_response(json.dumps(obj))

    def test_malformed_response_is_logged_and_not_repaired(self):
        class Bad(MockClient):
            def complete(self, messages, seed):
                return "not json", {}
        row = load_manifest(self.manifest)[0]
        events = []
        with self.assertRaises(ValueError):
            run_case(Bad(), row, row["cues"][0], "evidence", 1, events.append)
        self.assertEqual([x["event"] for x in events], ["request", "response"])

    def test_fenced_answer_and_absent_cue_in_all_modes_preserve_content(self):
        obj = {"interpretations": [
            {"label": label, "status": "supported", "evidence": [
                {"region": region, "feature": feature}], "against": []}
            for label, region, feature in [
                ("Car with snow on it", "top of car", "snow covering the top of the car"),
                ("Car with a face made of snow", "front of car", "snow forming a face on the front of the car"),
                ("Car covered in snow", "entire car", "snow covering the entire car")]],
            "cue_verdict": "supported"}
        raw = "```json\n" + json.dumps(obj, indent=2) + "\n```"
        received = []
        class Fenced(MockClient):
            def complete(self, messages, seed):
                received.append(json.loads(json.dumps(messages)))
                return raw, {}
        row = load_manifest(self.manifest)[0]
        for mode, calls in [("direct", 1), ("repeat", 3), ("hypotheses", 3), ("evidence", 3)]:
            with self.subTest(mode=mode):
                events = []
                result = run_case(Fenced(), row, row["cues"][0], mode, 42, events.append)
                self.assertEqual(result["interpretations"], obj["interpretations"])
                self.assertEqual(result["cue_verdict"], "not_applicable")
                responses = [e for e in events if e["event"] == "response"]
                self.assertEqual([e["raw"] for e in responses], [raw] * calls)
                normalizations = [e for e in events if e["event"] == "normalization"]
                self.assertEqual(len(normalizations), 2 * calls)
                self.assertTrue(all(e["original"] == "supported" for e in normalizations
                                    if e["kind"] == "absent_cue"))
        self.assertTrue(all(m["content"] == raw for history in received
                            for m in history if m["role"] == "assistant"))

    def test_only_outer_fence_is_accepted(self):
        payload = json.dumps({"interpretations": [], "cue_verdict": "not_applicable"})
        for wrapper in ["```json\n{}\n```", "```\n{}\n```", " \n```JSON\r\n{}\r\n```\n"]:
            self.assertEqual(parse_response(wrapper.format(payload)), json.loads(payload))
        for raw in ["Here is the answer: " + payload, "```json\n" + payload,
                    "```json\n" + payload + "\n```\nExplanation",
                    "```json\n" + payload + "\n```\n```json\n" + payload + "\n```"]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_response(raw)

    def test_real_cue_verdict_is_not_masked(self):
        row = load_manifest(self.manifest)[0]
        cue = {"id": "c1", "text": "A vase"}
        for verdict in ["supported", "rejected", "uncertain", "not_applicable"]:
            class Response(MockClient):
                def complete(self, messages, seed):
                    return json.dumps({"interpretations": [], "cue_verdict": verdict}), {}
            events = []
            if verdict == "not_applicable":
                with self.assertRaises(ValueError):
                    run_case(Response(), row, cue, "direct", 42, events.append)
            else:
                answer = run_case(Response(), row, cue, "direct", 42, events.append)
                self.assertEqual(answer["cue_verdict"], verdict)
            self.assertFalse(any(e["event"] == "normalization" for e in events))

    def test_budget_checked_before_requests_or_output_creation(self):
        output = self.root / "run"
        with self.assertRaises(SystemExit), patch.object(MockClient, "complete") as call:
            self.invoke("run", "--manifest", self.manifest, "--output", output, "--max-calls", "9")
        call.assert_not_called()
        self.assertFalse(output.exists())

    def test_mock_run_cannot_be_scored_and_cannot_be_overwritten(self):
        output = self.root / "run"
        self.invoke("run", "--manifest", self.manifest, "--output", output)
        rows = [json.loads(x) for x in (output / "results.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["synthetic"] for r in rows))
        with self.assertRaises(SystemExit):
            self.invoke("export-review", "--run", output, "--manifest", self.manifest,
                        "--output", self.root / "review")
        with self.assertRaises(SystemExit):
            self.invoke("run", "--manifest", self.manifest, "--output", output)

    def test_transport_payload_without_live_server(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps({"done": True, "message": {"content": "{}"}}).encode()
        with patch("urllib.request.urlopen", return_value=Response()) as request:
            OllamaClient("test-vision").complete([{"role": "user", "content": "test", "images": ["BASE64"]}], 7)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["messages"][0]["images"], ["BASE64"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["seed"], 7)

    def test_remote_endpoint_not_implicitly_enabled(self):
        with self.assertRaises(ValueError):
            OllamaClient("model", "https://example.com")


if __name__ == "__main__":
    unittest.main()

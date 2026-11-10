"""No external packages; mock by default, optional local Ollama transport."""
import json
import urllib.request
from urllib.parse import urlsplit


class MockClient:
    """Always abstains. Does not recognize images and must never be scored as a VLM."""
    synthetic = True

    def complete(self, messages, seed):
        has_cue = any("Suggested interpretation to check" in m["content"] for m in messages)
        return json.dumps({"interpretations": [],
                           "cue_verdict": "uncertain" if has_cue else "not_applicable"}), {}


class OllamaClient:
    synthetic = False

    def __init__(self, model, base_url="http://localhost:11434", temperature=0.0,
                 max_tokens=768, timeout=60):
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("This adapter supports local Ollama only")
        if parts.username or parts.password or parts.query or parts.fragment or parts.path not in {"", "/"}:
            raise ValueError("Use a plain local server origin")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout

    def complete(self, messages, seed):
        payload = {"model": self.model, "messages": messages, "stream": False, "format": "json",
                   "options": {"temperature": self.temperature, "seed": seed,
                               "num_predict": self.max_tokens}}
        request = urllib.request.Request(self.base_url + "/api/chat",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        if not value.get("done") or value.get("done_reason") == "length":
            raise ValueError("Incomplete model response")
        usage = {k: value.get(k) for k in ("model", "prompt_eval_count", "eval_count", "total_duration")}
        usage["cost_usd"] = None
        return value["message"]["content"], usage

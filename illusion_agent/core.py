"""Experimental conditions, strict response validation, and isolated conversations."""
import base64
import hashlib
import json
import time
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"
MODES = {
    "direct": ("describe",),
    "repeat": ("describe", "revise", "revise"),
    "hypotheses": ("describe", "alternatives", "retain"),
    "evidence": ("describe", "alternatives", "verify"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_response(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"interpretations", "cue_verdict"}:
        raise ValueError("Response must contain interpretations and cue_verdict only")
    if value["cue_verdict"] not in {"supported", "rejected", "uncertain", "not_applicable"}:
        raise ValueError("Invalid cue_verdict")
    items = value["interpretations"]
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError("Expected at most four interpretations")
    labels = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"label", "status", "evidence", "against"}:
            raise ValueError("Invalid interpretation fields")
        label = item["label"]
        if not isinstance(label, str) or not label.strip() or label.casefold().strip() in labels:
            raise ValueError("Labels must be nonempty and unique")
        labels.add(label.casefold().strip())
        if item["status"] not in {"supported", "rejected", "uncertain"}:
            raise ValueError("Invalid hypothesis status")
        if not isinstance(item["evidence"], list) or not isinstance(item["against"], list):
            raise ValueError("Evidence and against must be lists")
        if not all(isinstance(x, str) and x.strip() for x in item["against"]):
            raise ValueError("Counterevidence must be text")
        if item["status"] == "supported" and not item["evidence"]:
            raise ValueError("Supported hypotheses need evidence")
        for ev in item["evidence"]:
            if not isinstance(ev, dict) or set(ev) != {"region", "feature"}:
                raise ValueError("Evidence requires region and feature")
            if not all(isinstance(v, str) and v.strip() for v in ev.values()):
                raise ValueError("Empty evidence")
    # This validates structure, NOT whether the visual evidence is correct.
    return value


def load_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    ids, splits = set(), {}
    for row in rows:
        allowed = {"id", "original_id", "image", "sha256", "split", "cues"}
        if set(row) != allowed:
            raise ValueError("Manifest fields must be " + str(sorted(allowed)))
        if not all(isinstance(row[k], str) and row[k].strip() for k in allowed - {"cues"}):
            raise ValueError("Invalid manifest text field")
        if row["id"] in ids:
            raise ValueError("Duplicate image ID")
        ids.add(row["id"])
        if row["split"] not in {"debug", "dev", "test"}:
            raise ValueError("Unknown split")
        image = (path.parent / row["image"]).resolve()
        data = image.read_bytes()
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Use PNG, JPEG or WebP")
        if sha256(data) != row["sha256"]:
            raise ValueError("Image hash mismatch: " + row["id"])
        for key in ("original:" + row["original_id"], "hash:" + row["sha256"]):
            if key in splits and splits[key] != row["split"]:
                raise ValueError("Original/image hash occurs across splits")
            splits[key] = row["split"]
        cues = row["cues"]
        if not isinstance(cues, list) or not cues:
            raise ValueError("At least one cue condition is required")
        cue_ids = set()
        for cue in cues:
            if not isinstance(cue, dict) or set(cue) != {"id", "text"}:
                raise ValueError("Cue contains unexpected fields (do not include its truth label)")
            if not isinstance(cue["id"], str) or not cue["id"] or cue["id"] in cue_ids:
                raise ValueError("Cue IDs must be unique")
            if not isinstance(cue["text"], str):
                raise ValueError("Cue must be text")
            cue_ids.add(cue["id"])
        row["image_path"] = image
    return rows


def run_case(client, row, cue, mode, seed, emit):
    # Never send manifest IDs, source filenames, group, split, or labels to the VLM.
    messages = [{"role": "system", "content": (PROMPTS / "system.txt").read_text()}]
    image = base64.b64encode(row["image_path"].read_bytes()).decode("ascii")
    answer = None
    for step, stage in enumerate(MODES[mode]):
        content = (PROMPTS / (stage + ".txt")).read_text()
        if step == 0 and cue["text"]:
            content += "\nSuggested interpretation to check (may be wrong): " + cue["text"]
        message = {"role": "user", "content": content}
        if step == 0:
            message["images"] = [image]
        messages.append(message)
        # Save exact textual context plus a hash referring to the snapshotted image.
        logged = [{k: v for k, v in m.items() if k != "images"} for m in messages]
        request = {"stage": stage, "step": step, "seed": seed,
                   "image_sha256": row["sha256"], "messages": logged}
        emit({"event": "request", **request})
        start = time.monotonic()
        try:
            raw, usage = client.complete(messages, seed)
        except Exception as exc:
            emit({"event": "transport_error", "step": step, "error": type(exc).__name__,
                  "elapsed_s": time.monotonic() - start})
            raise
        emit({"event": "response", "step": step, "raw": raw, "usage": usage,
              "elapsed_s": time.monotonic() - start})
        answer = parse_response(raw)
        if bool(cue["text"]) == (answer["cue_verdict"] == "not_applicable"):
            raise ValueError("cue_verdict inconsistent with presence/absence of cue")
        messages.append({"role": "assistant", "content": raw})
    return answer

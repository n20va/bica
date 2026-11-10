"""In-kernel GPU inference and resumable pilot. No paid inference endpoints."""
import base64
import io
import json
import os
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .core import MODES, PROMPTS, load_manifest, run_case, sha256


class QwenClient:
    synthetic = False

    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", revision=None, max_tokens=768):
        import torch
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError("GPU не выделена. В Colab выберите Runtime → Change runtime type → T4 GPU.")
        revision = HfApi().model_info(model_id, revision=revision).sha
        self.processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa")
        self.model.eval()
        self.max_tokens = max_tokens
        self.metadata = {"model": model_id, "revision": revision, "torch": torch.__version__,
                         "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
                         "dtype": "float16", "attention": "sdpa", "max_tokens": max_tokens,
                         "min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28,
                         "decoding": "greedy", "python": platform.python_version()}

    def complete(self, messages, seed):
        import torch
        from PIL import Image
        from transformers import set_seed
        set_seed(seed)
        converted, images = [], []
        for message in messages:
            content = []
            for encoded in message.get("images", []):
                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                images.append(image)
                content.append({"type": "image"})
            content.append({"type": "text", "text": message["content"]})
            converted.append({"role": message["role"], "content": content})
        text = self.processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
        length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        generated = output[:, length:]
        raw = self.processor.batch_decode(generated, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
        # Preserve raw text even when the format or completeness is wrong; core validates it.
        return raw, {"input_tokens": length, "output_tokens": generated.shape[1],
                     "at_token_limit": generated.shape[1] >= self.max_tokens, "api_cost_usd": 0}


def _atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def run_pilot(client, manifest, output, modes=("direct", "repeat", "hypotheses", "evidence"),
              max_calls=60, seed=42, limit=None):
    """Resume completed cases exactly; partial cases restart and consume extra calls.

    max_calls limits attempted model calls across all sessions, including interrupted attempts.
    A saved error is a completed case, not automatically retried. Config must match on resume.
    """
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("Invalid modes")
    if max_calls < 1 or (limit is not None and limit < 1):
        raise ValueError("Limits must be positive")
    rows = load_manifest(manifest)
    if any(r["split"] != "debug" for r in rows):
        raise ValueError("This pilot runner accepts debug data only")
    rows = rows[:limit] if limit is not None else rows
    cases = [(row, cue, mode) for row in rows for cue in row["cues"] for mode in modes]
    random.Random(seed).shuffle(cases)
    config = {"schema_version": 1, "synthetic": client.synthetic,
              "client": client.metadata, "modes": list(modes), "seed": seed, "limit": limit,
              "manifest_sha256": sha256(Path(manifest).read_bytes()),
              "prompt_hashes": {p.name: sha256(p.read_bytes()) for p in PROMPTS.glob("*.txt")},
              "code_hashes": {p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "config.json"
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError("Конфигурация изменилась. Для нового эксперимента выберите новую папку.")
    else:
        if any(out.iterdir()):
            raise ValueError("Output is not an empty experiment directory")
        _atomic_json(config_file, config)
        shutil.copyfile(manifest, out / "manifest.jsonl")
    (out / "cases").mkdir(exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    for row in rows:
        dest = out / "images" / (row["sha256"] + row["image_path"].suffix.lower())
        if not dest.exists():
            shutil.copyfile(row["image_path"], dest)
    trace_path = out / "trace.jsonl"
    # A separate atomic counter survives partial final lines in trace.jsonl.
    counter_path = out / "call_counter.json"
    used = json.loads(counter_path.read_text())["attempted"] if counter_path.exists() else 0
    if not counter_path.exists() and trace_path.exists() and trace_path.stat().st_size:
        raise ValueError("Missing call counter for existing trace")
    stopped = False
    for index, (row, cue, mode) in enumerate(cases):
        case_id = f"case-{index:05d}"
        destination = out / "cases" / (case_id + ".json")
        if destination.exists():
            continue
        if used + len(MODES[mode]) > max_calls:
            stopped = True
            print(f"Лимит: использовано {used} вызовов. Для продолжения увеличьте max_calls.")
            break
        attempt = datetime.now(timezone.utc).isoformat()
        with trace_path.open("a") as trace:
            def emit(event):
                nonlocal used
                if event["event"] == "request":
                    used += 1
                    _atomic_json(counter_path, {"attempted": used})
                trace.write(json.dumps({"case_id": case_id, "attempt": attempt, **event},
                                       ensure_ascii=False) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
            result = {"case_id": case_id, "image_id": row["id"], "original_id": row["original_id"],
                      "image_sha256": row["sha256"], "cue_id": cue["id"], "mode": mode,
                      "repeat": 0, "synthetic": client.synthetic}
            try:
                result["answer"] = run_case(client, row, cue, mode, seed, emit)
                result["status"] = "ok"
            except Exception as exc:
                result.update(status="error", error=type(exc).__name__, detail=str(exc)[:500])
            _atomic_json(destination, result)
        print(f"{index + 1}/{len(cases)} {row['id']} {mode}: {result['status']}", flush=True)
    completed = [json.loads(p.read_text()) for p in sorted((out / "cases").glob("case-*.json"))]
    tmp = out / "results.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in completed))
    tmp.replace(out / "results.jsonl")
    summary = {"planned_cases": len(cases), "completed_cases": len(completed), "attempted_calls": used,
               "errors": sum(r["status"] != "ok" for r in completed), "stopped_on_budget": stopped,
               "synthetic": client.synthetic}
    _atomic_json(out / "summary.json", summary)
    return summary

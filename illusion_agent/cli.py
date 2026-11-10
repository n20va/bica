import argparse
import csv
import json
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .clients import MockClient, OllamaClient
from .core import MODES, PROMPTS, load_manifest, run_case, sha256
from .evaluation import score


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run(args):
    rows = load_manifest(args.manifest)
    selected = [r for r in rows if r["split"] == args.split]
    if not selected:
        raise ValueError("No examples in selected split")
    modes = args.modes.split(",")
    if len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("Invalid or duplicated mode")
    if args.repeats < 1 or args.max_calls < 1 or args.max_tokens < 1:
        raise ValueError("Counts must be positive")
    cases = [(row, cue, mode, repeat) for row in selected for cue in row["cues"]
             for mode in modes for repeat in range(args.repeats)]
    calls = sum(len(MODES[c[2]]) for c in cases)
    if calls > args.max_calls:
        raise ValueError(f"Plan requires {calls} calls, exceeding limit {args.max_calls}")
    if args.provider == "ollama" and not args.model:
        raise ValueError("Specify --model with an installed vision-capable Ollama model")
    random.Random(args.seed).shuffle(cases)
    plan = {"cases": len(cases), "max_calls": calls, "provider": args.provider,
            "synthetic": args.provider == "mock", "split": args.split}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    client = MockClient() if args.provider == "mock" else OllamaClient(
        args.model, args.base_url, args.temperature, args.max_tokens)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    (out / "images").mkdir()
    for row in selected:
        dst = out / "images" / (row["sha256"] + row["image_path"].suffix.lower())
        if not dst.exists():
            shutil.copyfile(row["image_path"], dst)
    shutil.copytree(PROMPTS, out / "prompts")
    shutil.copyfile(args.manifest, out / "manifest.jsonl")
    config = {k: v for k, v in vars(args).items() if k != "func"}
    config.update(plan, python=platform.python_version(), created_at=datetime.now(timezone.utc).isoformat(),
                  source_hashes={p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")},
                  prompt_hashes={p.name: sha256(p.read_bytes()) for p in PROMPTS.glob("*.txt")})
    write_json(out / "config.json", config)
    failures = 0
    with (out / "trace.jsonl").open("w") as trace, (out / "results.jsonl").open("w") as results:
        for index, (row, cue, mode, repeat) in enumerate(cases):
            case_id = f"case-{index:05d}"
            def emit(event):
                trace.write(json.dumps({"case_id": case_id, **event}, ensure_ascii=False) + "\n")
                trace.flush()
            result = {"case_id": case_id, "image_id": row["id"], "original_id": row["original_id"],
                      "image_sha256": row["sha256"], "cue_id": cue["id"], "mode": mode,
                      "repeat": repeat, "synthetic": client.synthetic}
            try:
                result["answer"] = run_case(client, row, cue, mode, args.seed + repeat, emit)
                result["status"] = "ok"
            except Exception as exc:
                failures += 1
                result.update(status="error", error=type(exc).__name__)
            results.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.flush()
    summary = {**plan, "errors": failures, "completed_cases": len(cases)}
    write_json(out / "summary.json", summary)
    print(json.dumps({"output": str(out.resolve()), **summary}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


def export_review(args):
    run_dir = Path(args.run)
    config = json.loads((run_dir / "config.json").read_text())
    if config["synthetic"]:
        raise ValueError("Mock runs cannot be exported for scientific scoring")
    rows = [json.loads(x) for x in (run_dir / "results.jsonl").read_text().splitlines()]
    manifest = {r["id"]: r for r in load_manifest(args.manifest)}
    # Review packets deliberately hide model and mode. Truth labels live outside inference.
    random.Random(args.seed).shuffle(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    packets, mapping = [], []
    for i, row in enumerate(rows):
        rid = f"review-{i:05d}"
        source = manifest[row["image_id"]]
        if source["sha256"] != row["image_sha256"]:
            raise ValueError("Review manifest differs from run images")
        image = source["sha256"] + source["image_path"].suffix.lower()
        shutil.copyfile(run_dir / "images" / image, out / image)
        cue = next(c["text"] for c in source["cues"] if c["id"] == row["cue_id"])
        packets.append({"review_id": rid, "image": image, "suggestion": cue,
                        "answer": row.get("answer"), "technical_error": row["status"] != "ok"})
        mapping.append({"review_id": rid, **{k: row[k] for k in (
            "case_id", "image_id", "original_id", "cue_id", "mode", "repeat", "status")}})
    write_json(out / "packets.json", packets)
    # Keep this file with the experiment operator, not the blinded reviewers.
    write_json(run_dir / (out.name + "_mapping.json"), mapping)
    with (out / "ratings.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "both_valid", "false_alternative", "cue_correct", "grounded", "notes"])
        writer.writerows([[p["review_id"], "", "", "", "", ""] for p in packets])
    print(out.resolve())


def main():
    parser = argparse.ArgumentParser(description="BICA visual ambiguity research prototype")
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="runs/demo")
    p.add_argument("--provider", choices=["mock", "ollama"], default="mock")
    p.add_argument("--model")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--split", choices=["debug", "dev", "test"], default="debug")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--max-calls", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=run)
    p = sub.add_parser("export-review")
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=91)
    p.set_defaults(func=export_review)
    p = sub.add_parser("score")
    p.add_argument("--run", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--ratings", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=score)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        parser.exit(2, f"Error: {exc}\n")

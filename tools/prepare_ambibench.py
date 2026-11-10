"""Download a small, revision-pinned DEBUG subset, not the full benchmark."""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "BLNL/AmbiBench"


def download(url, limit=8_000_000):
    with urllib.request.urlopen(url, timeout=45) as response:
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("Download exceeds per-file limit")
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/ambibench_debug")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    if not 1 <= args.limit <= 20:
        parser.error("Use 1–20 examples for this debug downloader")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; use a new directory")
    info = json.loads(download("https://huggingface.co/api/datasets/" + REPO))
    revision = args.revision or info["sha"]
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/"
    raw = download(base + "test/metadata.jsonl")
    metadata = [json.loads(s) for s in raw.decode().splitlines() if s.strip()]
    candidates = sorted((r for r in metadata if r["type"] == "bistable_open"), key=lambda r: r["file_name"])
    selected, seen = [], set()
    for row in candidates:
        if row["file_name"] not in seen:
            selected.append(row)
            seen.add(row["file_name"])
        if len(selected) == args.limit:
            break
    if len(selected) != args.limit:
        raise ValueError("Insufficient matching rows")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    manifest, annotations, provenance = [], [], []
    for i, row in enumerate(selected):
        name = row["file_name"]
        url = base + "test/" + urllib.parse.quote(name, safe="")
        blob = download(url)
        suffix = Path(name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Unexpected image extension")
        item_id = f"ambi-debug-{i:03d}"
        path = Path("images") / (item_id + suffix)
        (output / path).write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        manifest.append({"id": item_id, "original_id": item_id, "image": str(path),
                         "sha256": digest, "split": "debug", "cues": [{"id": "c0", "text": ""}]})
        annotations.append({"id": item_id, "group": "ambiguous", "reference_answer": row["answer"],
                            "source_question": row["question"], "review_status": "pending_two_human_reviewers",
                            "evidence": [], "cues": {"c0": "neutral"}})
        provenance.append({"id": item_id, "source_filename": name, "source_url": url, "sha256": digest})
    for name, rows in [("manifest.jsonl", manifest), ("annotations.jsonl", annotations)]:
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (output / "provenance.json").write_text(json.dumps({
        "repository": REPO, "revision": revision, "selection": "first unique bistable_open filenames, sorted",
        "purpose": "debug only; never an untouched test set", "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_card_license": info.get("cardData", {}).get("license"),
        "note": "Dataset card licensing is not independent verification of rights for every source artwork.",
        "items": provenance}, ensure_ascii=False, indent=2))
    print(f"Prepared {len(manifest)} debug images in {output}; annotations still require review")


if __name__ == "__main__":
    main()

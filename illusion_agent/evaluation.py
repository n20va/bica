"""Aggregate adjudicated human ratings; never use the agent as its own judge."""
import csv
import json
from collections import defaultdict
from pathlib import Path

METRICS = ("both_valid", "false_alternative", "cue_correct", "grounded")


def aggregate(mapping, ratings, annotations):
    items = {r["review_id"]: r for r in mapping}
    scores = {r["review_id"]: r for r in ratings}
    if len(items) != len(mapping) or len(scores) != len(ratings) or items.keys() != scores.keys():
        raise ValueError("Ratings must contain every review ID exactly once")
    groups = defaultdict(lambda: {"n": 0, "technical_errors": 0,
                                  **{m: [] for m in METRICS}})
    for rid, row in items.items():
        gold = annotations[row["image_id"]]
        if gold.get("review_status") != "adjudicated":
            raise ValueError("Annotations must be adjudicated before scoring")
        group = gold["group"]
        cue_kind = gold["cues"][row["cue_id"]]
        if group not in {"ambiguous", "ordinary", "abstract"} or cue_kind not in {"neutral", "supported", "misleading"}:
            raise ValueError("Unknown group or cue kind")
        bucket = groups[(row["mode"], group, cue_kind)]
        bucket["n"] += 1
        failed = row["status"] != "ok"
        bucket["technical_errors"] += int(failed)
        applicability = {"both_valid": group == "ambiguous", "false_alternative": group == "ordinary",
                         "cue_correct": cue_kind != "neutral", "grounded": True}
        for metric in METRICS:
            score = scores[rid].get(metric, "").strip()
            applicable = applicability[metric]
            if not applicable or failed:
                if score != "NA":
                    raise ValueError(f"{rid}/{metric}: use NA for inapplicable or technical errors")
                # Technical failures are failures for success metrics, not evidence of false alternatives.
                if applicable and failed and metric != "false_alternative":
                    bucket[metric].append(0)
            else:
                if score not in {"0", "1"}:
                    raise ValueError(f"{rid}/{metric}: completed applicable rating must be 0 or 1")
                bucket[metric].append(int(score))
    result = []
    for (mode, group, cue), bucket in sorted(groups.items()):
        record = {"mode": mode, "group": group, "cue_kind": cue,
                  "n": bucket["n"], "technical_errors": bucket["technical_errors"]}
        for metric in METRICS:
            values = bucket[metric]
            record[metric + "_n"] = len(values)
            record[metric + "_rate"] = sum(values) / len(values) if values else None
        result.append(record)
    return result


def score(args):
    run_dir = Path(args.run)
    if json.loads((run_dir / "config.json").read_text())["synthetic"]:
        raise ValueError("Synthetic runs cannot produce research metrics")
    mapping = json.loads(Path(args.mapping).read_text())
    with Path(args.ratings).open() as f:
        ratings = list(csv.DictReader(f))
    raw = [json.loads(s) for s in Path(args.annotations).read_text().splitlines() if s.strip()]
    annotations = {r["id"]: r for r in raw}
    if len(raw) != len(annotations):
        raise ValueError("Duplicate annotations")
    run_rows = [json.loads(s) for s in (run_dir / "results.jsonl").read_text().splitlines()]
    indexed = {r["case_id"]: r for r in run_rows}
    if {r["case_id"] for r in mapping} != indexed.keys() or len(mapping) != len(indexed):
        raise ValueError("Mapping must cover the entire run")
    for row in mapping:
        if any(row[k] != indexed[row["case_id"]][k] for k in row if k != "review_id"):
            raise ValueError("Mapping does not match saved run")
    output = aggregate(mapping, ratings, annotations)
    with Path(args.output).open("x") as f:
        json.dump({"note": "Descriptive rates only; repeat/image dependence not resolved by these counts",
                   "results": output}, f, ensure_ascii=False, indent=2)
    print(Path(args.output).resolve())

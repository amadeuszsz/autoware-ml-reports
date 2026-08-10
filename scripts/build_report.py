#!/usr/bin/env python
"""Build a metric report by filling report_layout.html from JSON release files.

Each input file is one release in the report_data.schema.json format (one model
per test set). Files are read IN ORDER — oldest first, last is the current
release — and their models are concatenated per test set so the report draws them
as comparison bars. The report layout and this builder only consume the JSON
contract; how each release JSON is produced is out of scope (see
scripts/mlflow_to_json.py for the ad-hoc extractor used for the demo data).

    python scripts/build_report.py --layout templates/report_layout.html \
        --out reports/ptv3_demo.html  data/ptv3_v0.1.json \
        data/ptv3_v0.2.json data/ptv3_v0.3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUPPORTED_SCHEMA = "0.1"
OPEN_TAG = '<script id="report-data" type="application/json">'
CLOSE_TAG = "</script>"


def _validate(doc, source):
    """Enforce the report_data.schema.json required fields (no jsonschema dep)."""
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: top level must be a JSON object.")
    version = doc.get("schema_version")
    if version != SUPPORTED_SCHEMA:
        raise ValueError(f"{source}: schema_version must be {SUPPORTED_SCHEMA!r}, got {version!r}.")
    if not isinstance(doc.get("test_sets"), list):
        raise ValueError(f"{source}: 'test_sets' must be a list.")
    for test_set in doc["test_sets"]:
        if not isinstance(test_set, dict):
            raise ValueError(f"{source}: every test set must be a JSON object.")
        for field in ("id", "version", "tasks", "models"):
            if field not in test_set:
                raise ValueError(f"{source}: test set missing required field {field!r}.")
        if not isinstance(test_set["models"], list):
            raise ValueError(f"{source}: test set {test_set['id']!r} 'models' must be a list.")
        for model in test_set["models"]:
            if not isinstance(model, dict) or "name" not in model or "metrics" not in model:
                raise ValueError(
                    f"{source}: test set {test_set['id']!r} models need 'name' and 'metrics'."
                )


def merge_releases(docs):
    """Concatenate release docs (in order) into one report-data document.

    Test sets are aligned by ``id`` (first-appearance order); each file's models
    are appended in file order. Shared fields (label, tasks, dataset,
    metric_spec) are backfilled from the first file that provides a value;
    ``version`` mismatches beyond the MAJOR digit only warn (the page greys
    hard MAJOR mismatches itself).
    """
    repo_url = None
    order = []
    merged = {}
    for source, doc in docs:
        if repo_url is None and doc.get("repo_url"):
            repo_url = doc["repo_url"]
        for test_set in doc["test_sets"]:
            test_set_id = test_set["id"]
            if test_set_id not in merged:
                merged[test_set_id] = {
                    "id": test_set_id,
                    "label": test_set.get("label", test_set_id),
                    "version": test_set["version"],
                    "tasks": test_set.get("tasks", []),
                    "dataset": test_set.get("dataset"),
                    "metric_spec": test_set.get("metric_spec"),
                    "models": [],
                }
                order.append(test_set_id)
            slot = merged[test_set_id]
            major = str(slot["version"]).split(".")[0]
            if str(test_set["version"]).split(".")[0] != major:
                print(f"WARNING: {source}: test set {test_set_id!r} MAJOR version "
                      f"{test_set['version']!r} differs from {slot['version']!r} — "
                      "releases are not comparable.", file=sys.stderr)
            if not slot["tasks"] and test_set.get("tasks"):
                slot["tasks"] = test_set["tasks"]
            if slot["label"] == test_set_id and test_set.get("label"):
                slot["label"] = test_set["label"]
            if slot["dataset"] is None and test_set.get("dataset") is not None:
                slot["dataset"] = test_set["dataset"]
            if slot["metric_spec"] is None and test_set.get("metric_spec") is not None:
                slot["metric_spec"] = test_set["metric_spec"]
            slot["models"].extend(test_set["models"])
    report = {"test_sets": [merged[test_set_id] for test_set_id in order]}
    if repo_url is not None:
        report = {"repo_url": repo_url, **report}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Release JSON files, oldest first (last = current).")
    parser.add_argument("--layout", default="templates/report_layout.html", help="Layout HTML to fill.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    args = parser.parse_args(argv)

    docs = []
    for path in args.files:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        _validate(doc, path)
        docs.append((path, doc))

    report = merge_releases(docs)
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False)
    # A literal "</..." inside a string would terminate the <script> block early.
    payload = payload.replace("</", "<\\/")

    with open(args.layout, encoding="utf-8") as handle:
        html = handle.read()
    if OPEN_TAG not in html:
        raise ValueError(f"{args.layout}: no {OPEN_TAG!r} block found.")
    start = html.index(OPEN_TAG) + len(OPEN_TAG)
    end = html.index(CLOSE_TAG, start)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html[:start] + "\n" + payload + "\n" + html[end:])

    summary = ", ".join(f"{t['id']}[{len(t['models'])}]" for t in report["test_sets"])
    print(f"built {args.out} from {len(docs)} file(s): {summary}")


if __name__ == "__main__":
    main()

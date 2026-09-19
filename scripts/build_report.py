#!/usr/bin/env python
"""Build a metric report by filling report_layout.html from JSON release files.

Each input file is one release in the report_data.schema.json format (one model
per test set). Files are read IN ORDER, oldest first, last is the current
release, and their models are concatenated per test set so the report draws them
as comparison bars. The current release defines the test set the page describes:
its taxonomy, dataset, metric spec, label and version win over older files. The
report layout and this builder only consume the JSON contract; how each release
JSON is produced is out of scope (see scripts/mlflow_to_json.py for the MLflow
extractor).

    python scripts/build_report.py --layout templates/report_layout.html \
        --out reports/litept_v0.1.0.html  data/ptv3_v0.0.1.json data/litept_v0.1.0.json
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path

SUPPORTED_SCHEMA = "0.2"
OPEN_TAG = '<script id="report-data" type="application/json">'
CLOSE_TAG = "</script>"
TITLE_RE = re.compile(r"<title>.*?</title>", re.DOTALL)


def _validate(doc, source):
    """Enforce the report_data.schema.json required fields (no jsonschema dep)."""
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: top level must be a JSON object.")
    version = doc.get("schema_version")
    if version != SUPPORTED_SCHEMA:
        raise ValueError(f"{source}: schema_version must be {SUPPORTED_SCHEMA!r}, got {version!r}.")
    if not isinstance(doc.get("test_sets"), list):
        raise ValueError(f"{source}: 'test_sets' must be a list.")
    release = doc.get("release")
    if release is not None:
        if not isinstance(release, dict):
            raise ValueError(f"{source}: 'release' must be a JSON object.")
        changes = release.get("changes", {})
        if not isinstance(changes, dict):
            raise ValueError(f"{source}: 'release.changes' must be a JSON object.")
        for key, items in changes.items():
            if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
                raise ValueError(f"{source}: 'release.changes.{key}' must be a list of strings.")
    for test_set in doc["test_sets"]:
        if not isinstance(test_set, dict):
            raise ValueError(f"{source}: every test set must be a JSON object.")
        for field in ("id", "version", "tasks", "taxonomy", "models"):
            if field not in test_set:
                raise ValueError(f"{source}: test set missing required field {field!r}.")
        if not isinstance(test_set["models"], list):
            raise ValueError(f"{source}: test set {test_set['id']!r} 'models' must be a list.")
        for task in test_set["tasks"]:
            taxonomy = test_set["taxonomy"].get(task)
            if not isinstance(taxonomy, dict) or not taxonomy.get("classes") or not isinstance(taxonomy.get("groups"), dict):
                raise ValueError(
                    f"{source}: test set {test_set['id']!r} needs a taxonomy with classes and groups for {task!r}."
                )
        for model in test_set["models"]:
            if not isinstance(model, dict) or "name" not in model or "metrics" not in model:
                raise ValueError(
                    f"{source}: test set {test_set['id']!r} models need 'name' and 'metrics'."
                )


def merge_releases(docs):
    """Concatenate release docs (in order) into one report-data document.

    Test sets are aligned by ``id`` (first-appearance order); each file's models
    are appended in file order. The descriptive fields (label, version, tasks,
    taxonomy, dataset, metric_spec, notes) and the top-level release notes are
    taken from the LAST file that provides a value, so the page describes the
    current release. A
    ``version`` whose MAJOR differs from an earlier file's only warns (the page
    greys hard MAJOR mismatches itself through each model's test_set_version).
    """
    repo_url = None
    release = None
    order = []
    merged = {}
    for source, doc in docs:
        if repo_url is None and doc.get("repo_url"):
            repo_url = doc["repo_url"]
        if doc.get("release") is not None:
            release = doc["release"]
        for test_set in doc["test_sets"]:
            test_set_id = test_set["id"]
            if test_set_id not in merged:
                merged[test_set_id] = {"id": test_set_id, "models": []}
                order.append(test_set_id)
            slot = merged[test_set_id]
            if "version" in slot and str(test_set["version"]).split(".")[0] != str(slot["version"]).split(".")[0]:
                print(f"WARNING: {source}: test set {test_set_id!r} MAJOR version "
                      f"{test_set['version']!r} differs from {slot['version']!r}, "
                      "earlier releases are greyed as a different test set.", file=sys.stderr)
            for field in ("label", "version", "tasks", "taxonomy", "dataset", "metric_spec", "notes"):
                if test_set.get(field) is not None:
                    slot[field] = test_set[field]
            slot.setdefault("label", test_set_id)
            slot["models"].extend(test_set["models"])
    report = {"test_sets": [merged[test_set_id] for test_set_id in order]}
    if release is not None:
        report = {"release": release, **report}
    if repo_url is not None:
        report = {"repo_url": repo_url, **report}
    return report


def default_title(current):
    """``<model> <release> model report`` from the current (last) release file."""
    for test_set in current["test_sets"]:
        if test_set["models"]:
            newest = test_set["models"][-1]
            family = newest.get("model") or newest["name"]
            return f"{family} {newest['name']} model report"
    raise ValueError("the current release file has no models, pass --title explicitly.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Release JSON files, oldest first (last = current).")
    parser.add_argument("--layout", default="templates/report_layout.html", help="Layout HTML to fill.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--title", default=None,
                        help="Document title (default: '<model> <release> model report' of the current release).")
    args = parser.parse_args(argv)

    docs = []
    for path in args.files:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        _validate(doc, path)
        docs.append((path, doc))

    report = merge_releases(docs)
    title = args.title or default_title(docs[-1][1])
    report["title"] = title
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False)
    # A literal "</..." inside a string would terminate the <script> block early.
    payload = payload.replace("</", "<\\/")

    with open(args.layout, encoding="utf-8") as handle:
        html = handle.read()
    if OPEN_TAG not in html:
        raise ValueError(f"{args.layout}: no {OPEN_TAG!r} block found.")
    if not TITLE_RE.search(html):
        raise ValueError(f"{args.layout}: no <title> element found.")
    html = TITLE_RE.sub(f"<title>{html_lib.escape(title)}</title>", html, count=1)
    start = html.index(OPEN_TAG) + len(OPEN_TAG)
    end = html.index(CLOSE_TAG, start)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(html[:start] + "\n" + payload + "\n" + html[end:])

    summary = ", ".join(f"{t['id']}[{len(t['models'])}]" for t in report["test_sets"])
    print(f"built {args.out} ({title!r}) from {len(docs)} file(s): {summary}")


if __name__ == "__main__":
    main()

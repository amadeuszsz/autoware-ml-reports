#!/usr/bin/env python
"""Turn MLflow test runs of one model release into a report-data release JSON.

One release file (schema/report_data.schema.json) holds one test set per test run.
Every ``--run`` names the test set a run was evaluated on; the run's buckets are
picked by the tasks of that test set (``det3d*`` buckets for a detection set,
``seg3d*`` for a segmentation set, all of them plus ``segdet3d`` for the joint
set).

Everything that describes the test set comes from the run itself, so the report
states exactly what was evaluated:

* the label vocabulary (classes and behaviour groups) from the database taxonomy
  the run logged as parameters;
* the scenes from the scenario lists of the test source that serves the test
  set's tasks, and the frame count from the per-frame rates the suites report;
* the class distributions from the run's ``gt_count_*`` keys (detection, the
  filtered eval GT set) and confusion-matrix row sums (segmentation, valid
  labeled points);
* the metric parameters, cross-checked against the values the run logged.

A run that reads several sources, detection from one corpus and segmentation
from another, serves several test sets: name it once per test set it covers.

    python scripts/mlflow_to_json.py \
        --db tmp/mlflow-v0.1.0.db \
        --release v0.1.0 --model "LitePT multi-head" \
        --run det3d-j6gen2=390184dc... --run seg3d-j6gen2=390184dc... \
        --run det3d-seg3d-j6gen2=3c45e618... \
        --out data/litept_v0.1.0.json
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEMA_VERSION = "0.2"
REPO_URL = "https://github.com/tier4/autoware-ml"

DET_BUCKETS = ("det3d", "det3d_grouped", "det3d_visible", "det3d_overall")
SEG_BUCKETS = ("seg3d", "seg3d_grouped")
JOINT_BUCKETS = ("segdet3d",)
# Suite prefix (first key segment after the stage) to report bucket. The merged
# point suite emits its behaviour-group view under a grouped/ sub-namespace.
BUCKET_ALIAS = {
    "det3d": "det3d",
    "det3d_grouped": "det3d_grouped",
    "det3d_visible": "det3d_visible",
    "det3d_overall": "det3d_overall",
    "seg3d": "seg3d",
    "seg3d_pt": "seg3d",
    "seg3d_grouped": "seg3d_grouped",
    "seg3d_pt/grouped": "seg3d_grouped",
    "segdet3d_pt": "segdet3d",
}
# Per task, the whole-scene count and the rate of the same quantity per frame.
# Their quotient is the number of frames the task's suite scored.
FRAME_COUNTERS = {
    "det3d": ("det3d", "confident_error_count_0m_121m", "confident_errors_per_frame_0m_121m"),
    "seg3d": ("seg3d", "error_cluster_count_0m_121m", "error_clusters_per_frame_0m_121m"),
}


@dataclass(frozen=True)
class TestSetDef:
    """Static identity of a test set: its tasks and the buckets it collects."""

    tasks: tuple[str, ...]
    buckets: tuple[str, ...]


TEST_SETS = {
    "det3d-j6gen2": TestSetDef(("det3d",), DET_BUCKETS),
    "seg3d-j6gen2": TestSetDef(("seg3d",), SEG_BUCKETS),
    "det3d-seg3d-j6gen2": TestSetDef(("det3d", "seg3d"), DET_BUCKETS + SEG_BUCKETS + JOINT_BUCKETS),
}

# Spec-versioned metric parameters. Values the run logs as parameters are
# cross-checked in ``check_spec_against_params``; the rest are code defaults.
ROAD = ["road", "road_shoulder", "crosswalk", "drivable_area", "intersection_area", "crosswalk_polygon"]
COLLISION_MODEL = {
    "horizon_s": 4.0, "dt_s": 0.1, "max_lateral_accel_mps2": 3.0, "min_radius_m": 6.4,
    "max_speed_mps": 16.7, "road": ROAD,
    "vehicle": {"wheel_base": 4.76, "front_overhang": 0.95, "rear_overhang": 1.53,
                "wheel_tread": 1.75, "left_overhang": 0.27, "right_overhang": 0.27},
}
BUNDLE = "0.2.0"
DET_SPEC = {
    "center_distance_matching": {"version": "0.1", "params": {"thresholds_m": [0.5, 1.0, 2.0, 4.0], "tp_threshold_m": 2.0}},
    "gt_filter": {"version": "0.1", "params": {"min_num_points": 2, "eval_class_range_m": 121.0}},
    "occlusion_split": {"version": "0.1", "params": {"visible_min_num_points": 1, "overall_min_num_points": 0}},
    "tp_errors": {"version": "0.1", "params": {"recall_targets": {"default": 0.10, "medium": 0.40}, "optimal": "max-F1"}},
    "corner_error": {"version": "0.1", "params": {"tp_threshold_m": 2.0, "percentiles": [95.0]}},
    "heading_flip": {"version": "0.1", "params": {"tp_threshold_m": 2.0, "flip_threshold_rad": round(math.pi / 2.0, 4)}},
    "nearest_surface_error": {"version": "0.1", "params": {"tp_threshold_m": 2.0, "percentiles": [5.0, 95.0]}},
    "critical_fp_fn": {"version": "0.2", "params": {"confidences": [0.3, 0.5], "match_threshold_m": 2.0,
                                                      "critical": "finite collision TTC within the horizon"}},
    "collision_weighted_map": {"version": "0.1", "params": {"decay": 0.5, "thresholds_m": [0.5, 1.0, 2.0, 4.0]}},
    "calibration_error": {"version": "0.1", "params": {"num_bins": 15, "tp_threshold_m": 2.0}},
    "confident_error": {"version": "0.1", "params": {"score_threshold": 0.5, "min_score": 0.1, "tp_threshold_m": 2.0}},
    "confusion_matrix": {"version": "0.1", "params": {"match_threshold_m": 2.0, "min_score": 0.1, "matched_only": True}},
    "collision_model": {"version": "0.1", "params": COLLISION_MODEL},
    "region_filter": {"version": "0.1", "params": {"road": ROAD, "walkway": ["walkway"], "margin_m": 0.0}},
    "corridor_filter": {"version": "0.1", "params": {"width_m": 3.0}},
    "collision_filter": {"version": "0.1", "params": COLLISION_MODEL},
}
SEG_SPEC = {
    "error_clusters": {"version": "0.1", "params": {"cluster_radius_m": 0.5, "min_cluster_points": 1}},
    "tolerant_error": {"version": "0.1", "params": {"radius_m": 0.2}},
    "calibration_error": {"version": "0.1", "params": {"num_bins": 15, "per_class": "predicted"}},
    "entropy_auroc": {"version": "0.1", "params": {"num_bins": 8192}},
    "confident_error": {"version": "0.1", "params": {"entropy_threshold": 0.3}},
    "confusion_matrix": {"version": "0.1", "params": {"level": "point"}},
    "region_filter": {"version": "0.1", "params": {"road": ROAD, "walkway": ["walkway"], "margin_m": -0.2}},
    "corridor_filter": {"version": "0.1", "params": {"width_m": 3.0}},
    "collision_filter": {"version": "0.1", "params": COLLISION_MODEL},
}
PARTIAL_DETECTION_SPEC = {"version": "0.1", "params": {"half_saturation": 1.0, "min_points": 1}}

# Logged parameter leaf -> value every occurrence must equal. Hydra logs only the
# fields the configs set, so the code defaults above are not checked here.
PARAM_CHECKS = {
    "min_num_points": {2, 1, 0},
    "width_m": {3.0},
    "horizon_s": {4.0},
    "dt_s": {0.1},
    "max_lateral_accel_mps2": {3.0},
    "min_radius_m": {6.4},
    "max_speed_mps": {16.7},
    "wheel_base": {4.76},
    "margin": {-0.2},
}


def parse_value(text: str):
    """A logged parameter back into a Python value, the text itself if not literal."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def read_metrics(con: sqlite3.Connection, run_id: str) -> dict[str, float | None]:
    """Latest metrics of one run; NaN-flagged values become ``None``."""
    rows = con.execute(
        "SELECT key, value, is_nan FROM latest_metrics WHERE run_uuid = ?", (run_id,)
    ).fetchall()
    if not rows:
        raise ValueError(f"run {run_id!r} has no metrics.")
    return {key: (None if is_nan else value) for key, value, is_nan in rows}


def read_params(con: sqlite3.Connection, run_id: str) -> dict[str, object]:
    """Logged (resolved) configuration of one run, keyed by slash path."""
    rows = con.execute("SELECT key, value FROM params WHERE run_uuid = ?", (run_id,)).fetchall()
    if not rows:
        raise ValueError(f"run {run_id!r} has no parameters.")
    return {key: parse_value(value) for key, value in rows}


def read_info(con: sqlite3.Connection, run_id: str) -> dict[str, str]:
    """Config name, git sha, checkpoint, training run and date of one run."""
    tags = dict(con.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,)))
    start = con.execute("SELECT start_time FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
    if start is None:
        raise ValueError(f"run {run_id!r} not found.")
    for tag in ("config_name", "git_sha", "stage", "checkpoint_path", "source_run_id"):
        if tag not in tags:
            raise ValueError(f"run {run_id!r} is missing the {tag!r} tag.")
    if tags["stage"] != "test":
        raise ValueError(f"run {run_id!r} is a {tags['stage']!r} run, a test run is required.")
    return {
        "config_name": tags["config_name"],
        "git_sha": tags["git_sha"],
        "checkpoint": tags["checkpoint_path"],
        "train_run": tags["source_run_id"],
        "date": datetime.datetime.fromtimestamp(start[0] / 1000.0).strftime("%Y-%m-%d"),
    }


def bucketize(flat: dict[str, float | None], keep: tuple[str, ...]) -> dict[str, dict[str, float | None]]:
    """Group ``test/<suite>/<name>`` keys into report buckets via BUCKET_ALIAS."""
    out: dict[str, dict[str, float | None]] = {}
    for key, value in flat.items():
        if not key.startswith("test/"):
            continue
        rest = key[len("test/"):]
        head, sep, name = rest.partition("/")
        if not sep:
            continue
        two_segment = "/".join(rest.split("/")[:2])
        if two_segment in BUCKET_ALIAS and rest.count("/") >= 2:
            head, name = two_segment, rest[len(two_segment) + 1:]
        alias = BUCKET_ALIAS.get(head)
        if alias is None or alias not in keep:
            continue
        out.setdefault(alias, {})[name] = value
    missing = [bucket for bucket in keep if bucket not in out]
    if missing:
        raise ValueError(f"run carries no keys for buckets {missing}.")
    return out


def taxonomy_of(params: dict[str, object], task: str) -> dict:
    """Classes and behaviour groups of one task from the logged database taxonomy."""
    prefix = f"database/taxonomy/{task}/"
    classes = params.get(prefix + "class_names")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"run logs no class names under {prefix}class_names.")
    groups = {
        key[len(prefix + "class_groups/"):]: members
        for key, members in params.items()
        if key.startswith(prefix + "class_groups/")
    }
    if not groups:
        raise ValueError(f"run logs no class groups under {prefix}class_groups.")
    folded = sorted(member for members in groups.values() for member in members)
    if folded != sorted(classes):
        raise ValueError(f"{task} class groups do not partition the class names: {folded} vs {classes}.")
    return {"classes": list(classes), "groups": dict(sorted(groups.items()))}


def scenario_lists(params: dict[str, object]) -> dict[str, dict[str, list[str]]]:
    """Per dataset name, the ``<scenario_id>/<version>`` entries of every split."""
    lists: dict[str, dict[str, list[str]]] = {}
    for key, value in params.items():
        if not (key.startswith("database/scenarios/") and key.endswith("/dataset_name")):
            continue
        scenarios_key = key.split("/dataset_params/")[0]
        root = params[scenarios_key + "/scenario_root_path"]
        path = Path(str(root)) / f"{value}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"scenario list {path} of the run's database is missing.")
        with open(path, encoding="utf-8") as handle:
            splits = yaml.safe_load(handle)
        lists[str(value)] = {split: list(splits[split]) for split in ("train", "val", "test")}
    if not lists:
        raise ValueError("run logs no database scenario lists.")
    return lists


def source_params(params: dict[str, object], tasks: tuple[str, ...]) -> dict[str, object]:
    """The logged parameters of the test source that serves exactly ``tasks``.

    One run can read several sources, each bound to its own database, so a run
    that scores detection on one corpus and segmentation on another describes
    two test sets. The source whose task flags match is the one that describes
    this test set, and its keys are returned rooted at ``database/``.
    """
    matches = []
    index = 0
    while f"datamodule/test_sources/{index}/database/version" in params:
        prefix = f"datamodule/test_sources/{index}/"
        served = tuple(task for task in ("det3d", "seg3d") if params.get(prefix + task) is True)
        if served == tuple(tasks):
            matches.append(
                {key[len(prefix):]: value for key, value in params.items() if key.startswith(prefix)}
            )
        index += 1
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one of the run's {index} test sources to serve {list(tasks)}, "
            f"found {len(matches)}."
        )
    return matches[0]


def frames_scored(metrics: dict[str, dict[str, float | None]], task: str) -> int:
    """Number of frames the task's suite scored, from its own per-frame rate.

    Every suite reports a count beside the rate of the same quantity per frame,
    so their quotient is the frame count the suite divided by. Reading it back
    from the run needs no dataset next to the tracking database, and it states
    what was scored rather than what the test split nominally holds.
    """
    bucket_name, count_key, rate_key = FRAME_COUNTERS[task]
    bucket = metrics.get(bucket_name, {})
    count, rate = bucket.get(count_key), bucket.get(rate_key)
    if not count or not rate:
        raise ValueError(
            f"cannot read the {task} frame count, the run carries no non-zero "
            f"{bucket_name}/{count_key} and {bucket_name}/{rate_key}."
        )
    frames = count / rate
    if abs(frames - round(frames)) > 1e-3:
        raise ValueError(
            f"{bucket_name}/{count_key} over {rate_key} gives {frames} frames, not a whole number."
        )
    return int(round(frames))


def dataset_stats(
    source: dict[str, object], metrics: dict[str, dict[str, float | None]], tasks: tuple[str, ...]
) -> dict:
    """Frames scored, scenes and per dataset scene lists of the test split."""
    lists = scenario_lists(source)
    test_ids = {entry.split("/")[0] for splits in lists.values() for entry in splits["test"]}
    frames = {task: frames_scored(metrics, task) for task in tasks}
    if len(set(frames.values())) != 1:
        raise ValueError(f"the tasks of this test set scored different frame counts: {frames}.")
    sources = [
        {"db": name, "scenes": sorted(f"{name}/{entry}" for entry in splits["test"])}
        for name, splits in sorted(lists.items())
    ]
    return {
        "database_version": str(source["database/version"]),
        "num_frames": next(iter(frames.values())),
        "num_scenes": len(test_ids),
        "sources": sources,
    }


def det_distribution(bucket: dict[str, float | None], classes: list[str]) -> tuple[int, list[dict]]:
    """GT objects per class from the run's own ``gt_count_*`` keys."""
    bars = []
    for name in classes:
        value = bucket.get(f"gt_count_{name}")
        if value is None:
            raise ValueError(f"det bucket carries no gt_count_{name}.")
        bars.append({"label": name, "value": int(value)})
    return sum(bar["value"] for bar in bars), bars


def seg_distribution(bucket: dict[str, float | None], classes: list[str]) -> tuple[int, list[dict]]:
    """Valid labeled points per class from the confusion-matrix row sums."""
    bars = []
    for true_name in classes:
        total = 0.0
        for pred_name in classes:
            value = bucket.get(f"confusion_{true_name}__{pred_name}")
            if value is not None:
                total += value
        bars.append({"label": true_name, "value": int(total)})
    if not any(bar["value"] for bar in bars):
        raise ValueError("seg bucket carries no confusion cells.")
    return sum(bar["value"] for bar in bars), bars


def check_spec_against_params(params: dict[str, object]) -> None:
    """Every logged occurrence of a checked leaf must be one of its allowed values."""
    for leaf, allowed in PARAM_CHECKS.items():
        seen = {value for key, value in params.items() if key.startswith("model/metrics/") and key.endswith("/" + leaf)}
        unexpected = seen - allowed
        if unexpected:
            raise ValueError(f"run logs {leaf}={sorted(unexpected)}, the spec constants allow {sorted(allowed)}.")


def metric_spec(tasks: tuple[str, ...], params: dict[str, object]) -> dict:
    metrics: dict[str, dict] = {}
    if "det3d" in tasks:
        metrics.update(DET_SPEC)
    if "seg3d" in tasks:
        metrics.update(SEG_SPEC)
    if tasks == ("det3d", "seg3d"):
        box_classes = params.get("database/taxonomy/detection3d/partial_detection_classes")
        if not isinstance(box_classes, list):
            raise ValueError("run logs no partial_detection_classes in the detection taxonomy.")
        metrics["partial_detection"] = {
            "version": PARTIAL_DETECTION_SPEC["version"],
            "params": {**PARTIAL_DETECTION_SPEC["params"], "box_classes": list(box_classes)},
        }
    return {"bundle": BUNDLE, "metrics": metrics}


def build_test_set(
    con: sqlite3.Connection, test_set_id: str, run_id: str, release: str, model: str,
    test_set_version: str, note: str | None,
) -> dict:
    definition = TEST_SETS[test_set_id]
    info = read_info(con, run_id)
    params = read_params(con, run_id)
    check_spec_against_params(params)
    metrics = bucketize(read_metrics(con, run_id), definition.buckets)
    source = source_params(params, definition.tasks)
    taxonomy = {task: taxonomy_of(source, {"det3d": "detection3d", "seg3d": "segmentation3d"}[task])
                for task in definition.tasks}

    dataset = dataset_stats(source, metrics, definition.tasks)
    distributions = []
    if "det3d" in definition.tasks:
        num_objects, bars = det_distribution(metrics["det3d"], taxonomy["det3d"]["classes"])
        dataset["num_objects"] = num_objects
        distributions.append({"title": "GT objects per class (eval set: at least 2 lidar points, up to 121 m)",
                              "unit": "objects", "bars": bars})
    if "seg3d" in definition.tasks:
        num_points, bars = seg_distribution(metrics["seg3d"], taxonomy["seg3d"]["classes"])
        dataset["num_points"] = num_points
        distributions.append({"title": "Labeled points per class (valid, whole scene)", "unit": "points", "bars": bars})
    dataset["distributions"] = distributions

    spec = metric_spec(definition.tasks, params)
    entry = {
        "name": release,
        "model": model,
        "date": info["date"],
        "test_set_version": test_set_version,
        "metric_spec_bundle": spec["bundle"],
        "repo_commit": info["git_sha"],
        "eval_config": info["config_name"],
        "checkpoint": info["checkpoint"],
        "train_run": info["train_run"],
        "run_id": run_id,
        "metrics": metrics,
    }
    test_set = {
        "id": test_set_id,
        "label": f"{test_set_id}-v{test_set_version}",
        "version": test_set_version,
        "tasks": list(definition.tasks),
        "taxonomy": taxonomy,
        "dataset": dataset,
        "metric_spec": spec,
        "models": [entry],
    }
    if note:
        test_set["notes"] = note
    return test_set


def parse_assignments(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        test_set_id, sep, value = item.partition("=")
        if not sep or test_set_id not in TEST_SETS:
            raise ValueError(f"{what} must be <test-set-id>=<value> with an id in {sorted(TEST_SETS)}, got {item!r}.")
        if test_set_id in out:
            raise ValueError(f"{what} names test set {test_set_id!r} twice.")
        out[test_set_id] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="MLflow SQLite DB path.")
    parser.add_argument("--release", required=True, help="Release name shown in the report, e.g. v0.1.0.")
    parser.add_argument("--model", required=True, help="Model family, drives the chart colour, e.g. 'LitePT multi-head'.")
    parser.add_argument("--run", action="append", required=True, metavar="TEST_SET=RUN_ID",
                        help="Test run evaluated on the named test set; repeat per test set.")
    parser.add_argument("--test-set-version", default=None,
                        help="Version stamped on every test set (default: the release number without its v).")
    parser.add_argument("--note", action="append", default=[], metavar="TEST_SET=TEXT",
                        help="Free-text note shown in the report's test-set section.")
    parser.add_argument("--out", required=True, help="Output release JSON path.")
    args = parser.parse_args(argv)

    runs = parse_assignments(args.run, "--run")
    notes = parse_assignments(args.note, "--note")
    version = args.test_set_version or args.release.lstrip("v")

    con = sqlite3.connect(args.db)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "repo_url": REPO_URL,
        "test_sets": [
            build_test_set(con, test_set_id, run_id, args.release, args.model, version, notes.get(test_set_id))
            for test_set_id, run_id in runs.items()
        ],
    }
    con.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = ", ".join(
        f"{ts['id']}({ts['dataset']['num_frames']}f/{ts['dataset']['num_scenes']}s, "
        f"{sum(len(b) for b in ts['models'][0]['metrics'].values())} keys)"
        for ts in doc["test_sets"]
    )
    print(f"wrote {out_path}: {summary}")


if __name__ == "__main__":
    main()

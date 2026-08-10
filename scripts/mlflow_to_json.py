#!/usr/bin/env python
"""Turn PTv3 multi-head MLflow test runs into a report-data release JSON.

Reads two test runs from an MLflow SQLite DB and emits one release file
(schema/report_data.schema.json):

* ``--split-run`` — the run evaluated on two separate datasets; its ``det3d*``
  buckets become the ``det3d`` test set and its ``seg3d*`` buckets the ``seg3d``
  test set.
* ``--joint-run`` — the run evaluated on the joint seg+det dataset; all its
  buckets become the ``seg3d-det3d`` test set.

Class distributions come from the runs themselves (detection ``gt_count_*``
keys = the eval GT set; segmentation confusion-matrix row sums = valid labeled
points), so the report describes exactly what was evaluated. Frame/scene
counts and per-DB scene lists come from the runs' test info pkls.

    python scripts/mlflow_to_json.py \
        --db tmp/mlruns-report-v0.1/mlruns/mlflow.db \
        --split-run 011f8a8bb536421f9191b7ac15ae5fc4 \
        --joint-run 22e3437fd9e64ce8a80dd6b017a7b775 \
        --det-pkl data/t4dataset/info/lidarseg_pseudo_new/t4dataset_j6gen2_base_infos_test.pkl \
        --seg-pkl data/t4dataset/info/lidarseg/t4dataset_j6gen2_lidarseg_infos_test.pkl \
        --joint-pkl data/t4dataset/info/segdet3d/t4dataset_j6gen2_segdet3d_infos_test.pkl \
        --out data/ptv3_v0.0.1.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path

DET_CLASSES = [
    "car", "truck", "bus", "train", "motorcycle", "bicycle", "pedestrian",
    "animal", "barrier", "traffic_cone", "debris", "bicycle_rack", "vehicle_extension",
]
SEG_CLASSES = DET_CLASSES + [
    "drivable_flat", "non_drivable_flat", "vegetation", "building",
    "vertical_thin", "static_clutter", "noise",
]

DET_BUCKETS = ("det3d", "det3d_grouped", "det3d_visible", "det3d_overall")
SEG_BUCKETS = ("seg3d", "seg3d_grouped")
BUCKET_ALIAS = {
    "det3d": "det3d", "det3d_grouped": "det3d_grouped",
    "det3d_visible": "det3d_visible", "det3d_overall": "det3d_overall",
    "seg3d": "seg3d", "seg3d_pt": "seg3d",
    "seg3d_grouped": "seg3d_grouped",
    # the merged point suite emits its behaviour-group view under grouped/
    "seg3d_pt/grouped": "seg3d_grouped",
}

# Spec-versioned metric parameters mirroring the shipped evaluation configs.
DET_SPEC = {"bundle": "0.1.0", "metrics": {
    "center_distance_matching": {"version": "0.1", "params": {"thresholds_m": [0.5, 1.0, 2.0, 4.0]}},
    "gt_filter": {"version": "0.1", "params": {"min_num_points": 2, "eval_class_range_m": 121.0}},
    "corner_error": {"version": "0.1", "params": {"tp_threshold_m": 2.0}},
    "critical_fp_fn": {"version": "0.1", "params": {"match_threshold_m": 2.0, "confidences": [0.3, 0.5],
        "horizon_s": 4.0}},
    "collision_weighted_map": {"version": "0.1", "params": {"lambda": 0.5, "horizon_s": 4.0, "dt_s": 0.1,
        "max_lateral_accel_mps2": 3.0}},
    "calibration_error": {"version": "0.1", "params": {"num_bins": 15}},
    "confusion_matrix": {"version": "0.1", "params": {"match_threshold_m": 2.0, "min_score": 0.1,
        "matched_only": True}}}}
SEG_SPEC = {"bundle": "0.1.0", "metrics": {
    "error_clusters": {"version": "0.1", "params": {"cluster_radius_m": 0.5, "min_cluster_points": 1}},
    "tolerant_error": {"version": "0.1", "params": {"radius_m": 0.2}},
    "calibration_error": {"version": "0.1", "params": {"num_bins": 15, "per_class": "predicted"}},
    "entropy_auroc": {"version": "0.1", "params": {"num_bins": 8192}},
    "confident_error": {"version": "0.1", "params": {"entropy_threshold": 0.3}},
    "region_filter": {"version": "0.1", "params": {"road_margin_m": 0.2,
        "road": ["road", "road_shoulder", "crosswalk", "drivable_area", "intersection_area", "crosswalk_polygon"]}}}}
BOTH_SPEC = {"bundle": "0.1.0", "metrics": {**DET_SPEC["metrics"], **SEG_SPEC["metrics"]}}


def read_run(db_path: str, run_id: str) -> dict[str, float | None]:
    """Latest metrics of one run; NaN-flagged values become ``None``."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT key, value, is_nan FROM latest_metrics WHERE run_uuid = ?", (run_id,)
    ).fetchall()
    con.close()
    if not rows:
        raise ValueError(f"run {run_id!r} has no metrics in {db_path!r}.")
    return {key: (None if is_nan else value) for key, value, is_nan in rows}


def run_info(db_path: str, run_id: str) -> dict[str, str]:
    """Config name, git sha and date of one run (fail-loud on missing tags)."""
    con = sqlite3.connect(db_path)
    tags = dict(con.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (run_id,)))
    start = con.execute("SELECT start_time FROM runs WHERE run_uuid = ?", (run_id,)).fetchone()
    con.close()
    if start is None:
        raise ValueError(f"run {run_id!r} not found in {db_path!r}.")
    for tag in ("config_name", "git_sha"):
        if tag not in tags:
            raise ValueError(f"run {run_id!r} is missing the {tag!r} tag.")
    date = datetime.datetime.fromtimestamp(start[0] / 1000.0).strftime("%Y-%m-%d")
    return {"config_name": tags["config_name"], "git_sha": tags["git_sha"], "date": date}


def bucketize(flat: dict[str, float | None], keep: tuple[str, ...], stage: str = "test") -> dict:
    """Group ``stage/<suite>/<name>`` keys into report buckets via BUCKET_ALIAS."""
    out: dict[str, dict[str, float | None]] = {}
    prefix = f"{stage}/"
    for key, value in flat.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        head, sep, name = rest.partition("/")
        # Longest-prefix wins: a suite's grouped/ sub-namespace maps to its own bucket.
        two_segment = "/".join(rest.split("/")[:2])
        if two_segment in BUCKET_ALIAS and rest.count("/") >= 2:
            head, name = two_segment, rest[len(two_segment) + 1:]
        alias = BUCKET_ALIAS.get(head)
        if not sep or alias is None or alias not in keep:
            continue
        out.setdefault(alias, {})[name] = value
    missing = [bucket for bucket in keep if bucket not in out]
    if missing:
        raise ValueError(f"run carries no keys for buckets {missing}.")
    return out


def scene_fragment(lidar_path: str) -> str:
    """``<db>/<scene_uuid>/<version>`` from a relative lidar path."""
    parts = Path(lidar_path).parts
    if len(parts) < 4:
        raise ValueError(f"lidar path too short for a scene fragment: {lidar_path!r}")
    return "/".join(parts[:3])


def dataset_stats(pkl_path: str) -> dict:
    """Frames, scenes and per-DB scene lists of one test info pkl."""
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)
    frames = data["data_list"]
    scenes_by_db: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        fragment = scene_fragment(frame["lidar_points"]["lidar_path"])
        scenes_by_db[fragment.split("/")[0]].add(fragment)
    sources = [
        {"db": db, "scenes": sorted(scenes)}
        for db, scenes in sorted(scenes_by_db.items())
    ]
    return {
        "num_frames": len(frames),
        "num_scenes": sum(len(s["scenes"]) for s in sources),
        "sources": sources,
    }


def det_distribution(bucket: dict[str, float | None]) -> tuple[int, list[dict]]:
    """GT objects per class from the run's own ``gt_count_*`` keys."""
    bars = []
    for name in DET_CLASSES:
        value = bucket.get(f"gt_count_{name}")
        if value is None:
            raise ValueError(f"det bucket carries no gt_count_{name}.")
        bars.append({"label": name, "value": int(value)})
    return sum(bar["value"] for bar in bars), bars


def seg_distribution(bucket: dict[str, float | None]) -> list[dict]:
    """Valid labeled points per class from the confusion-matrix row sums."""
    bars = []
    for true_name in SEG_CLASSES:
        total = 0.0
        for pred_name in SEG_CLASSES:
            value = bucket.get(f"confusion_{true_name}__{pred_name}")
            if value is not None:
                total += value
        bars.append({"label": true_name, "value": int(total)})
    if not any(bar["value"] for bar in bars):
        raise ValueError("seg bucket carries no confusion cells.")
    return bars


def build_dataset(pkl_path: str, det_bucket: dict | None, seg_bucket: dict | None) -> dict:
    dataset = dataset_stats(pkl_path)
    distributions = []
    if det_bucket is not None:
        num_objects, bars = det_distribution(det_bucket)
        dataset["num_objects"] = num_objects
        distributions.append({
            "title": "GT objects per class (eval set: ≥2 lidar points, ≤121 m)",
            "unit": "objects",
            "bars": bars,
        })
    if seg_bucket is not None:
        distributions.append({
            "title": "Labeled points per class (valid, whole scene)",
            "unit": "points",
            "bars": seg_distribution(seg_bucket),
        })
    dataset["distributions"] = distributions
    return dataset


def model_entry(info: dict[str, str], version_name: str, metrics: dict, test_set_version: str) -> dict:
    return {
        "name": version_name,
        "model": "PTv3 multi-head",
        "date": info["date"],
        "test_set_version": test_set_version,
        "repo_commit": info["git_sha"],
        "eval_config": info["config_name"],
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="MLflow SQLite DB path.")
    parser.add_argument("--split-run", required=True, help="Test run with separate det + seg datasets.")
    parser.add_argument("--joint-run", required=True, help="Test run on the joint seg+det dataset.")
    parser.add_argument("--det-pkl", required=True, help="Det test info pkl of the split run.")
    parser.add_argument("--seg-pkl", required=True, help="Seg test info pkl of the split run.")
    parser.add_argument("--joint-pkl", required=True, help="Joint test info pkl of the joint run.")
    parser.add_argument("--out", required=True, help="Output release JSON path.")
    parser.add_argument("--release", default="v0.0.1", help="Release name shown in the report.")
    args = parser.parse_args(argv)

    split_info = run_info(args.db, args.split_run)
    joint_info = run_info(args.db, args.joint_run)
    split = bucketize(read_run(args.db, args.split_run), DET_BUCKETS + SEG_BUCKETS)
    joint = bucketize(read_run(args.db, args.joint_run), DET_BUCKETS + SEG_BUCKETS)

    det_metrics = {bucket: split[bucket] for bucket in DET_BUCKETS}
    seg_metrics = {bucket: split[bucket] for bucket in SEG_BUCKETS}

    doc = {
        "schema_version": "0.1",
        "repo_url": "https://github.com/tier4/autoware-ml",
        "test_sets": [
            {
                "id": "det3d-j6gen2", "label": "det3d-j6gen2-v0.0.1", "version": "0.0.1",
                "tasks": ["det3d"],
                "dataset": build_dataset(args.det_pkl, split["det3d"], None),
                "metric_spec": DET_SPEC,
                "models": [model_entry(split_info, args.release, det_metrics, "0.0.1")],
            },
            {
                "id": "seg3d-j6gen2", "label": "seg3d-j6gen2-v0.0.1", "version": "0.0.1",
                "tasks": ["seg3d"],
                "dataset": build_dataset(args.seg_pkl, None, split["seg3d"]),
                "metric_spec": SEG_SPEC,
                "models": [model_entry(split_info, args.release, seg_metrics, "0.0.1")],
            },
            {
                "id": "det3d-seg3d-j6gen2", "label": "det3d-seg3d-j6gen2-v0.0.1", "version": "0.0.1",
                "tasks": ["det3d", "seg3d"],
                "dataset": build_dataset(args.joint_pkl, joint["det3d"], joint["seg3d"]),
                "metric_spec": BOTH_SPEC,
                "models": [model_entry(joint_info, args.release, joint, "0.0.1")],
            },
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = ", ".join(
        f"{ts['id']}({ts['dataset']['num_frames']}f/{ts['dataset']['num_scenes']}s)"
        for ts in doc["test_sets"]
    )
    print(f"wrote {out_path}: {summary}")


if __name__ == "__main__":
    main()

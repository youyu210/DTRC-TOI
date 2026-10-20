"""Summarize the B1 profile without double-counting nested converter time."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/quality-cost-v3/b1-profile-v1"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rms(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    difference = a - b
    return {
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def pearson(x: list[float], y: list[float]) -> float | None:
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    profile_path = DATA / "runs/current_profile/r01"
    profile_frames = rows(profile_path / "frames.csv")
    profiles = [json.loads(line) for line in
                (profile_path / "stage_profile.jsonl").read_text().splitlines()]
    total_ms = sum(float(row["step_ms"]) for row in profile_frames)
    newton = sum(int(row["newton_iterations"]) for row in profile_frames)

    # These are exclusive leaves. linear_system.convert is intentionally
    # replaced by its converter children so nested time is counted once.
    leaves: list[tuple[str, str, str, str]] = []
    for key in profiles[0]["gradient_hessian"]:
        if key not in ("inclusive_ms", "subprofile_ms", "unattributed_ms"):
            leaves.append((f"gradient_hessian.{key}", "gradient_hessian", key,
                           "gradient_hessian_calls"))
    for key in profiles[0]["linear_system"]:
        if key not in ("inclusive_ms", "subprofile_ms", "unattributed_ms",
                       "convert"):
            leaves.append((f"linear_system.{key}", "linear_system", key,
                           "linear_build_calls"))
    for key in profiles[0]["converter"]:
        if key not in ("inclusive_ms", "subprofile_ms", "unattributed_ms"):
            leaves.append((f"converter.{key}", "converter", key,
                           "convert_calls"))

    contact_per_gradient = [
        row["counts"]["contact_triplets"]
        / row["counts"]["gradient_hessian_calls"] for row in profiles
    ]
    operations = []
    for name, section, key, call_counter in leaves:
        elapsed = sum(float(row[section][key]) for row in profiles)
        calls = sum(int(row["counts"][call_counter]) for row in profiles)
        per_frame_per_call = [
            float(row[section][key]) / row["counts"][call_counter]
            for row in profiles
        ]
        operations.append({
            "operation": name,
            "elapsed_ms": elapsed,
            "step_share": elapsed / total_ms,
            "ms_per_newton": elapsed / newton,
            "calls": calls,
            "calls_per_newton": calls / newton,
            "ms_per_call": elapsed / calls,
            "contact_triplets_per_call_time_pearson":
                pearson(contact_per_gradient, per_frame_per_call),
        })
    operations.sort(key=lambda row: row["elapsed_ms"], reverse=True)

    counts = {
        key: sum(int(row["counts"][key]) for row in profiles)
        for key in profiles[0]["counts"]
    }
    scales = {
        "contact_triplets_per_gradient_call":
            counts["contact_triplets"] / counts["gradient_hessian_calls"],
        "total_triplets_per_gradient_call":
            counts["total_triplets"] / counts["gradient_hessian_calls"],
        "convert_input_triplets_per_call":
            counts["convert_input_triplets"] / counts["convert_calls"],
        "unique_blocks_per_convert_call":
            counts["unique_blocks"] / counts["convert_calls"],
        "rhs_dofs_per_linear_build":
            counts["rhs_dofs"] / counts["linear_build_calls"],
        "tri_edges_per_gradient_call":
            counts["tri_edges"] / counts["gradient_hessian_calls"],
        "triangles_per_gradient_call":
            counts["triangles"] / counts["gradient_hessian_calls"],
        "zero_contact_frames": sum(value == 0 for value in contact_per_gradient),
        "contact_triplets_per_gradient_call_min": min(contact_per_gradient),
        "contact_triplets_per_gradient_call_median":
            statistics.median(contact_per_gradient),
        "contact_triplets_per_gradient_call_max": max(contact_per_gradient),
    }

    broad_names = ("gradient_hessian_ms", "residual_check_ms",
                   "linear_system_ms", "ccd_ms", "line_search_ms",
                   "graph_prepare_ms")
    broad = {
        name: {
            "elapsed_ms": (elapsed := sum(float(row[name])
                                          for row in profile_frames)),
            "step_share": elapsed / total_ms,
        }
        for name in broad_names
    }
    broad_sum = sum(item["elapsed_ms"] for item in broad.values())

    timing: dict[str, object] = {}
    for version in ("current_off", "frozen_v2"):
        runs = []
        for repeat in range(1, 4):
            path = DATA / "runs" / version / f"r{repeat:02d}"
            frame_rows = rows(path / "frames.csv")
            elapsed = sum(float(row["step_ms"]) for row in frame_rows) / 1000
            iterations = sum(int(row["newton_iterations"])
                             for row in frame_rows)
            pcg = sum(int(row["pcg_iterations"]) for row in frame_rows)
            runs.append({
                "repeat": repeat,
                "seconds": elapsed,
                "newton": iterations,
                "pcg": pcg,
                "ms_per_newton": elapsed * 1000 / iterations,
            })
        timing[version] = {
            "runs": runs,
            "median_seconds": statistics.median(row["seconds"] for row in runs),
            "median_ms_per_newton":
                statistics.median(row["ms_per_newton"] for row in runs),
            "median_newton": statistics.median(row["newton"] for row in runs),
            "median_pcg": statistics.median(row["pcg"] for row in runs),
        }
    current = timing["current_off"]
    frozen = timing["frozen_v2"]
    comparisons = {
        "median_wall_speedup":
            frozen["median_seconds"] / current["median_seconds"],
        "median_per_newton_speedup":
            frozen["median_ms_per_newton"] / current["median_ms_per_newton"],
        "paired_wall_change_vs_frozen": [
            (old["seconds"] - new["seconds"]) / old["seconds"]
            for new, old in zip(current["runs"], frozen["runs"], strict=True)
        ],
    }

    numerical: dict[str, object] = {}
    for filename in ("final.f64", "final_velocities.f64"):
        values = {
            version: [np.fromfile(DATA / "runs" / version / f"r{repeat:02d}"
                                  / filename, dtype="<f8")
                      for repeat in range(1, 4)]
            for version in ("current_off", "frozen_v2")
        }
        numerical[filename] = {
            "current_self": [rms(values["current_off"][a],
                                 values["current_off"][b])
                             for a, b in itertools.combinations(range(3), 2)],
            "frozen_self": [rms(values["frozen_v2"][a],
                                values["frozen_v2"][b])
                            for a, b in itertools.combinations(range(3), 2)],
            "paired": [rms(values["current_off"][index],
                           values["frozen_v2"][index])
                       for index in range(3)],
        }

    analysis = {
        "profile": {
            "frames": len(profile_frames),
            "newton": newton,
            "total_step_ms": total_ms,
            "broad_stages": broad,
            "broad_profiled_ms": broad_sum,
            "broad_unattributed_ms": total_ms - broad_sum,
            "exclusive_operations": operations,
            "top_three": operations[:3],
            "counts": counts,
            "scales": scales,
        },
        "unprofiled_timing": timing,
        "comparison": comparisons,
        "numerical": numerical,
    }
    (DATA / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    evidence: dict[str, str] = {}
    for path in sorted((DATA / "runs").rglob("*")):
        if path.is_file():
            evidence[path.relative_to(ROOT).as_posix()] = sha256(path)
    for path in (DATA / "MATRIX_COMPLETE.json", DATA / "analysis.json"):
        evidence[path.relative_to(ROOT).as_posix()] = sha256(path)
    (DATA / "EVIDENCE_MANIFEST.json").write_text(
        json.dumps({"sha256": evidence}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()

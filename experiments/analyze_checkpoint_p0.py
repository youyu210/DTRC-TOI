"""Analyze frame-boundary checkpoint repeatability without modifying run data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vector_error(lhs: Path, rhs: Path) -> dict[str, float]:
    a = np.fromfile(lhs, dtype="<f8").reshape(-1, 3)
    b = np.fromfile(rhs, dtype="<f8").reshape(-1, 3)
    if a.shape != b.shape:
        raise ValueError(f"State shape mismatch: {lhs} {a.shape}, {rhs} {b.shape}")
    distance = np.linalg.norm(a - b, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(distance * distance))),
        "max": float(distance.max(initial=0.0)),
    }


def row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"Expected one restored frame: {path}")
    return rows[0]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def trace_error(candidate: Path, reference: Path) -> dict[str, float]:
    metadata = json.loads((candidate / "metadata.json").read_text())
    other = json.loads((reference / "metadata.json").read_text())
    if (metadata["frames"], metadata["vertex_count"]) != (
        other["frames"], other["vertex_count"]
    ):
        raise ValueError("Compatibility trace shape mismatch")
    config = json.loads((candidate / "cloth_scene.json").read_text())
    boundary = np.fromfile(candidate / "boundary.i32", dtype="<i4")
    mask = (np.asarray(config["roles"]) > 0) & (boundary == 0)
    frames = metadata["frames"]
    vertices = metadata["vertex_count"]
    result: dict[str, float] = {"free_cloth_vertices": int(mask.sum())}
    for filename, label in (("states.f64", "position"),
                            ("velocities.f64", "velocity")):
        a = np.memmap(candidate / filename, dtype="<f8", mode="r",
                      shape=(frames, vertices, 3))
        b = np.memmap(reference / filename, dtype="<f8", mode="r",
                      shape=(frames, vertices, 3))
        rms = []
        maximum = []
        for lhs, rhs in zip(a, b):
            distance = np.linalg.norm(lhs[mask] - rhs[mask], axis=1)
            rms.append(float(np.sqrt(np.mean(distance * distance))))
            maximum.append(float(distance.max(initial=0.0)))
        result[f"{label}_max_frame_rms"] = max(rms)
        result[f"{label}_final_rms"] = rms[-1]
        result[f"{label}_max"] = max(maximum)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    continuous = root / "continuous"
    restores = sorted(root.glob("restored_r*"))
    if len(restores) < 2:
        raise ValueError("At least two restored runs are required")

    checkpoint = root / "checkpoint_f1"
    roundtrip = root / "checkpoint_f1_roundtrip"
    checkpoint_hashes = {
        path.name: sha256(path) for path in sorted(checkpoint.iterdir()) if path.is_file()
    }
    roundtrip_hashes = {
        path.name: sha256(path) for path in sorted(roundtrip.iterdir()) if path.is_file()
    }
    mismatches = sorted(
        name
        for name in checkpoint_hashes.keys() | roundtrip_hashes.keys()
        if checkpoint_hashes.get(name) != roundtrip_hashes.get(name)
    )

    fields = {
        "position": "final.f64",
        "velocity": "final_velocities.f64",
    }
    continuation: dict[str, list[dict[str, object]]] = {}
    repeat_pairs: dict[str, list[dict[str, object]]] = {}
    for label, filename in fields.items():
        continuation[label] = [
            {"run": run.name, **vector_error(continuous / filename, run / filename)}
            for run in restores
        ]
        repeat_pairs[label] = [
            {
                "lhs": lhs.name,
                "rhs": rhs.name,
                **vector_error(lhs / filename, rhs / filename),
            }
            for lhs, rhs in itertools.combinations(restores, 2)
        ]

    frame_rows = {run.name: row(run / "frames.csv") for run in restores}
    control = {
        run: {
            "newton_iterations": int(data["newton_iterations"]),
            "pcg_iterations": int(data["pcg_iterations"]),
            "exit_reason": data["exit_reason"],
            "nonfinite_vertices": int(data["nonfinite_vertices"]),
            "newton_cap_hits": int(data["newton_cap_hits"]),
        }
        for run, data in frame_rows.items()
    }
    result = {
        "checkpoint_schema": json.loads((checkpoint / "manifest.json").read_text())["schema"],
        "checkpoint_roundtrip_mismatches": mismatches,
        "restored_initial_matches_checkpoint_vertices": all(
            sha256(run / "initial.f64") == sha256(checkpoint / "fem_vertexes.bin")
            for run in restores
        ),
        "continuation_vs_continuous": continuation,
        "restored_repeat_pairs": repeat_pairs,
        "control_flow": control,
    }
    corrupt_code_file = root / "corrupt_rejection.exitcode.txt"
    result["corrupt_checkpoint_rejected"] = (
        corrupt_code_file.exists()
        and int(corrupt_code_file.read_text().strip()) != 0
        and "Checkpoint size mismatch"
        in (root / "corrupt_rejection.stderr.log").read_text()
    )

    repository = root.parents[2]
    historical = json.loads(
        (repository / "experiments/dual-tolerance-v2/summary.json").read_text()
    )["original_graph_repeat_difference"]
    compatibility = {}
    rhs_audits = {}
    for case in ("table", "bunny"):
        frozen = root / "compat" / case / "frozen_v2"
        current = root / "compat" / case / "checkpoint_v3"
        error = trace_error(current, frozen)
        position_noise = max(
            item["position_max_frame_rms"] for item in historical[case]
        )
        velocity_noise = max(
            item["velocity_max_frame_rms"] for item in historical[case]
        )
        error.update({
            "historical_position_repeat_noise": position_noise,
            "historical_velocity_repeat_noise": velocity_noise,
            "within_historical_repeat_noise": (
                error["position_max_frame_rms"] <= position_noise
                and error["velocity_max_frame_rms"] <= velocity_noise
            ),
            "initial_state_equal": (
                sha256(current / "initial.f64") == sha256(frozen / "initial.f64")
            ),
            "topology_equal": (
                sha256(current / "scene_triangles.u32")
                == sha256(frozen / "scene_triangles.u32")
            ),
        })
        frozen_metadata = json.loads((frozen / "metadata.json").read_text())
        current_metadata = json.loads((current / "metadata.json").read_text())
        frozen_rows = rows(frozen / "frames.csv")
        current_rows = rows(current / "frames.csv")
        error.update({
            "tolerances_equal": all(
                metadata["newton_tolerance"] == 0.001
                and metadata["pcg_tolerance"] == 0.001
                for metadata in (frozen_metadata, current_metadata)
            ),
            "all_original_exits": all(
                item["exit_reason"] == "original_direction"
                for item in frozen_rows + current_rows
            ),
            "cold_graph_captures": {
                "frozen_v2": int(frozen_rows[0]["captures"]),
                "checkpoint_v3": int(current_rows[0]["captures"]),
            },
            "hot_graph_captures": {
                "frozen_v2": sum(int(item["captures"]) for item in frozen_rows[1:]),
                "checkpoint_v3": sum(int(item["captures"]) for item in current_rows[1:]),
            },
        })
        compatibility[case] = error
        rhs_audits[case] = {}
        for variant in ("frozen_v2", "checkpoint_v3"):
            audit_path = root / "compat_audit" / case / variant / "audits.json"
            audits = json.loads(audit_path.read_text())
            rhs_audits[case][variant] = {
                "samples": len(audits),
                "max_relative_l2_direction_diff": max(
                    item["relative_l2_direction_diff"] for item in audits
                ),
                "iteration_counts_agree": all(
                    item["actual_iterations"] == item["eager_iterations"]
                    for item in audits
                ),
                "passed": bool(audits) and all(
                    item["actual_iterations"] == item["eager_iterations"]
                    and item["relative_l2_direction_diff"] < 1e-6
                    for item in audits
                ),
            }
    result["default_path_compatibility"] = compatibility
    result["native_rhs_audits"] = rhs_audits
    for label in fields:
        result[f"{label}_continuation_max_rms"] = max(
            item["rms"] for item in continuation[label]
        )
        result[f"{label}_repeat_noise_max_rms"] = max(
            item["rms"] for item in repeat_pairs[label]
        )
        result[f"{label}_continuation_within_repeat_noise"] = (
            result[f"{label}_continuation_max_rms"]
            <= result[f"{label}_repeat_noise_max_rms"]
        )
    result["passed"] = (
        not mismatches
        and result["restored_initial_matches_checkpoint_vertices"]
        and result["corrupt_checkpoint_rejected"]
        and result["position_continuation_within_repeat_noise"]
        and result["velocity_continuation_within_repeat_noise"]
        and all(
            item["within_historical_repeat_noise"]
            and item["initial_state_equal"]
            and item["topology_equal"]
            and item["tolerances_equal"]
            and item["all_original_exits"]
            and item["cold_graph_captures"]["frozen_v2"] == 2
            and item["cold_graph_captures"]["checkpoint_v3"] == 2
            and item["hot_graph_captures"]["frozen_v2"] == 0
            and item["hot_graph_captures"]["checkpoint_v3"] == 0
            for item in compatibility.values()
        )
        and all(
            audit["passed"]
            for case_audits in rhs_audits.values()
            for audit in case_audits.values()
        )
        and all(
            data["newton_iterations"] == 4
            and data["pcg_iterations"] in (63, 64)
            and data["exit_reason"] == "original_direction"
            and data["nonfinite_vertices"] == 0
            and data["newton_cap_hits"] == 0
            for data in control.values()
        )
    )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.write:
        (root / "analysis.json").write_text(rendered + "\n", encoding="utf-8")
        lines = [
            "# P0 checkpoint validation",
            "",
            f"- Schema: `{result['checkpoint_schema']}`",
            f"- Byte-identical load/save roundtrip: `{not mismatches}`",
            f"- Restored initial state equals checkpoint state: `{result['restored_initial_matches_checkpoint_vertices']}`",
            f"- Truncated checkpoint rejected: `{result['corrupt_checkpoint_rejected']}`",
            f"- Position continuation max RMS: `{result['position_continuation_max_rms']:.17g}` m",
            f"- Position repeat-noise max RMS: `{result['position_repeat_noise_max_rms']:.17g}` m",
            f"- Velocity continuation max RMS: `{result['velocity_continuation_max_rms']:.17g}` m/s",
            f"- Velocity repeat-noise max RMS: `{result['velocity_repeat_noise_max_rms']:.17g}` m/s",
            f"- Default path compatible with frozen V2 on table/bunny: `{all(item['within_historical_repeat_noise'] for item in compatibility.values())}`",
            f"- Native RHS audits passed: `{all(audit['passed'] for case_audits in rhs_audits.values() for audit in case_audits.values())}`",
            f"- P0 numerical/repeatability checks passed: `{result['passed']}`",
            "",
            "See `analysis.json` for all pairwise errors and per-run control-flow counts.",
        ]
        (root / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

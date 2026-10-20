"""Run and freeze the preregistered A2 representative-state selection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"
OUT = ROOT / "experiments/quality-cost-v3/a2-selection-v1"
SCENES = ("table", "bunny")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def validate_run(path: Path) -> list[dict[str, object]]:
    metadata = json.loads((path / "metadata.json").read_text())
    status = json.loads((path / "status.json").read_text())
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    traces = [json.loads(line) for line in
              (path / "newton_trace.jsonl").read_text().splitlines()]
    assert status == {"completed": True, "frames_completed": 50,
                      "finite_state": True}
    assert metadata["termination"] == "original"
    assert metadata["newton_tolerance_scale"] == 1.0
    assert metadata["pcg_tolerance_scale"] == 1.0
    assert len(rows) == len(traces) == 50
    result = []
    previous_contacts = None
    for frame, (row, trace) in enumerate(zip(rows, traces), 1):
        assert int(row["frame"]) == frame
        assert trace["physical_frame"] == frame
        assert trace["exit_reason"] == row["exit_reason"]
        assert int(row["diagnostic_budget_hits"]) == 0
        assert int(row["newton_cap_hits"]) == 0
        assert int(row["nonfinite_vertices"]) == 0
        samples = trace["samples"]
        updates = [sample for sample in samples if "line_search" in sample]
        assert len(updates) == int(row["newton_iterations"])
        assert len(samples) == len(updates) + 1
        signatures = {
            (sample["contacts_before"]["contact_count"],
             sample["contacts_before"]["contact_hash_xor"],
             sample["contacts_before"]["contact_hash_sum"])
            for sample in samples
        }
        start_contacts = samples[0]["contacts_before"]["contact_count"]
        end_contacts = samples[-1]["contacts_before"]["contact_count"]
        onset = 0 if previous_contacts is None else max(0, end_contacts - previous_contacts)
        result.append({
            "frame": frame,
            "checkpoint_frame": frame - 1,
            "newton": len(updates),
            "pcg": int(row["pcg_iterations"]),
            "step_ms": float(row["step_ms"]),
            "start_contacts": start_contacts,
            "end_contacts": end_contacts,
            "contacts_added_vs_previous_frame": onset,
            "contact_count_change_within_frame": end_contacts - start_contacts,
            "unique_contact_states": len(signatures),
            "contact_changes_within_updates": sum(
                (sample["contacts_before"]["contact_count"],
                 sample["contacts_before"]["contact_hash_xor"],
                 sample["contacts_before"]["contact_hash_sum"])
                != (sample["contacts_after"]["contact_count"],
                    sample["contacts_after"]["contact_hash_xor"],
                    sample["contacts_after"]["contact_hash_sum"])
                for sample in updates
            ),
            "accepted_alpha_min": min(
                sample["accepted_update_alpha"] for sample in updates
            ),
            "energy_backtracks": sum(
                sample["line_search"]["energy_backtracks"] for sample in updates
            ),
            "exit_direction_ratio": (
                samples[-1]["previous_direction_infinity_norm"]
                / samples[-1]["direction_stop_threshold"]
            ),
        })
        previous_contacts = end_contacts
    return result


def run_scene(scene: str) -> list[dict[str, object]]:
    path = OUT / "prefix" / scene
    if path.exists():
        return validate_run(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(EXE), "--scene", "1", "--pcg-dual-graph",
        "--benchmark-frames", "50", "--benchmark-output-dir", str(path),
        "--export-states", "--termination", "original",
        "--newton-tolerance-scale", "1", "--pcg-tolerance-scale", "1",
        "--trace-newton",
    ]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("GIPC_", "CIPC_"))}
    env.update(
        CUDA_VISIBLE_DEVICES="0",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CIPC_SCENE=scene,
        CIPC_INITIAL_VY="0",
        CIPC_META_OUT=str(path / "cloth_scene.json"),
        CIPC_INITIAL_OUT=str(path / "scene_initial.f64"),
        CIPC_TRIANGLES_OUT=str(path / "scene_triangles.u32"),
    )
    env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
    print(f"START selection/{scene}", flush=True)
    begin = time.monotonic()
    with path.with_suffix(".stdout.log").open("wb") as stdout, \
         path.with_suffix(".stderr.log").open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                stderr=stderr, timeout=1800)
    execution = {
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - begin,
        "exe_sha256": sha256(EXE),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    if result.returncode != 0:
        write_json(path.with_suffix(".execution.json"), execution)
        raise RuntimeError(f"A2 selection prefix failed: {scene}")
    write_json(path / "execution.json", execution)
    metrics = validate_run(path)
    print(f"DONE selection/{scene}: Newton={sum(x['newton'] for x in metrics)}, "
          f"wall={execution['wall_seconds']:.3f}s", flush=True)
    return metrics


def choose_states(scene: str, metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    eligible = [item for item in metrics if item["frame"] >= 2]
    rankings = {
        "contact_onset": sorted(
            eligible,
            key=lambda item: (item["contacts_added_vs_previous_frame"],
                              item["contact_count_change_within_frame"],
                              item["newton"], -item["frame"]),
            reverse=True,
        ),
        "dense_contact": sorted(
            eligible,
            key=lambda item: (item["end_contacts"], item["newton"],
                              -item["frame"]), reverse=True,
        ),
        "contact_rearrangement": sorted(
            eligible,
            key=lambda item: (item["unique_contact_states"],
                              item["contact_changes_within_updates"],
                              -item["accepted_alpha_min"], item["newton"],
                              -item["frame"]), reverse=True,
        ),
        "quiet_motion": sorted(
            eligible,
            key=lambda item: (item["newton"], item["unique_contact_states"],
                              abs(item["contact_count_change_within_frame"]),
                              -item["accepted_alpha_min"], item["frame"]),
        ),
    }
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for category in ("contact_onset", "dense_contact",
                     "contact_rearrangement", "quiet_motion"):
        candidate = next(item for item in rankings[category]
                         if item["frame"] not in used)
        used.add(candidate["frame"])
        selected.append({
            "id": f"{scene}-f{candidate['frame']:02d}",
            "scene": scene,
            "category": category,
            **candidate,
        })
    return selected


def main() -> None:
    frozen = OUT / "SELECTED_STATES.json"
    if frozen.exists():
        raise RuntimeError(f"Selection already frozen; refusing overwrite: {frozen}")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {scene: run_scene(scene) for scene in SCENES}
    selected = [item for scene in SCENES for item in choose_states(scene, metrics[scene])]
    payload = {
        "status": "frozen_before_candidate_runs",
        "method": "graph_n1 50-frame prefix; deterministic category rankings",
        "binary_sha256": sha256(EXE),
        "selection_inputs": {
            scene: {
                "newton_trace_sha256": sha256(OUT / "prefix" / scene
                                                / "newton_trace.jsonl"),
                "frames_sha256": sha256(OUT / "prefix" / scene / "frames.csv"),
            }
            for scene in SCENES
        },
        "selected_states": selected,
        "all_frame_metrics": metrics,
    }
    write_json(frozen, payload)
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Scout small-alpha states while saving every candidate checkpoint in the same run."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"
OUT = ROOT / "experiments/quality-cost-v3/a4-v2-scout"
A2_SELECTION = ROOT / "experiments/quality-cost-v3/a2-selection-v1/SELECTED_STATES.json"
SCENES = ("table", "bunny")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_scene(scene: str) -> list[dict[str, object]]:
    path = OUT / "scout" / scene
    metadata = read_json(path / "metadata.json")
    status = read_json(path / "status.json")
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    traces = [json.loads(line) for line in
              (path / "newton_trace.jsonl").read_text().splitlines()]
    assert status == {"completed": True, "frames_completed": 50,
                      "finite_state": True}
    assert metadata["checkpoint_frames"] == list(range(1, 50))
    assert len(rows) == len(traces) == 50
    metrics = []
    for frame, (row, trace) in enumerate(zip(rows, traces), 1):
        assert int(row["frame"]) == trace["physical_frame"] == frame
        assert int(row["nonfinite_vertices"]) == 0
        updates = [sample for sample in trace["samples"] if "line_search" in sample]
        assert len(updates) == int(row["newton_iterations"])
        metrics.append({
            "frame": frame,
            "checkpoint_frame": frame - 1,
            "newton": len(updates),
            "pcg": int(row["pcg_iterations"]),
            "step_ms": float(row["step_ms"]),
            "accepted_alpha_min": min(
                sample["accepted_update_alpha"] for sample in updates),
            "accepted_alpha_below_0_25": sum(
                sample["accepted_update_alpha"] < 0.25 for sample in updates),
            "accepted_alpha_below_0_10": sum(
                sample["accepted_update_alpha"] < 0.10 for sample in updates),
            "energy_backtracks": sum(
                sample["line_search"]["energy_backtracks"] for sample in updates),
        })
        if frame < 50:
            manifest = read_json(OUT / "checkpoints" / scene / f"S{frame}"
                                 / "manifest.json")
            assert manifest["schema"] == "stiffgipc-step-boundary-v3"
            assert manifest["complete"] and manifest["physical_frame"] == frame
    return metrics


def run_scene(scene: str) -> list[dict[str, object]]:
    path = OUT / "scout" / scene
    if path.exists():
        return validate_scene(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = OUT / "checkpoints" / scene
    command = [
        str(EXE), "--scene", "1", "--pcg-dual-graph",
        "--benchmark-frames", "50", "--benchmark-output-dir", str(path),
        "--termination", "original", "--newton-tolerance-scale", "1",
        "--pcg-tolerance-scale", "1", "--trace-newton",
        "--checkpoint-output-root", str(checkpoint_root),
        "--checkpoint-frames", ",".join(str(frame) for frame in range(1, 50)),
    ]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("GIPC_", "CIPC_"))}
    env.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CIPC_SCENE=scene, CIPC_INITIAL_VY="0",
               CIPC_META_OUT=str(path / "cloth_scene.json"),
               CIPC_INITIAL_OUT=str(path / "scene_initial.f64"),
               CIPC_TRIANGLES_OUT=str(path / "scene_triangles.u32"))
    env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
    print(f"START A4-v2 scout/{scene}", flush=True)
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
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    if result.returncode != 0:
        write_json(path.with_suffix(".execution.json"), execution)
        raise RuntimeError(f"A4-v2 scout failed: {scene}")
    write_json(path / "execution.json", execution)
    metrics = validate_scene(scene)
    print(f"DONE A4-v2 scout/{scene}: "
          f"min_alpha={min(item['accepted_alpha_min'] for item in metrics):.9g}",
          flush=True)
    return metrics


def main() -> None:
    frozen = OUT / "A4_V2_SELECTED_STATES.json"
    if frozen.exists():
        raise RuntimeError(f"Selection already frozen; refusing overwrite: {frozen}")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {scene: run_scene(scene) for scene in SCENES}
    prior = read_json(A2_SELECTION)["selected_states"]
    excluded = {scene: {item["frame"] for item in prior if item["scene"] == scene}
                for scene in SCENES}
    selected = []
    for scene in SCENES:
        eligible = [item for item in metrics[scene]
                    if 2 <= item["frame"] <= 50
                    and item["frame"] not in excluded[scene]]
        candidate = min(eligible, key=lambda item: (
            item["accepted_alpha_min"], -item["accepted_alpha_below_0_10"],
            -item["accepted_alpha_below_0_25"], -item["newton"], item["frame"]))
        selected.append({"id": f"{scene}-f{candidate['frame']:02d}",
                         "scene": scene, **candidate})
    payload = {
        "status": "frozen_before_a4_v2_candidate_runs",
        "method": ("Graph N1 50-frame scout; checkpoints S1..S49 saved in the same "
                   "run; exclude the eight A2 states; select minimum accepted alpha"),
        "reason": ("A4-v1 restored trajectories did not reproduce the alphas from an "
                   "independent selection run."),
        "shadow_epsilon_r": 0.03,
        "artificial_ccd_alpha_changes": False,
        "binary_sha256": sha256(EXE),
        "a2_selection_sha256": sha256(A2_SELECTION),
        "selection_inputs": {
            scene: {
                "newton_trace_sha256": sha256(
                    OUT / "scout" / scene / "newton_trace.jsonl"),
                "frames_sha256": sha256(OUT / "scout" / scene / "frames.csv"),
            } for scene in SCENES
        },
        "states": selected,
        "all_frame_metrics": metrics,
    }
    write_json(frozen, payload)
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

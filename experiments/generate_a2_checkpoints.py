"""Generate all frozen A2 state checkpoints with one replay per scene."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"
SELECTION_ROOT = ROOT / "experiments/quality-cost-v3/a2-selection-v1"
OUT = ROOT / "experiments/quality-cost-v3/a2-checkpoints-v1"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_frames() -> dict[str, list[int]]:
    frozen = read_json(SELECTION_ROOT / "SELECTED_STATES.json")
    assert frozen["status"] == "frozen_before_candidate_runs"
    result: dict[str, list[int]] = {}
    for state in frozen["selected_states"]:
        result.setdefault(state["scene"], []).append(state["checkpoint_frame"])
    return {scene: sorted(frames) for scene, frames in result.items()}


def validate_prefix(scene: str, frames: list[int]) -> None:
    path = OUT / "prefix" / scene
    status = read_json(path / "status.json")
    metadata = read_json(path / "metadata.json")
    assert status == {"completed": True, "frames_completed": max(frames),
                      "finite_state": True}
    assert metadata["checkpoint_frames"] == frames
    assert metadata["trace_newton"] is True
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == max(frames)
    assert all(int(row["diagnostic_budget_hits"]) == 0 for row in rows)
    for frame in frames:
        checkpoint = OUT / "checkpoints" / scene / f"S{frame}"
        manifest = read_json(checkpoint / "manifest.json")
        assert manifest["schema"] == "stiffgipc-step-boundary-v3"
        assert manifest["complete"] is True
        assert manifest["physical_frame"] == frame


def run_scene(scene: str, frames: list[int]) -> None:
    path = OUT / "prefix" / scene
    if path.exists():
        validate_prefix(scene, frames)
        print(f"KEEP checkpoints/{scene}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = OUT / "checkpoints" / scene
    command = [
        str(EXE), "--scene", "1", "--pcg-dual-graph",
        "--benchmark-frames", str(max(frames)),
        "--benchmark-output-dir", str(path), "--export-states",
        "--termination", "original", "--newton-tolerance-scale", "1",
        "--pcg-tolerance-scale", "1", "--trace-newton",
        "--checkpoint-output-root", str(checkpoint_root),
        "--checkpoint-frames", ",".join(map(str, frames)),
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
    print(f"START checkpoints/{scene}: {frames}", flush=True)
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
        "selection_sha256": sha256(SELECTION_ROOT / "SELECTED_STATES.json"),
    }
    if result.returncode != 0:
        write_json(path.with_suffix(".execution.json"), execution)
        raise RuntimeError(f"Checkpoint prefix failed: {scene}")
    write_json(path / "execution.json", execution)
    validate_prefix(scene, frames)
    print(f"DONE checkpoints/{scene}: {execution['wall_seconds']:.3f}s", flush=True)


def replay_error(scene: str, frames: int) -> dict[str, float]:
    reference = SELECTION_ROOT / "prefix" / scene
    replay = OUT / "prefix" / scene
    metadata = read_json(replay / "metadata.json")
    count = metadata["vertex_count"]
    boundary = np.fromfile(replay / "boundary.i32", dtype="<i4")
    cloth = np.asarray(read_json(replay / "cloth_scene.json")["roles"]) > 0
    mask = cloth & (boundary == 0)
    result: dict[str, float] = {"free_cloth_vertices": int(mask.sum())}
    for filename, label in (("states.f64", "position"),
                            ("velocities.f64", "velocity")):
        lhs = np.memmap(replay / filename, dtype="<f8", mode="r",
                        shape=(frames, count, 3))[:, mask]
        rhs = np.memmap(reference / filename, dtype="<f8", mode="r",
                        shape=(50, count, 3))[:frames, mask]
        distance = np.linalg.norm(lhs - rhs, axis=2)
        rms = np.sqrt(np.mean(distance * distance, axis=1))
        result[f"{label}_max_frame_rms"] = float(rms.max())
        result[f"{label}_final_rms"] = float(rms[-1])
        result[f"{label}_maximum"] = float(distance.max())
    return result


def main() -> None:
    frames = selected_frames()
    OUT.mkdir(parents=True, exist_ok=True)
    for scene in ("table", "bunny"):
        run_scene(scene, frames[scene])
    analysis = {
        "selection_sha256": sha256(SELECTION_ROOT / "SELECTED_STATES.json"),
        "binary_sha256": sha256(EXE),
        "checkpoint_frames": frames,
        "prefix_replay_error": {
            scene: replay_error(scene, max(scene_frames))
            for scene, scene_frames in frames.items()
        },
        "checkpoints": {
            scene: {
                f"S{frame}": {
                    "manifest_sha256": sha256(
                        OUT / "checkpoints" / scene / f"S{frame}" / "manifest.json"
                    ),
                    "contact_count": read_json(
                        OUT / "checkpoints" / scene / f"S{frame}" / "manifest.json"
                    )["contact_count"],
                }
                for frame in scene_frames
            }
            for scene, scene_frames in frames.items()
        },
    }
    write_json(OUT / "analysis.json", analysis)
    evidence_paths = [
        path for folder in (OUT / "prefix", OUT / "checkpoints")
        for path in folder.rglob("*") if path.is_file()
    ] + [OUT / "analysis.json", OUT / "CHECKPOINT_REPORT.md", OUT / "README.md",
         ROOT / "experiments/generate_a2_checkpoints.py", EXE,
         SELECTION_ROOT / "SELECTED_STATES.json"]
    write_json(OUT / "EVIDENCE_MANIFEST.json", {
        "files": [{
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in sorted(evidence_paths)]
    })
    print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

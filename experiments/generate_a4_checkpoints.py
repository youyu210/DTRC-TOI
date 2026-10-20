"""Generate exact checkpoints for the frozen A4 natural small-alpha states."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/gipc.exe"
FROZEN = ROOT / "experiments/quality-cost-v3/a4-selection-v1/A4_SELECTED_STATES.json"
REFERENCE_ROOT = ROOT / "experiments/quality-cost-v3/a2-selection-v1/prefix"
OUT = ROOT / "experiments/quality-cost-v3/a4-checkpoints-v1"


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(state: dict[str, object]) -> None:
    scene, frame = state["scene"], state["checkpoint_frame"]
    path = OUT / "prefix" / scene
    checkpoint = OUT / "checkpoints" / scene / f"S{frame}"
    if path.exists():
        manifest = read(checkpoint / "manifest.json")
        assert manifest["complete"] and manifest["physical_frame"] == frame
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(EXE), "--scene", "1", "--pcg-dual-graph",
        "--benchmark-frames", str(frame), "--benchmark-output-dir", str(path),
        "--export-states", "--termination", "original", "--trace-newton",
        "--checkpoint-output-root", str(checkpoint.parent),
        "--checkpoint-frames", str(frame),
    ]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("GIPC_", "CIPC_"))}
    env.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CIPC_SCENE=scene, CIPC_INITIAL_VY="0",
               CIPC_META_OUT=str(path / "cloth_scene.json"),
               CIPC_INITIAL_OUT=str(path / "scene_initial.f64"),
               CIPC_TRIANGLES_OUT=str(path / "scene_triangles.u32"))
    env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
    print(f"START A4 checkpoint {scene}/S{frame}", flush=True)
    begin = time.monotonic()
    with path.with_suffix(".stdout.log").open("wb") as stdout, \
         path.with_suffix(".stderr.log").open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                stderr=stderr, timeout=1800)
    execution = {"command": command, "returncode": result.returncode,
                 "wall_seconds": time.monotonic() - begin,
                 "exe_sha256": sha(EXE), "selection_sha256": sha(FROZEN)}
    if result.returncode != 0:
        write(path.with_suffix(".execution.json"), execution)
        raise RuntimeError(f"A4 checkpoint generation failed: {scene}")
    write(path / "execution.json", execution)
    manifest = read(checkpoint / "manifest.json")
    assert manifest["schema"] == "stiffgipc-step-boundary-v3"
    assert manifest["complete"] and manifest["physical_frame"] == frame
    print(f"DONE A4 checkpoint {scene}/S{frame}", flush=True)


def replay_error(state: dict[str, object]) -> dict[str, float]:
    scene, frames = state["scene"], state["checkpoint_frame"]
    lhs_root, rhs_root = OUT / "prefix" / scene, REFERENCE_ROOT / scene
    metadata = read(lhs_root / "metadata.json")
    count = metadata["vertex_count"]
    boundary = np.fromfile(lhs_root / "boundary.i32", dtype="<i4")
    roles = np.asarray(read(lhs_root / "cloth_scene.json")["roles"])
    mask = (boundary == 0) & (roles > 0)
    result: dict[str, float] = {"free_cloth_vertices": int(mask.sum())}
    for filename, label in (("states.f64", "position"),
                            ("velocities.f64", "velocity")):
        lhs = np.memmap(lhs_root / filename, dtype="<f8", mode="r",
                        shape=(frames, count, 3))[:, mask]
        rhs = np.memmap(rhs_root / filename, dtype="<f8", mode="r",
                        shape=(50, count, 3))[:frames, mask]
        distance = np.linalg.norm(lhs - rhs, axis=2)
        rms = np.sqrt(np.mean(distance * distance, axis=1))
        result[f"{label}_max_frame_rms"] = float(rms.max())
        result[f"{label}_final_rms"] = float(rms[-1])
    return result


def main() -> None:
    frozen = read(FROZEN)
    assert frozen["status"] == "frozen_before_a4_candidate_runs"
    OUT.mkdir(parents=True, exist_ok=True)
    for state in frozen["states"]:
        run(state)
    analysis = {
        "selection_sha256": sha(FROZEN),
        "binary_sha256": sha(EXE),
        "states": {
            state["id"]: {
                "checkpoint": f"{state['scene']}/S{state['checkpoint_frame']}",
                "manifest_sha256": sha(
                    OUT / "checkpoints" / state["scene"]
                    / f"S{state['checkpoint_frame']}" / "manifest.json"
                ),
                "replay_error": replay_error(state),
            }
            for state in frozen["states"]
        },
    }
    write(OUT / "analysis.json", analysis)
    paths = [path for folder in (OUT / "prefix", OUT / "checkpoints")
             for path in folder.rglob("*") if path.is_file()]
    paths += [OUT / "analysis.json", OUT / "CHECKPOINT_REPORT.md", OUT / "README.md",
              ROOT / "experiments/generate_a4_checkpoints.py", EXE, FROZEN]
    write(OUT / "EVIDENCE_MANIFEST.json", {"files": [{
        "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
        "sha256": sha(path)} for path in sorted(paths)]})
    print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

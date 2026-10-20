"""Verify restored A4-v2 checkpoints reproduce post-activation alpha targets."""

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
SCOUT = ROOT / "experiments/quality-cost-v3/a4-v2-scout"
SELECTION = SCOUT / "A4_V2_POSTACTIVATION_SELECTED_STATES.json"
OUT = ROOT / "experiments/quality-cost-v3/a4-v2-postactivation-restore-verification"


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path: Path, state: dict[str, object]) -> dict[str, object]:
    status = read_json(path / "status.json")
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    traces = [json.loads(line) for line in
              (path / "newton_trace.jsonl").read_text().splitlines()]
    assert status == {"completed": True, "frames_completed": 1,
                      "finite_state": True}
    assert len(rows) == len(traces) == 1
    updates = [sample for sample in traces[0]["samples"]
               if "line_search" in sample and sample["k"] >= 5]
    restored_min = min(sample["accepted_update_alpha"] for sample in updates)
    source_min = state["accepted_alpha_post_activation_min"]
    reproduced = restored_min <= source_min * 1.1 + 1e-6
    return {
        "source_post_activation_min_alpha": source_min,
        "restored_post_activation_min_alpha": restored_min,
        "source_newton": state["newton"],
        "restored_newton": int(rows[0]["newton_iterations"]),
        "restored_post_activation_updates": len(updates),
        "reproduced_post_activation_alpha": reproduced,
    }


def main() -> None:
    selection = read_json(SELECTION)
    assert selection["status"] == "frozen_before_a4_v2_postactivation_candidate_runs"
    results = {}
    for state in selection["states"]:
        path = OUT / state["id"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = SCOUT / "checkpoints" / state["scene"] / f"S{state['checkpoint_frame']}"
            command = [
                str(EXE), "--scene", "1", "--pcg-dual-graph",
                "--benchmark-frames", "1", "--benchmark-output-dir", str(path),
                "--termination", "original", "--newton-tolerance-scale", "1",
                "--pcg-tolerance-scale", "1", "--trace-newton",
                "--checkpoint-in", str(checkpoint),
                "--diagnostic-max-ms", "120000", "--diagnostic-max-newton", "2000",
            ]
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GIPC_", "CIPC_"))}
            env.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CIPC_SCENE=state["scene"], CIPC_INITIAL_VY="0",
                       CIPC_META_OUT=str(path / "cloth_scene.json"),
                       CIPC_INITIAL_OUT=str(path / "scene_initial.f64"),
                       CIPC_TRIANGLES_OUT=str(path / "scene_triangles.u32"))
            env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
            print(f"START A4-v2 postactivation verification/{state['id']}", flush=True)
            begin = time.monotonic()
            with path.with_suffix(".stdout.log").open("wb") as stdout, \
                 path.with_suffix(".stderr.log").open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout,
                                         stderr=stderr, timeout=180)
            write_json(path / "execution.json", {
                "command": command,
                "returncode": process.returncode,
                "wall_seconds": time.monotonic() - begin,
                "exe_sha256": sha256(EXE),
                "selection_sha256": sha256(SELECTION),
                "checkpoint_manifest_sha256": sha256(checkpoint / "manifest.json"),
            })
            if process.returncode != 0:
                raise RuntimeError(f"Restore verification failed: {state['id']}")
        result = validate(path, state)
        if not result["reproduced_post_activation_alpha"]:
            raise RuntimeError(f"Post-activation alpha not reproduced: {state['id']}: {result}")
        results[state["id"]] = result
        print(f"DONE {state['id']}: "
              f"source={result['source_post_activation_min_alpha']:.9g} "
              f"restored={result['restored_post_activation_min_alpha']:.9g}", flush=True)
    write_json(OUT / "VERIFICATION.json", {
        "status": "passed_before_candidate_runs",
        "criterion": "minimum accepted alpha among Newton k>=5",
        "binary_sha256": sha256(EXE),
        "selection_sha256": sha256(SELECTION),
        "results": results,
    })


if __name__ == "__main__":
    main()

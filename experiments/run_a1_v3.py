"""Run the preregistered A1 table frame-23 tolerance matrix from one S22."""

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
OUT = ROOT / "experiments/quality-cost-v3/a1-v1"
CHECKPOINT = OUT / "S22"
CONFIGS = {
    "N1-P1": (1.0, 1.0),
    "N4-P1": (0.25, 1.0),
    "N1-P4": (1.0, 0.25),
    "N4-P4": (0.25, 0.25),
}
ORDER = [
    ("N1-P1", 1), ("N4-P1", 1), ("N1-P4", 1), ("N4-P4", 1),
    ("N4-P4", 2), ("N1-P4", 2), ("N4-P1", 2), ("N1-P1", 2),
    ("N1-P4", 3), ("N1-P1", 3), ("N4-P4", 3), ("N4-P1", 3),
]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def validate(path: Path, newton_scale: float, pcg_scale: float) -> dict[str, str]:
    metadata = json.loads((path / "metadata.json").read_text())
    status = json.loads((path / "status.json").read_text())
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    trace = json.loads((path / "newton_trace.jsonl").read_text())
    budget_hit = int(rows[0]["diagnostic_budget_hits"]) == 1
    if budget_hit:
        assert status == {
            "completed": False, "frames_completed": 1, "finite_state": False
        }
        assert trace["exit_reason"] == "diagnostic_budget"
        assert int(rows[0]["newton_iterations"]) == 2000
    else:
        assert status == {
            "completed": True, "frames_completed": 1, "finite_state": True
        }
    assert len(rows) == 1 and trace["physical_frame"] == 23
    assert metadata["initial_physical_frame"] == 22
    assert metadata["newton_tolerance_scale"] == newton_scale
    assert metadata["pcg_tolerance_scale"] == pcg_scale
    assert metadata["checkpoint_in"].replace("\\", "/").endswith(
        "experiments/quality-cost-v3/a1-v1/S22"
    )
    assert int(rows[0]["nonfinite_vertices"]) == 0
    assert int(rows[0]["newton_cap_hits"]) == 0
    assert trace["exit_reason"] == rows[0]["exit_reason"]
    expected_samples = int(rows[0]["newton_iterations"]) + (0 if budget_hit else 1)
    assert len(trace["samples"]) == expected_samples
    return rows[0]


def main() -> None:
    assert CHECKPOINT.is_dir()
    exe_hash = sha256(EXE)
    for config, repeat in ORDER:
        newton_scale, pcg_scale = CONFIGS[config]
        path = OUT / "runs" / config / f"r{repeat:02d}"
        if path.exists():
            row = validate(path, newton_scale, pcg_scale)
            failed_execution = path.with_suffix(".execution.json")
            if not (path / "execution.json").exists() and failed_execution.exists():
                (path / "execution.json").write_bytes(failed_execution.read_bytes())
            print(f"KEEP {config}/r{repeat:02d}: Newton={row['newton_iterations']}, "
                  f"exit={row['exit_reason']}",
                  flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(EXE), "--scene", "1", "--pcg-dual-graph",
            "--benchmark-frames", "1", "--benchmark-output-dir", str(path),
            "--export-states", "--termination", "original",
            "--newton-tolerance-scale", str(newton_scale),
            "--pcg-tolerance-scale", str(pcg_scale),
            "--checkpoint-in", str(CHECKPOINT), "--trace-newton",
            "--diagnostic-max-ms", "300000", "--diagnostic-max-newton", "2000",
        ]
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("GIPC_", "CIPC_"))}
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CIPC_SCENE="table",
            CIPC_META_OUT=str(path / "cloth_scene.json"),
            CIPC_INITIAL_OUT=str(path / "scene_initial.f64"),
            CIPC_TRIANGLES_OUT=str(path / "scene_triangles.u32"),
        )
        env["PATH"] = str(EXE.parent) + os.pathsep + env["PATH"]
        print(f"START {config}/r{repeat:02d}", flush=True)
        begin = time.monotonic()
        with path.with_suffix(".stdout.log").open("wb") as stdout, \
             path.with_suffix(".stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=env,
                                    stdout=stdout, stderr=stderr, timeout=330)
        execution = {
            "command": command,
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - begin,
            "exe_sha256": exe_hash,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "checkpoint_manifest_sha256": sha256(CHECKPOINT / "manifest.json"),
        }
        row = validate(path, newton_scale, pcg_scale)
        if result.returncode not in (0, 2):
            write(path.with_suffix(".execution.json"), execution)
            raise RuntimeError(f"A1 run failed: {config}/r{repeat:02d}")
        if result.returncode == 2 and row["exit_reason"] != "diagnostic_budget":
            write(path.with_suffix(".execution.json"), execution)
            raise RuntimeError(f"Unexpected incomplete A1 run: {config}/r{repeat:02d}")
        write(path / "execution.json", execution)
        print(
            f"DONE {config}/r{repeat:02d}: Newton={row['newton_iterations']}, "
            f"PCG={row['pcg_iterations']}, step_ms={float(row['step_ms']):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

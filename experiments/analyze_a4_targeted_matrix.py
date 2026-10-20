"""Analyze the frozen A4 natural-small-alpha paired single-step matrix."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/quality-cost-v3/a4-targeted-v1"
SELECTION = ROOT / "experiments/quality-cost-v3/a4-selection-v1/A4_SELECTED_STATES.json"
CONFIGS = ("graph_n1", "toi", "comp03", "gate03", "strict2", "strict4")
CANDIDATES = CONFIGS[:-2]
POSITION_BANDS = (1e-6, 1e-5, 1e-4)
DT = 0.01


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def median_range(values: list[float]) -> dict[str, float]:
    return {"median": float(statistics.median(values)),
            "min": float(min(values)), "max": float(max(values))}


def states() -> list[dict[str, object]]:
    return read_json(SELECTION)["states"]


def run_path(state_id: str, config: str, repeat: int) -> Path:
    return OUT / "runs" / state_id / config / f"r{repeat:02d}"


def row(path: Path) -> dict[str, str]:
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    return rows[0]


def trace_summary(path: Path) -> dict[str, int | float]:
    records = [json.loads(line) for line in
               (path / "termination_trace.jsonl").read_text().splitlines()]
    assert len(records) == 1
    samples = records[0]["samples"]
    active = [sample for sample in samples if sample["active"] and sample["valid"]]
    shadow = [sample for sample in active
              if sample["beta"] <= 0.001
              and sample["normalized"] <= 0.03
              and sample["budget"] > 0.001]
    return {
        "samples": len(samples),
        "active_samples": len(active),
        "shadow_history_veto_samples": len(shadow),
        "minimum_positive_alpha": min(
            (sample["alpha"] for sample in active if sample["alpha"] > 0.0),
            default=1.0),
        "minimum_beta": min((sample["beta"] for sample in active), default=1.0),
        "minimum_budget": min((sample["budget"] for sample in active), default=1.0),
    }


def run_summary(path: Path) -> dict[str, object]:
    value = row(path)
    status = read_json(path / "status.json")
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "completed": bool(status["completed"]),
        "exit_reason": value["exit_reason"],
        "newton": int(value["newton_iterations"]),
        "pcg": int(value["pcg_iterations"]),
        "step_ms": float(value["step_ms"]),
        "residual_rejections": int(value["residual_rejections"]),
        "history_rejections": int(value["history_rejections"]),
        "guard_rejections": int(value["guard_rejections"]),
        **trace_summary(path),
    }


def free_mask(path: Path) -> np.ndarray:
    boundary = np.fromfile(path / "boundary.i32", dtype="<i4")
    roles = np.asarray(read_json(path / "cloth_scene.json")["roles"])
    mask = (boundary == 0) & (roles > 0)
    assert mask.any()
    return mask


def state_error(candidate: Path, reference: Path) -> dict[str, float]:
    mask = free_mask(reference)
    assert np.array_equal(mask, free_mask(candidate))
    result: dict[str, float] = {"free_cloth_vertices": int(mask.sum())}
    for filename, label in (("final.f64", "position"),
                            ("final_velocities.f64", "velocity")):
        lhs = np.fromfile(candidate / filename, dtype="<f8").reshape(-1, 3)[mask]
        rhs = np.fromfile(reference / filename, dtype="<f8").reshape(-1, 3)[mask]
        distance = np.linalg.norm(lhs - rhs, axis=1)
        result[f"{label}_rms"] = float(np.sqrt(np.mean(distance * distance)))
        result[f"{label}_max"] = float(distance.max())
    return result


def quality_band(error: dict[str, float]) -> str:
    for band in POSITION_BANDS:
        if (error["position_rms"] <= band
                and error["velocity_rms"] <= band / DT
                and error["position_max"] <= 10 * band
                and error["velocity_max"] <= 10 * band / DT):
            return f"{band:g}"
    return ">0.0001"


def band_rank(label: str) -> int:
    return {"1e-06": 0, "1e-05": 1, "0.0001": 2,
            ">0.0001": 3, "unresolved": 4}[label]


def main() -> None:
    marker = read_json(OUT / "MATRIX_COMPLETE.json")
    assert marker["jobs"] == 36
    runs: dict[str, dict[str, list[dict[str, object]]]] = {}
    references: dict[str, object] = {}
    for state in states():
        state_id = state["id"]
        runs[state_id] = {
            config: [run_summary(run_path(state_id, config, repeat))
                     for repeat in range(1, 4)]
            for config in CONFIGS
        }
        strict4_complete = all(run["completed"] for run in runs[state_id]["strict4"])
        strict2_complete = all(run["completed"] for run in runs[state_id]["strict2"])
        chosen = "strict4" if strict4_complete else ("strict2" if strict2_complete else None)
        agreement = ([state_error(run_path(state_id, "strict2", repeat),
                                  run_path(state_id, "strict4", repeat))
                      for repeat in range(1, 4)]
                     if strict2_complete and strict4_complete else [])
        noise = ([state_error(run_path(state_id, chosen, lhs),
                              run_path(state_id, chosen, rhs))
                  for lhs, rhs in itertools.combinations(range(1, 4), 2)]
                 if chosen else [])
        if agreement:
            pos_resolution = max(item["position_rms"] for item in agreement + noise)
            vel_resolution = max(item["velocity_rms"] for item in agreement + noise)
            resolved = next((f"{band:g}" for band in POSITION_BANDS
                             if pos_resolution <= 0.1 * band
                             and vel_resolution <= 0.1 * band / DT), "unresolved")
        else:
            pos_resolution = vel_resolution = None
            resolved = "unresolved"
        references[state_id] = {
            "chosen": chosen,
            "strict2_completed": sum(run["completed"] for run in runs[state_id]["strict2"]),
            "strict4_completed": sum(run["completed"] for run in runs[state_id]["strict4"]),
            "strict2_vs_strict4": agreement,
            "repeat_noise": noise,
            "position_resolution_rms": pos_resolution,
            "velocity_resolution_rms": vel_resolution,
            "finest_resolved_band": resolved,
        }

    per_state: dict[str, dict[str, object]] = {}
    for state in states():
        state_id = state["id"]
        per_state[state_id] = {}
        reference = references[state_id]["chosen"]
        for config in CONFIGS:
            group = runs[state_id][config]
            graph = runs[state_id]["graph_n1"]
            errors = ([state_error(run_path(state_id, config, repeat),
                                   run_path(state_id, reference, repeat))
                       for repeat in range(1, 4)] if reference else [])
            raw_band = max((quality_band(error) for error in errors),
                           key=band_rank, default="unresolved")
            ref_band = references[state_id]["finest_resolved_band"]
            validated = ("unresolved" if ref_band == "unresolved"
                         else max((raw_band, ref_band), key=band_rank))
            per_state[state_id][config] = {
                "completed_repeats": sum(run["completed"] for run in group),
                "exit_reasons": sorted({run["exit_reason"] for run in group}),
                "newton": median_range([float(run["newton"]) for run in group]),
                "step_ms": median_range([run["step_ms"] for run in group]),
                "paired_speedup_vs_graph_n1": median_range([
                    graph[index]["step_ms"] / group[index]["step_ms"]
                    for index in range(3)]),
                "errors_vs_reference": errors,
                "position_rms": median_range([error["position_rms"] for error in errors])
                    if errors else None,
                "velocity_rms": median_range([error["velocity_rms"] for error in errors])
                    if errors else None,
                "position_max": max((error["position_max"] for error in errors), default=None),
                "velocity_max": max((error["velocity_max"] for error in errors), default=None),
                "worst_raw_quality_band": raw_band,
                "validated_quality_band": validated,
                "residual_rejections": sum(run["residual_rejections"] for run in group),
                "history_rejections": sum(run["history_rejections"] for run in group),
                "guard_rejections": sum(run["guard_rejections"] for run in group),
                "active_samples": sum(run["active_samples"] for run in group),
                "shadow_history_veto_samples": sum(
                    run["shadow_history_veto_samples"] for run in group),
                "minimum_positive_alpha": min(
                    run["minimum_positive_alpha"] for run in group),
            }

    write_json(OUT / "analysis.json", {
        "matrix": marker,
        "references": references,
        "per_state": per_state,
        "runs": runs,
    })

    lines = [
        "# A4 natural-small-alpha targeted matrix",
        "",
        "Two states selected before candidate runs from naturally accepted small line-search steps were run "
        "with six configurations and three repeats (36 paired single steps). No CCD alpha was changed. "
        "Every method starts from the same schema-v3 checkpoint within each state.",
        "",
        "## Reference resolution",
        "",
        "| State | strict2 complete | strict4 complete | chosen | strict2↔strict4 max position RMS (m) | repeat-noise max RMS (m) | resolved band |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for state in states():
        state_id = state["id"]
        value = references[state_id]
        agreement = max((item["position_rms"] for item in value["strict2_vs_strict4"]),
                        default=float("nan"))
        noise = max((item["position_rms"] for item in value["repeat_noise"]),
                    default=float("nan"))
        lines.append(f"| {state_id} | {value['strict2_completed']}/3 | "
                     f"{value['strict4_completed']}/3 | {value['chosen'] or 'unresolved'} | "
                     f"{agreement:.6g} | {noise:.6g} | {value['finest_resolved_band']} |")
    lines += ["", "## Results", ""]
    for state in states():
        state_id = state["id"]
        lines += [
            f"### {state_id} (natural minimum accepted alpha {state['accepted_alpha_min']:.6g})",
            "",
            "| Config | Newton | seconds | speedup | position RMS max (m) | raw / validated band | history rejects | shadow history-veto samples | active samples | observed min alpha |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for config in CANDIDATES:
            value = per_state[state_id][config]
            lines.append(
                f"| {config} | {value['newton']['median']:g} | "
                f"{value['step_ms']['median']/1000:.4f} | "
                f"{value['paired_speedup_vs_graph_n1']['median']:.3f}× | "
                f"{value['position_rms']['max']:.6g} | "
                f"{value['worst_raw_quality_band']} / {value['validated_quality_band']} | "
                f"{value['history_rejections']} | {value['shadow_history_veto_samples']} | "
                f"{value['active_samples']} | {value['minimum_positive_alpha']:.6g} |"
            )
        lines.append("")
    total_history = sum(per_state[state["id"]]["comp03"]["history_rejections"]
                        for state in states())
    total_shadow = sum(per_state[state["id"]]["comp03"]["shadow_history_veto_samples"]
                       for state in states())
    if total_history == 0 and total_shadow == 0:
        finding = ("- The compensated method recorded zero history-only rejections and zero shadow "
                   "history-veto samples at εr=0.03 on the restored trajectories.")
    else:
        finding = (f"- The compensated method recorded {total_history} history-only rejections and "
                   f"{total_shadow} shadow history-veto samples at εr=0.03; inspect the per-state rows "
                   "before deciding whether the historical term has an independent effect.")
    lines += [
        "## Findings",
        "",
        finding,
        "- The selection minima occurred in the first Newton update, before residual/history gating "
        "became active at six accepted updates. Within the active termination window the minimum positive "
        "alphas were about 0.653/0.5. This A4-v1 result is therefore inconclusive for history behavior "
        "under smaller post-activation steps.",
        "- Timing and output differences between compensated and residual-gated runs cannot be attributed "
        "to history when the history-only counter is zero; independent GPU trajectories remain a source "
        "of run-to-run divergence.",
        "- The strict reference is a numerical reference. Its disagreement and repeat noise determine the "
        "finest defensible error band above.",
        "",
        "Derived values are in `analysis.json`; raw file hashes are in `EVIDENCE_MANIFEST.json`.",
        "",
    ]
    (OUT / "A4_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    evidence = [path for path in (OUT / "runs").rglob("*") if path.is_file()]
    evidence += [OUT / "MATRIX_COMPLETE.json", OUT / "analysis.json",
                 OUT / "A4_REPORT.md", ROOT / "experiments/run_a4_targeted_matrix.py",
                 ROOT / "experiments/analyze_a4_targeted_matrix.py", ROOT / "build/gipc.exe",
                 SELECTION]
    write_json(OUT / "EVIDENCE_MANIFEST.json", {
        "files": [{"path": path.relative_to(ROOT).as_posix(),
                   "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(evidence)]
    })


if __name__ == "__main__":
    main()

"""Analyze the frozen A3 paired single-step matrix."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/quality-cost-v3/a3-single-step-v2"
SELECTION = ROOT / "experiments/quality-cost-v3/a2-selection-v1/SELECTED_STATES.json"
CONFIGS = (
    "graph_n1", "graph_n2", "graph_n4", "graph_n8", "toi",
    "comp001", "comp01", "comp03", "comp10",
    "gate01", "gate03", "gate10", "strict2", "strict4",
)
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


def state_list() -> list[dict[str, object]]:
    return read_json(SELECTION)["selected_states"]


def run_path(state_id: str, config: str, repeat: int) -> Path:
    return OUT / "runs" / state_id / config / f"r{repeat:02d}"


def row(path: Path) -> dict[str, str]:
    with (path / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    return rows[0]


def run_summary(path: Path) -> dict[str, object]:
    value = row(path)
    status = read_json(path / "status.json")
    completed = bool(status["completed"])
    assert completed == (value["exit_reason"] != "diagnostic_budget")
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "completed": completed,
        "exit_reason": value["exit_reason"],
        "newton": int(value["newton_iterations"]),
        "pcg": int(value["pcg_iterations"]),
        "step_ms": float(value["step_ms"]),
        "residual_rejections": int(value["residual_rejections"]),
        "history_rejections": int(value["history_rejections"]),
        "guard_rejections": int(value["guard_rejections"]),
        "exit_residual": float(value["exit_residual"]),
        "exit_beta": float(value["exit_beta"]),
        "exit_budget": float(value["exit_budget"]),
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
    complete_marker = read_json(OUT / "MATRIX_COMPLETE.json")
    assert complete_marker["jobs"] == 336
    states = state_list()
    runs: dict[str, dict[str, list[dict[str, object]]]] = {}
    references: dict[str, object] = {}

    for state in states:
        state_id = state["id"]
        runs[state_id] = {}
        for config in CONFIGS:
            paths = [run_path(state_id, config, repeat) for repeat in range(1, 4)]
            assert all(path.is_dir() for path in paths)
            runs[state_id][config] = [run_summary(path) for path in paths]
        strict4_complete = all(run["completed"] for run in runs[state_id]["strict4"])
        strict2_complete = all(run["completed"] for run in runs[state_id]["strict2"])
        chosen = "strict4" if strict4_complete else ("strict2" if strict2_complete else None)
        references[state_id] = {
            "chosen": chosen,
            "strict2_completed": sum(run["completed"] for run in runs[state_id]["strict2"]),
            "strict4_completed": sum(run["completed"] for run in runs[state_id]["strict4"]),
            "strict2_vs_strict4": [],
            "repeat_noise": [],
        }
        if strict2_complete and strict4_complete:
            references[state_id]["strict2_vs_strict4"] = [
                state_error(run_path(state_id, "strict2", repeat),
                            run_path(state_id, "strict4", repeat))
                for repeat in range(1, 4)
            ]
        if chosen:
            references[state_id]["repeat_noise"] = [
                state_error(run_path(state_id, chosen, lhs),
                            run_path(state_id, chosen, rhs))
                for lhs, rhs in itertools.combinations(range(1, 4), 2)
            ]
        agreement = references[state_id]["strict2_vs_strict4"]
        noise = references[state_id]["repeat_noise"]
        if agreement:
            position_resolution = max(
                [item["position_rms"] for item in agreement]
                + [item["position_rms"] for item in noise]
            )
            velocity_resolution = max(
                [item["velocity_rms"] for item in agreement]
                + [item["velocity_rms"] for item in noise]
            )
            resolved_band = next((
                f"{band:g}" for band in POSITION_BANDS
                if position_resolution <= 0.1 * band
                and velocity_resolution <= 0.1 * band / DT
            ), "unresolved")
        else:
            position_resolution = None
            velocity_resolution = None
            resolved_band = "unresolved"
        references[state_id]["position_resolution_rms"] = position_resolution
        references[state_id]["velocity_resolution_rms"] = velocity_resolution
        references[state_id]["finest_resolved_band"] = resolved_band

    per_state: dict[str, dict[str, object]] = {}
    for state in states:
        state_id = state["id"]
        per_state[state_id] = {}
        reference = references[state_id]["chosen"]
        for config in CONFIGS:
            group = runs[state_id][config]
            graph = runs[state_id]["graph_n1"]
            errors = []
            if reference:
                errors = [
                    state_error(run_path(state_id, config, repeat),
                                run_path(state_id, reference, repeat))
                    for repeat in range(1, 4)
                ]
            labels = [quality_band(error) for error in errors]
            raw_band = max(labels, key=band_rank) if labels else "unresolved"
            reference_band = references[state_id]["finest_resolved_band"]
            if reference_band == "unresolved":
                validated_band = "unresolved"
            else:
                validated_band = max((raw_band, reference_band), key=band_rank)
            per_state[state_id][config] = {
                "completed_repeats": sum(run["completed"] for run in group),
                "exit_reasons": sorted({run["exit_reason"] for run in group}),
                "newton": median_range([float(run["newton"]) for run in group]),
                "pcg": median_range([float(run["pcg"]) for run in group]),
                "step_ms": median_range([run["step_ms"] for run in group]),
                "paired_speedup_vs_graph_n1": median_range([
                    graph[index]["step_ms"] / group[index]["step_ms"]
                    for index in range(3)
                ]),
                "errors_vs_reference": errors,
                "position_rms": median_range([error["position_rms"] for error in errors])
                    if errors else None,
                "velocity_rms": median_range([error["velocity_rms"] for error in errors])
                    if errors else None,
                "position_max": max((error["position_max"] for error in errors), default=None),
                "velocity_max": max((error["velocity_max"] for error in errors), default=None),
                "worst_raw_quality_band": raw_band,
                "validated_quality_band": validated_band,
                "residual_rejections": sum(run["residual_rejections"] for run in group),
                "history_rejections": sum(run["history_rejections"] for run in group),
                "guard_rejections": sum(run["guard_rejections"] for run in group),
            }

    aggregate: dict[str, object] = {}
    for config in CANDIDATES:
        cells = [per_state[state["id"]][config] for state in states]
        speedups = [cell["paired_speedup_vs_graph_n1"]["median"] for cell in cells]
        raw_bands = [cell["worst_raw_quality_band"] for cell in cells]
        validated_bands = [cell["validated_quality_band"] for cell in cells]
        position_values = [cell["position_rms"]["max"] for cell in cells
                           if cell["position_rms"]]
        velocity_values = [cell["velocity_rms"]["max"] for cell in cells
                           if cell["velocity_rms"]]
        aggregate[config] = {
            "completed_repeats": sum(cell["completed_repeats"] for cell in cells),
            "total_repeats": 24,
            "state_median_speedup": median_range(speedups),
            "worst_position_rms": max(position_values),
            "worst_velocity_rms": max(velocity_values),
            "worst_raw_quality_band": max(raw_bands, key=band_rank),
            "worst_validated_quality_band": max(validated_bands, key=band_rank),
            "reference_resolved_states": sum(
                band != "unresolved" for band in validated_bands
            ),
            "states_validated_within_1um": sum(
                band_rank(band) <= 0 for band in validated_bands
            ),
            "states_validated_within_10um": sum(
                band_rank(band) <= 1 for band in validated_bands
            ),
            "states_validated_within_100um": sum(
                band_rank(band) <= 2 for band in validated_bands
            ),
            "history_rejections": sum(cell["history_rejections"] for cell in cells),
            "residual_rejections": sum(cell["residual_rejections"] for cell in cells),
            "guard_rejections": sum(cell["guard_rejections"] for cell in cells),
        }

    analysis = {
        "matrix": complete_marker,
        "references": references,
        "aggregate": aggregate,
        "per_state": per_state,
        "runs": runs,
    }
    write_json(OUT / "analysis.json", analysis)

    lines = [
        "# A3 paired single-step quality/cost matrix",
        "",
        "Eight frozen states, 14 configurations, and three repeats produced 336 paired single steps. "
        "Every comparison starts from the exact same schema-v3 checkpoint. Heavy Newton tracing was off; "
        "the 120 s / 2000-Newton diagnostic limits remained active.",
        "",
        "## Reference resolution",
        "",
        "| State | strict2 complete | strict4 complete | Chosen reference | strict2↔strict4 max position RMS (m) | repeat-noise max RMS (m) | finest resolved band |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for state in states:
        state_id = state["id"]
        value = references[state_id]
        agreement = max((item["position_rms"] for item in value["strict2_vs_strict4"]),
                        default=float("nan"))
        noise = max((item["position_rms"] for item in value["repeat_noise"]),
                    default=float("nan"))
        lines.append(
            f"| {state_id} | {value['strict2_completed']}/3 | {value['strict4_completed']}/3 | "
            f"{value['chosen'] or 'unresolved'} | {agreement:.6g} | {noise:.6g} | "
            f"{value['finest_resolved_band']} |"
        )

    lines += [
        "",
        "## Aggregate candidate results",
        "",
        "Raw quality is the worst paired error band across all eight states and three repeats. "
        "A band requires position/velocity RMS and 10× maximum-error limits. Speedup is paired to Graph N1 "
        "within the same state and repeat, then summarized across states. Validated counts additionally require "
        "the strict-reference disagreement and repeat noise to be no more than 10% of that band.",
        "",
        "| Config | Complete | median state speedup [range] | worst position RMS (m) | worst velocity RMS (m/s) | worst raw band | resolved states | validated ≤1/10/100 μm | history rejects |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for config in CANDIDATES:
        value = aggregate[config]
        speed = value["state_median_speedup"]
        lines.append(
            f"| {config} | {value['completed_repeats']}/24 | "
            f"{speed['median']:.3f}× [{speed['min']:.3f}, {speed['max']:.3f}] | "
            f"{value['worst_position_rms']:.6g} | {value['worst_velocity_rms']:.6g} | "
            f"{value['worst_raw_quality_band']} | {value['reference_resolved_states']}/8 | "
            f"{value['states_validated_within_1um']}/"
            f"{value['states_validated_within_10um']}/"
            f"{value['states_validated_within_100um']} | "
            f"{value['history_rejections']} |"
        )

    lines += ["", "## Per-state medians", ""]
    for state in states:
        state_id = state["id"]
        lines += [
            f"### {state_id} ({state['category']})",
            "",
            "| Config | Newton | seconds | speedup | position RMS max (m) | velocity RMS max (m/s) | raw / validated band | exits |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for config in CANDIDATES:
            value = per_state[state_id][config]
            lines.append(
                f"| {config} | {value['newton']['median']:g} | "
                f"{value['step_ms']['median']/1000:.4f} | "
                f"{value['paired_speedup_vs_graph_n1']['median']:.3f}× | "
                f"{value['position_rms']['max']:.6g} | {value['velocity_rms']['max']:.6g} | "
                f"{value['worst_raw_quality_band']} / {value['validated_quality_band']} | "
                f"{', '.join(value['exit_reasons'])} |"
            )
        lines.append("")

    lines += [
        "## Findings",
        "",
        "- No candidate is validated within the 100 μm band on all six reference-resolved states; "
        "Graph N1 reaches five of six. The production tolerance remains unapproved.",
        "- Pure TOI has the largest median state speedup, but its worst paired position RMS is in the millimeter range.",
        "- Residual compensation records zero history-only rejections in every tested configuration. "
        "The selected states therefore show no independent protection from the historical budget.",
        "- comp03 and gate03 have similar median speedups and error scale; neither establishes a global 100 μm guarantee.",
        "",
        "## Interpretation limits",
        "",
        "The chosen strict result is a numerical reference, not an exact physical solution. "
        "Reference disagreement and repeat noise above quantify its resolution. A diagnostic-budget result "
        "is censored and never selected as the reference. These one-step results rank local quality/cost "
        "behavior; long-horizon trajectory validation remains a later stage.",
        "",
        "Derived values are in `analysis.json`; raw file hashes are in `EVIDENCE_MANIFEST.json`.",
        "",
    ]
    (OUT / "A3_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    evidence_paths = [path for path in (OUT / "runs").rglob("*") if path.is_file()]
    evidence_paths += [OUT / "MATRIX_COMPLETE.json", OUT / "analysis.json",
                       OUT / "A3_REPORT.md", OUT / "README.md",
                       ROOT / "experiments/run_a3_single_step_matrix.py",
                       ROOT / "experiments/analyze_a3_single_step_matrix.py",
                       ROOT / "build/gipc.exe", SELECTION]
    write_json(OUT / "EVIDENCE_MANIFEST.json", {
        "files": [{
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        } for path in sorted(evidence_paths)]
    })


if __name__ == "__main__":
    main()

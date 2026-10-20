"""Freeze A4 states by the smallest accepted alpha after termination activation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOUT = ROOT / "experiments/quality-cost-v3/a4-v2-scout"
PREACTIVATION = SCOUT / "A4_V2_SELECTED_STATES.json"
OUT = SCOUT / "A4_V2_POSTACTIVATION_SELECTED_STATES.json"
A2_SELECTION = ROOT / "experiments/quality-cost-v3/a2-selection-v1/SELECTED_STATES.json"
SCENES = ("table", "bunny")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Selection already frozen; refusing overwrite: {OUT}")
    prior = read_json(A2_SELECTION)["selected_states"]
    excluded = {scene: {item["frame"] for item in prior if item["scene"] == scene}
                for scene in SCENES}
    selected = []
    candidates = {}
    for scene in SCENES:
        traces = [json.loads(line) for line in
                  (SCOUT / "scout" / scene / "newton_trace.jsonl")
                  .read_text().splitlines()]
        scene_candidates = []
        for trace in traces:
            frame = trace["physical_frame"]
            updates = [sample for sample in trace["samples"]
                       if "line_search" in sample and sample["k"] >= 5]
            if frame < 2 or frame in excluded[scene] or not updates:
                continue
            minimum = min(updates, key=lambda sample: sample["accepted_update_alpha"])
            scene_candidates.append({
                "frame": frame,
                "checkpoint_frame": frame - 1,
                "newton": len([sample for sample in trace["samples"]
                               if "line_search" in sample]),
                "post_activation_updates": len(updates),
                "accepted_alpha_post_activation_min": minimum["accepted_update_alpha"],
                "minimum_at_newton_k": minimum["k"],
                "post_activation_alpha_below_0_75": sum(
                    sample["accepted_update_alpha"] < 0.75 for sample in updates),
            })
        scene_candidates.sort(key=lambda item: (
            item["accepted_alpha_post_activation_min"],
            -item["post_activation_alpha_below_0_75"],
            -item["post_activation_updates"], item["frame"]))
        candidates[scene] = scene_candidates
        choice = scene_candidates[0]
        selected.append({"id": f"{scene}-f{choice['frame']:02d}",
                         "scene": scene, **choice})
    payload = {
        "status": "frozen_before_a4_v2_postactivation_candidate_runs",
        "method": ("Use the same-run Graph N1 scout and checkpoints; exclude A2 states; "
                   "select the minimum naturally accepted alpha among Newton k>=5, where "
                   "the termination state is active for the following accepted-update sample."),
        "correction_reason": ("The preactivation selection minimized all Newton updates and "
                              "selected k=0, before the Kmin=6 residual/history gate."),
        "shadow_epsilon_r": 0.03,
        "minimum_updates": 6,
        "artificial_ccd_alpha_changes": False,
        "preactivation_selection_sha256": sha256(PREACTIVATION),
        "a2_selection_sha256": sha256(A2_SELECTION),
        "selection_inputs": {
            scene: {
                "newton_trace_sha256": sha256(
                    SCOUT / "scout" / scene / "newton_trace.jsonl"),
                "frames_sha256": sha256(SCOUT / "scout" / scene / "frames.csv"),
            } for scene in SCENES
        },
        "states": selected,
        "ranked_candidates": candidates,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

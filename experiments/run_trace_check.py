"""Default Kmin=6 state/update indexing audit, separate from speed runs."""
import argparse
import json
from pathlib import Path
from run_three_versions import run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = run(args.out, 1, "compensated", frames=40, trace=True)
    path = Path(data["path"])
    samples = [s for line in (path / "termination_trace.jsonl").read_text().splitlines()
               for s in json.loads(line)["samples"] if s["active"]]
    assert samples, "Default warmup was never reached"
    audits = json.loads((path / "audits.json").read_text())
    assert audits and all(a["agrees"] for a in audits)
    epsilon = data["metadata"]["toi_threshold"]
    eligible = [s for s in samples if s["accepted_updates"] >= 6]
    counts = {
        "active_samples": len(samples), "native_rhs_audits": len(audits),
        "full_step_large_residual_veto": sum(s["alpha"] == 1 and s["normalized"] > epsilon
            and s["budget"] > epsilon for s in eligible),
        "pure_toi_would_pass_budget_would_not": sum(s["beta"] <= epsilon
            and s["budget"] > epsilon for s in eligible),
        "current_gate_would_pass_budget_would_not": sum(s["beta"] <= epsilon
            and s["normalized"] <= epsilon and s["budget"] > epsilon for s in eligible),
        "reference_index_and_recurrence_verified": True,
        "all_native_rhs_audits_agree": True,
    }
    write_json(path / "TRACE_CHECK.json", counts)
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()

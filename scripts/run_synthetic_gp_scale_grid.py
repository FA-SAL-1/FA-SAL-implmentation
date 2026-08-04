from __future__ import annotations
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
import run_synthetic_gp_hyperparameter_update as base
import run_synthetic_gp_initial_scale_joint_noise50_lf_update_m4 as protocol
SCALES = (0.5, 1.0, 2.0)
CONDITIONS = tuple(f"sf_{sf:g}_sl_{sl:g}" for sf in SCALES for sl in SCALES)
SCALE_PAIRS = {
    f"sf_{sf:g}_sl_{sl:g}": (sf, sl)
    for sf in SCALES
    for sl in SCALES
}
for sf in SCALES:
    for sl in SCALES:
        base.CONDITIONS[f"sf_{sf:g}_sl_{sl:g}"] = {"length_scale": 1.0, "variance": 1.0, "retrain": False}
def common_kwargs(condition):
    kwargs = protocol._base_kwargs()
    sf, sl = SCALE_PAIRS[condition]
    kwargs["fa_lf_scale"] = sf
    kwargs["fa_l_ell_scale"] = 0.14 * sl
    kwargs["n_eval"] = 100
    kwargs["recovery_eval_interval"] = 10
    return kwargs
def run(task):
    condition, _ = task
    base._common_kwargs = lambda: common_kwargs(condition)
    base._scalar_metrics = protocol._metrics
    return base._run_task(task)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [(condition, seed) for condition in CONDITIONS for seed in range(args.trials)]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            item = future.result()
            records.append(item)
            print(f"[{index}/{len(tasks)}] {item['condition']} seed={item['seed']}", flush=True)
    records.sort(key=lambda item: (CONDITIONS.index(item["condition"]), item["seed"]))
    payload = {"experiment": {"benchmark": "penalty_stress", "method": "FA-SAL uncertainty-max", "horizon": 4, "evaluation_horizon": 4, "eval_points": 100, "recovery_eval_interval": 10, "rounds": 150, "paired_seeds": list(range(args.trials)), "scales": list(SCALES), "conditions": list(CONDITIONS), "scale_definition": {"s_f": "fa_lf_scale", "s_ell": "0.14 * s_ell"}, "gp_kernel": {"length_scale": 1.0, "variance": 1.0, "retrain": False}, "hyperparameter_update": "none"}, "trials": records, "summary": {condition: base._summarize([item for item in records if item["condition"] == condition]) for condition in CONDITIONS}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
if __name__ == "__main__":
    main()

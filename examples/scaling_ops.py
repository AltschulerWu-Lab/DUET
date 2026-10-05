"""Time DUET's three stages on an OPS problem the size of the paper's medium-scale CRISPRi benchmark.

The pool is built as in the paper's OPS benchmark: 1,000 genes of the
hCRISPRi-v2.1 library (Horlbeck et al. 2016) with their top 10 sgRNAs, 200
non-targeting controls, 10 sequencing rounds (quota 2 per gene). Needs a
source checkout, where the full table lives (data/), or DUET_DATA_DIR; the
core install is enough.

One `duet.design_ops` call with the default settings (num_samples=10,000,
eval_samples=5,000, six lambdas) is timed stage by stage, using the engine's
own stopwatches:

- PEP: the pairwise error probability matrix (GPU when a GPU device is given);
- sweep: the lambda sweep (always CPU, one process per lambda);
- evaluation: the Monte Carlo evaluation of every codebook (GPU when given).

Usage (run each on an otherwise idle machine):
    python examples/scaling_ops.py --device cpu
    python examples/scaling_ops.py --device gpu:0
    python examples/scaling_ops.py --device gpu:all
Results are summarized in examples/README.md.
"""

import argparse
import json
import os
import time

import duet
import duet.utils.time as duet_time
from duet.candidate_pool_factory import create_pool_from_source

STAGES = {
    "PEP matrix computation": "pep",
    "DUET optimization": "sweep",
    "Evaluation cache initialization": "evaluation",
    "Per-solution evaluation (matmul)": "evaluation",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", help='"cpu", "gpu:0" or "gpu:all"')
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU processes (default: all CPUs; the sweep uses one per lambda)")
    parser.add_argument("--json", default=None, help="append the timings to this JSON-lines file")
    args = parser.parse_args()

    timings = {"pep": 0.0, "sweep": 0.0, "evaluation": 0.0}
    original_exit = duet_time.Stopwatch.__exit__

    def recording_exit(self, *exc):
        stage = STAGES.get(self.label)
        if stage is not None:
            timings[stage] += time.perf_counter() - self.start_time
        return original_exit(self, *exc)

    duet_time.Stopwatch.__exit__ = recording_exit

    pool = create_pool_from_source(
        source="WeissmanCRISPRi", seq_rounds=10, quota=2, num_controls=200,
        min_rank=10, num_groups=1000, seed=0,
    )
    U = len(pool.unique_sequences)
    print(f"pool: {pool.pool_size:,} candidates, {pool.num_groups:,} groups, U = {U:,}")
    start = time.perf_counter()
    res = duet.design_ops(pool, duet.channels.symmetric(0.1), seed=0, device=args.device,
                          workers=args.workers, verbose=False)
    total = time.perf_counter() - start
    record = {
        "device": args.device, "U": U, "candidates": pool.pool_size,
        "num_samples": 10_000, "eval_samples": 5_000, "lambdas": 6,
        "cpu_processes": res.provenance["workers"]["cpu_processes"],
        "sweep_processes": res.provenance["workers"]["lambda_sweep"],
        "pep_s": round(timings["pep"], 1), "sweep_s": round(timings["sweep"], 1),
        "evaluation_s": round(timings["evaluation"], 1), "total_s": round(total, 1),
        "pep_storage": res.provenance["pep"]["storage"],
        "cpus_available": len(os.sched_getaffinity(0)),
    }
    print(json.dumps(record))
    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

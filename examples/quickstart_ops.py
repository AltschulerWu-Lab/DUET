"""Quickstart: design an optical pooled screen (OPS) sgRNA library with DUET.

Toy scale. This demo designs a library for 50 genes of the Horlbeck et al.
2016 hCRISPRi-v2.1 library (plus 50 non-targeting controls) on CPU, so that a
reviewer without a GPU can run it in about a minute. Real libraries (thousands
of genes) should run on GPU: install the `gpu` extra and use device="auto".

Usage:
    python examples/quickstart_ops.py OUTDIR

What it does:
    1. Reads examples/data/quickstart_ops.csv (550 sgRNAs, 10 per gene).
    2. Builds a candidate pool: select 2 sgRNAs per gene and 10 controls,
       decoded from the first 10 bases (10 sequencing rounds).
    3. Runs DUET for 6 values of lambda (lambda = 1 is decode-only) under a
       symmetric sequencing error of 10% per base, then evaluates every
       codebook with fresh simulated reads.
    4. Prints the result table, and writes it with every codebook, the
       settings and a plot of the Pareto front under OUTDIR.

Expected output (Intel Xeon E5-2640 v4, 4 CPU cores, CPU only; runtime
15 s):

    DUET 0.2.0 quickstart (OPS). Toy scale, on CPU: real libraries should run on GPU (pip install "duet-codebook[gpu]", device="auto").
    Candidate pool
      groups                  51
      candidates              550
      codebook size           110
      unique codewords (U)    550
      duplicate candidates    0 (share a codeword with an earlier candidate)
      sequence length         10
      swaps per iteration     1,390
      PEP reads               5,500,000 (10,000 per unique codeword)
      PEP in memory           605.0 kB (U x U uint16)
      PEP on disk if cached   1.2 MB (raw and symmetric copies)
       codebook  accuracy_mean  accuracy_p10  surrogate_accuracy  mean_score  duplicate_codewords  on_pareto_front
        initial         0.9501        0.9187              0.9072      0.7169                    0            False
       lambda=0         0.9449        0.8959              0.8973      0.8447                    0             True
    lambda=0.05         0.9495        0.9032              0.9056      0.8447                    0             True
     lambda=0.1         0.9511        0.9040              0.9085      0.8444                    0             True
    lambda=0.25         0.9543        0.9252              0.9140      0.8433                    0             True
     lambda=0.5         0.9614        0.9453              0.9247      0.8355                    0             True
       lambda=1         0.9726        0.9670              0.9476      0.6907                    0             True
      max_score         0.9456        0.8989              0.8987      0.8447                    0            False
    Operating point (Fig. 3b rule, 97.5% of the maximum mean score): lambda=0.5
    Wrote OUTDIR/table.csv, codebooks.csv, settings.json, pareto_front.svg
    Runtime: 15 s
"""

import argparse
import time
from pathlib import Path

import pandas as pd

import duet

DATA = Path(__file__).resolve().parent / "data" / "quickstart_ops.csv"
COLUMNS = ["codebook", "accuracy_mean", "accuracy_p10", "surrogate_accuracy",
           "mean_score", "duplicate_codewords", "on_pareto_front"]


def main(outdir: Path) -> duet.DesignResult:
    start = time.perf_counter()
    print(f"DUET {duet.__version__} quickstart (OPS). Toy scale, on CPU: real libraries "
          "should run on GPU (pip install \"duet-codebook[gpu]\", device=\"auto\").")

    table = pd.read_csv(DATA)
    quota = {g: (10 if g == "negative_control" else 2) for g in table["gene"].unique()}
    pool = duet.CandidatePool.from_table(
        table, group="gene", sequence="sequence", score="activity",
        quota=quota, seq_length=10,
    )
    print(pool.describe(num_samples=10_000))

    res = duet.design_ops(
        pool, duet.channels.symmetric(0.1),
        lambdas=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
        seed=0, num_samples=10_000, eval_samples=5_000,
        device="cpu", workers=4, verbose=False,
    )
    with pd.option_context("display.width", 120, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(res.table[COLUMNS].to_string(index=False))
    op = res.operating_point(score_fraction=0.975)
    print(f"Operating point (Fig. 3b rule, 97.5% of the maximum mean score): lambda={op['lambda']:g}")

    res.save(outdir)
    res.plot().figure.savefig(outdir / "pareto_front.svg")
    print(f"Wrote {outdir}/table.csv, codebooks.csv, settings.json, pareto_front.svg")
    print(f"Runtime: {time.perf_counter() - start:.0f} s")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, help="directory for the outputs")
    main(parser.parse_args().outdir)

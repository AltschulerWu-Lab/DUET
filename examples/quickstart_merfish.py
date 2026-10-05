"""Quickstart: design a MERFISH codebook with DUET.

Toy scale. This demo redesigns a 40-gene, 16-bit MERFISH codebook on CPU so
that a reviewer without a GPU can run it in about a minute. Real panels
(hundreds to thousands of genes, 32 bits) should run on GPU: install the
`gpu` extra and use device="auto".

It samples 40 candidate barcodes per gene instead of the default 1,000, to
fit the CPU budget; every other setting is a default.

Inputs:
    - Gene names and barcodes: the first 40 genes of the Hamming-weight-4,
      16-bit codebook of Boström et al. 2025 as assigned in
      examples/data/bostrom/bostrom_hw4_16bit_panel.csv. It is the initial
      codebook S0, and the crowding objective 1 - C(S)/C(S0) is anchored at it.
    - Expression: SYNTHETIC, drawn from a log-normal distribution
      (numpy.random.default_rng(0)). The panel file's measured counts
      (mean_raw_counts) are not used.
    - Channel: an asymmetric binary readout error (0 read as 1 with
      probability 0.015, 1 read as 0 with probability 0.056).

Usage:
    python examples/quickstart_merfish.py OUTDIR

Expected output (Intel Xeon E5-2640 v4, 4 CPU cores, CPU only; runtime
44 s):

    DUET 0.2.0 quickstart (MERFISH). Toy scale, on CPU: real panels should run on GPU (pip install "duet-codebook[gpu]", device="auto").
    Using 40 candidate barcodes per gene (default 1,000) to fit the CPU budget. Expression is SYNTHETIC (log-normal).
       codebook  accuracy_mean  accuracy_p10  surrogate_accuracy  crowding  duplicate_codewords  on_pareto_front
        initial         0.9632        0.9562              0.9405    0.0000                    0            False
       lambda=0         0.8940        0.8744              0.8714    0.2474                    2             True
     lambda=0.1         0.9564        0.9189              0.9421    0.2451                    0             True
     lambda=0.3         0.9662        0.9566              0.9554    0.2441                    0             True
     lambda=0.5         0.9679        0.9578              0.9595    0.2413                    0            False
     lambda=0.7         0.9682        0.9625              0.9626    0.2419                    0             True
     lambda=0.8         0.9682        0.9605              0.9617    0.2302                    0            False
     lambda=0.9         0.9677        0.9624              0.9637    0.2385                    0            False
    lambda=0.95         0.9677        0.9615              0.9630    0.2276                    0            False
       lambda=1         0.9691        0.9632              0.9641    0.1692                    0             True
    C(S0) of the initial codebook: 2.567e+06
    Wrote OUTDIR/table.csv, codebooks.csv, settings.json, pareto_front.svg
    Runtime: 44 s
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import duet

PANEL = Path(__file__).resolve().parent / "data" / "bostrom" / "bostrom_hw4_16bit_panel.csv"
COLUMNS = ["codebook", "accuracy_mean", "accuracy_p10", "surrogate_accuracy",
           "crowding", "duplicate_codewords", "on_pareto_front"]
N_GENES = 40
CANDIDATES_PER_GENE = 40


def main(outdir: Path) -> duet.DesignResult:
    start = time.perf_counter()
    print(f"DUET {duet.__version__} quickstart (MERFISH). Toy scale, on CPU: real panels "
          "should run on GPU (pip install \"duet-codebook[gpu]\", device=\"auto\").")
    print(f"Using {CANDIDATES_PER_GENE} candidate barcodes per gene (default 1,000) to fit the "
          "CPU budget. Expression is SYNTHETIC (log-normal).")

    panel = pd.read_csv(PANEL, dtype={"Sequence": str}).head(N_GENES)
    rng = np.random.default_rng(0)
    genes = pd.DataFrame({
        "gene": panel["Gene"],
        "expression": rng.lognormal(mean=3.0, sigma=1.5, size=len(panel)),  # synthetic
    })
    initial = pd.DataFrame({"gene": panel["Gene"], "barcode": panel["Sequence"]})
    channel = duet.channels.asymmetric([[0.985, 0.015], [0.056, 0.944]], alphabet="01")

    res = duet.design_merfish(
        genes, channel, gene="gene", expression="expression",
        n_bits=16, hamming_weights={4: None}, candidates_per_gene=CANDIDATES_PER_GENE,
        init=initial, seed=0, num_samples=10_000, eval_samples=5_000,
        device="cpu", workers=4, verbose=False,
    )
    with pd.option_context("display.width", 120, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(res.table[COLUMNS].to_string(index=False))
    print(f"C(S0) of the initial codebook: {res.provenance['merfish']['crowding_anchor_C_S0']:.4g}")

    res.save(outdir)
    res.plot().figure.savefig(outdir / "pareto_front.svg")
    print(f"Wrote {outdir}/table.csv, codebooks.csv, settings.json, pareto_front.svg")
    print(f"Runtime: {time.perf_counter() - start:.0f} s")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outdir", type=Path, help="directory for the outputs")
    main(parser.parse_args().outdir)

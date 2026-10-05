"""build_candidates: the MERFISH runner's candidate pool (steps 1-5).

The pool's sequence order is what the PEP cache fingerprint hashes, and
experiments/merfish_expression_prior relies on two properties checked here:
each position is anchored by the warm-start and baseline codewords of its
row, and the pool depends on those row-wise codewords but not on which genes
the rows carry.
"""
from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import pandas as pd

from duet.merfish_benchmark.config import MerfishBenchmarkConfig
from duet.merfish_benchmark.runner import build_candidates

SEED = 42
# HW-2 codewords on 8 rounds, in "file order"; panels use the first 4.
CODEWORDS = ["11000000", "00110000", "00001100", "00000011", "10100000", "01010000"]
MHD4_PERM = [2, 0, 3, 1]   # row i of the shuffled baseline gets CODEWORDS[MHD4_PERM[i]]


def _write_panel(tmp_path, tag, genes, expression, perm=MHD4_PERM):
    """Expression table plus Bostrom-like and MHD4-like panels, rows sorted by expression."""
    order = sorted(range(len(genes)), key=lambda i: -expression[i])
    genes = [genes[i] for i in order]
    expr = pd.DataFrame({"gene_symbol": genes + ["filler"],
                         "mean_cpm": [expression[i] for i in order] + [1.0]})
    bostrom = pd.DataFrame({"Gene": genes, "Sequence": CODEWORDS[:4]})
    mhd4 = pd.DataFrame({"Gene": genes, "Sequence": [CODEWORDS[j] for j in perm]})
    paths = {name: tmp_path / f"{tag}_{name}.csv" for name in ("expr", "bostrom", "mhd4")}
    expr.to_csv(paths["expr"], index=False)
    bostrom.to_csv(paths["bostrom"], index=False)
    mhd4.to_csv(paths["mhd4"], index=False)
    return paths


def _config(tmp_path, paths):
    component = {
        "noise_channel": {"type": "symmetric", "epsilon": 0.05},
        "decoding_metric": {"type": "hamming"},
        "decoding_rule": {"type": "unique_minimum"},
        "num_samples": 10, "num_cpus": 1,
    }
    return MerfishBenchmarkConfig.from_dict({
        "seed": SEED,
        "outdir": str(tmp_path / "out"),
        "candidates": {"seq_rounds": 8, "codebook_size": 4, "hamming_weights": [2],
                       "per_codeword_sample_size": 6, "sample_seed": SEED},
        "evaluator": component,
        "expression": {"path": str(paths["expr"]), "gene_col": "gene_symbol",
                       "expression_col": "mean_cpm"},
        "genes": {"type": "from_csv", "path": str(paths["bostrom"]), "gene_col": "Gene"},
        "initialization": {"type": "warm_start", "path": str(paths["bostrom"]),
                           "sequence_col": "Sequence", "noise_percent": 0.0},
        "duet": {"lambda": [1.0], "max_iter": 1, "max_patience": 1, "pep": component},
        "baselines": [
            {"name": "Bostrom", "path": str(paths["bostrom"]), "gene_col": "Gene",
             "sequence_col": "Sequence"},
            {"name": "MHD4", "path": str(paths["mhd4"]), "gene_col": "Gene",
             "sequence_col": "Sequence"},
        ],
    })


def test_each_position_is_anchored_by_its_rows_codewords(tmp_path):
    paths = _write_panel(tmp_path, "a", ["g1", "g2", "g3", "g4"], [5.0, 40.0, 1.0, 12.0])
    built = build_candidates(_config(tmp_path, paths), SEED)

    groups = built.candidates.to_dataframe().groupby("Group")["Sequence"].apply(set)
    for i in range(4):
        assert {CODEWORDS[i], CODEWORDS[MHD4_PERM[i]]} <= groups[f"pos_{i}"]
    assert built.gene_names == ["g2", "g4", "g1", "g3"]   # expression order


def test_pool_does_not_depend_on_which_genes_the_rows_carry(tmp_path):
    a = _write_panel(tmp_path, "a", ["g1", "g2", "g3", "g4"], [5.0, 40.0, 1.0, 12.0])
    b = _write_panel(tmp_path, "b", ["h1", "h2", "h3", "h4"], [300.0, 2.0, 7.0, 0.5])

    pool_a = list(build_candidates(_config(tmp_path, a), SEED).candidates.sequences)
    pool_b = list(build_candidates(_config(tmp_path, b), SEED).candidates.sequences)

    assert pool_a == pool_b


def test_pool_order_depends_on_the_shuffled_baselines_rows(tmp_path):
    a = _write_panel(tmp_path, "a", ["g1", "g2", "g3", "g4"], [5.0, 40.0, 1.0, 12.0])
    b = _write_panel(tmp_path, "b", ["g1", "g2", "g3", "g4"], [5.0, 40.0, 1.0, 12.0],
                     perm=[1, 3, 0, 2])

    pool_a = list(build_candidates(_config(tmp_path, a), SEED).candidates.sequences)
    pool_b = list(build_candidates(_config(tmp_path, b), SEED).candidates.sequences)

    assert pool_a != pool_b

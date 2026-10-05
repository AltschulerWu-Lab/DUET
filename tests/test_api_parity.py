"""Parity of the public API with the paper's benchmark runners and fixtures.

- MERFISH pool parity: the pool design_merfish builds equals the runner's, in
  shared-group mode (merfish_asymmetric reference) and per-gene mode (a tiny
  CPU run of the MERFISH runner with an initial codebook and one baseline
  anchored).
- Operating point: duet.results.select_operating_point picks the same lambda
  as scripts/benchmark/visualize_benchmark.py on the ops_uniform reference.
- Paper parity (GPU, slow): design_ops reproduces the ops_uniform fixture's
  DUET selections for trial 1 exactly. Accuracies are not compared: the
  benchmark evaluates DUET together with the baselines.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import duet
from duet.api import _build_merfish_pool
from duet.results import select_operating_point

REPO = Path(__file__).resolve().parents[1]
REGRESSION = REPO / "scripts" / "benchmark" / "regression"


def test_merfish_pool_matches_the_shared_group_reference():
    """merfish_asymmetric: 16 bits, weights 3-5, codebook_1 warm start, one shared group."""
    ref = pd.read_csv(REGRESSION / "merfish_asymmetric" / "reference" / "candidates.csv.gz",
                      dtype={"Sequence": str})
    genes = pd.read_csv(
        REGRESSION / "merfish_asymmetric" / "reference" / "selected_codewords_lambda0.00.csv.gz"
    )["Gene"].astype(str).tolist()
    codebook_1 = pd.read_csv(REPO / "examples/data/processed/MERFISH/codebook_1.csv",
                             dtype={"Sequence": str})
    pool = _build_merfish_pool(
        genes, 16, {3: None, 4: None, 5: None}, None,
        init_barcodes=codebook_1["Sequence"].tolist(),  # warm start: row i -> position i
        includes=[("codebook_1", codebook_1["Gene"].astype(str).tolist(),
                   codebook_1["Sequence"].tolist())],
        pool_seed=42,
    )
    got = pool.to_dataframe()
    assert len(got) == len(ref) == 6748
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref, check_dtype=False)


def _write_tiny_merfish_config(tmp_path: Path) -> Path:
    """Per-gene mode, an initial codebook and one baseline anchored at a panel gene."""
    import yaml

    genes = ["Ga", "Gb", "Gc", "Gd"]
    pd.DataFrame({"Gene": genes}).to_csv(tmp_path / "genes.csv", index=False)
    pd.DataFrame({"gene": genes + ["other"], "cpm": [50.0, 20.0, 5.0, 80.0, 1.0]}).to_csv(
        tmp_path / "expr.csv", index=False)
    init = ["11100000", "00011100", "10010010", "01101000"]
    pd.DataFrame({"Gene": genes, "Sequence": init}).to_csv(tmp_path / "init.csv", index=False)
    # The baseline gives Gb a different barcode (anchored), plus a gene outside the panel.
    pd.DataFrame({"Gene": ["Gb", "Zz"], "Sequence": ["11000100", "00000111"]}).to_csv(
        tmp_path / "ref.csv", index=False)
    cfg = {
        "seed": 0,
        "outdir": str(tmp_path / "out"),
        "candidates": {"seq_rounds": 8, "codebook_size": 4, "hamming_weights": [3, 4],
                       "hamming_weight_limits": {4: 40}, "per_codeword_sample_size": 6,
                       "sample_seed": 5, "subsample_seed": 5},
        "evaluator": {"noise_channel": {"type": "symmetric", "epsilon": 0.1},
                      "decoding_metric": {"type": "hamming"},
                      "decoding_rule": {"type": "unique_minimum"},
                      "num_samples": 100, "num_cpus": 1, "seed": 1},
        "expression": {"path": "expr.csv", "gene_col": "gene", "expression_col": "cpm"},
        "genes": {"type": "from_csv", "path": "genes.csv", "gene_col": "Gene"},
        "initialization": {"type": "warm_start", "path": "init.csv", "sequence_col": "Sequence"},
        "duet": {"lambda": [1.0], "max_iter": 50, "max_patience": 10, "num_cpus": 1,
                 "use_mmap": False,
                 "pep": {"noise_channel": {"type": "symmetric", "epsilon": 0.1},
                         "decoding_metric": {"type": "hamming"},
                         "decoding_rule": {"type": "unique_minimum"},
                         "num_samples": 100, "num_cpus": 1, "seed": 2}},
        "baselines": [{"name": "ref", "path": "ref.csv", "gene_col": "Gene", "sequence_col": "Sequence"}],
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_merfish_pool_matches_a_per_gene_runner_run(tmp_path):
    from _optional_deps import require_benchmark_extra

    require_benchmark_extra()
    import yaml

    from duet.merfish_benchmark import MerfishBenchmarkConfig, run_merfish_benchmark

    cfg_path = _write_tiny_merfish_config(tmp_path)
    d = yaml.safe_load(cfg_path.read_text())
    config = MerfishBenchmarkConfig.from_dict(d, config_dir=cfg_path.parent)
    run_merfish_benchmark(config, args=argparse.Namespace(debug_pep=False, config=str(cfg_path)))
    runner = pd.read_csv(tmp_path / "out" / "candidates.csv", dtype={"Sequence": str})

    ref = pd.read_csv(tmp_path / "ref.csv", dtype={"Sequence": str})
    init = pd.read_csv(tmp_path / "init.csv", dtype={"Sequence": str})
    pool = _build_merfish_pool(
        ["Ga", "Gb", "Gc", "Gd"], 8, {3: None, 4: 40}, 6,
        init_barcodes=init["Sequence"].tolist(),
        includes=[("ref", ref["Gene"].tolist(), ref["Sequence"].tolist())],
        pool_seed=5,
    )
    got = pool.to_dataframe()
    assert "11000100" in set(got.loc[got["Group"] == "pos_1", "Sequence"])  # the anchored baseline
    pd.testing.assert_frame_equal(got[["Group", "Sequence"]].reset_index(drop=True),
                                  runner[["Group", "Sequence"]], check_dtype=False)


def _visualize_benchmark():
    sys.path.insert(0, str(REPO / "scripts" / "benchmark"))
    try:
        import visualize_benchmark
    finally:
        sys.path.pop(0)
    return visualize_benchmark


def test_operating_point_matches_the_figure_script_on_ops_uniform():
    from _optional_deps import require_benchmark_extra

    require_benchmark_extra()
    import yaml

    vb = _visualize_benchmark()
    lambdas = yaml.safe_load((REGRESSION / "ops_uniform" / "ops_uniform.yaml").read_text())
    lambdas = lambdas["duet"]["optimizer"]["lambda"]
    label_of = {f"DUET (lambda={lam:.2f})": lam for lam in lambdas}
    _, agg = vb.load_and_process_results(REGRESSION / "ops_uniform" / "reference" / "results.csv.gz")
    checked = 0
    for trial in sorted(agg["Trial"].unique()):
        arms = vb._reference_arms(agg, trial)
        max_act = agg.loc[(agg["Method"] == vb.MAX_ACTIVITY_METHOD) & (agg["Trial"] == trial),
                          "Mean activity score"].iloc[0]
        duet_rows = agg[(agg["Method group"] == "DUET") & (agg["Trial"] == trial)]
        for suffix, fraction in vb.DUET_REFERENCE_FRACTIONS.items():
            lam = select_operating_point(
                [label_of[m] for m in duet_rows["Method"]],
                duet_rows["Mean decode accuracy"], duet_rows["Mean activity score"],
                max_act, fraction,
            )
            script = vb.select_duet_lambda(agg, trial, fraction * max_act)
            assert (None if lam is None else f"DUET (lambda={lam:.2f})") == script
            assert arms.get(suffix) == script
            checked += 1
    assert checked == 6


def _gpu_available():
    from duet.gpu_utils import has_gpu

    return has_gpu()


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_available(), reason="needs CuPy and a CUDA GPU")
def test_design_ops_reproduces_the_ops_uniform_selections_for_trial_1():
    """Paper parity: the facade reproduces the fixture's DUET codebooks bit for bit.

    The pool is built with the runner's own factory call, so the row order
    matches; the stage seeds are the fixture's (trial seed for the initial
    codebook and the optimizer, 42 for the PEP). Run only when no other job
    uses the GPUs: python -m pytest tests/test_api_parity.py -m slow
    """
    from _optional_deps import require_benchmark_extra

    require_benchmark_extra()
    from duet.candidate_pool_factory import create_pool_from_source
    from duet.ops_benchmark import OpsBenchmarkConfig

    config = OpsBenchmarkConfig.from_yaml(REGRESSION / "ops_uniform" / "ops_uniform.yaml")
    trial_seed = int(np.random.default_rng(config.seed).integers(0, 2**31 - 1, size=config.trials)[0])
    cp = config.candidate_pool
    pool = create_pool_from_source(
        source=cp.source, seq_rounds=cp.seq_rounds, quota=cp.quota,
        num_controls=cp.num_controls, min_rank=cp.min_rank, num_groups=cp.num_groups,
        seed=trial_seed, pairing_strategy=cp.pairing_strategy,
        max_control_pairs=cp.max_control_pairs, csv_path=cp.csv_path,
        control_quotas=cp.control_quotas, score_method=cp.score_method,
        candidates_per_group=cp.candidates_per_group, alphabet_size=cp.alphabet_size,
    )
    opt = config.duet_optimizer
    pep = config.duet_pep
    assert pep.noise_channel.to_dict() == {"type": "symmetric", "epsilon": 0.1}
    assert pep.decoding_metric.type == "hamming" and pep.decoding_rule.type == "unique_minimum"
    res = duet.design_ops(
        pool, duet.channels.symmetric(0.1),
        lambdas=opt.lambda_, num_samples=pep.num_samples, eval_samples=0,
        device="gpu:all", workers=opt.num_cpus,
        max_patience=opt.max_patience, max_iter=opt.max_iter,
        avoid_duplicates=opt.avoid_duplicates, duplicate_penalty=opt.duplicate_penalty,
        temperature=opt.temperature, pep_storage="mmap",  # the fixture's path: cache + mmap
        _stage_seeds={"init": trial_seed, "optimizer": trial_seed, "pep": pep.seed,
                      "evaluation": pep.seed + 1},
    )
    golden = pd.read_csv(REGRESSION / "ops_uniform" / "reference" / "results.csv.gz")
    golden = golden[golden["Trial"] == 1]
    for lam in opt.lambda_:
        expect = golden.loc[golden["Method"] == f"DUET (lambda={lam:.2f})", "Index"].to_numpy()
        np.testing.assert_array_equal(res.codebook(lam)["candidate"].to_numpy(), expect,
                                      err_msg=f"lambda={lam}")

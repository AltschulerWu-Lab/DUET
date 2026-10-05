"""Tests for the public API (duet.api, duet.channels, duet.results).

The parity contract (docs/adr/0003-public-api.md): given the same pool, row
order, initial codebook, stage seeds and device, the facade returns the same
selections as run_duet_ops / run_duet_merfish, and its accuracies equal
evaluate_codebooks_by_sequence on the same ordered list of codebooks. The
decoding-only duet.design selects what design_ops(lambdas=[1]) selects.
"""

import inspect
import itertools
import json
import random
import warnings

import numpy as np
import pandas as pd
import pytest

import duet
from duet.api import derive_stage_seeds
from duet.benchmark.metrics import (
    _build_multiset_union,
    compute_metrics,
    evaluate_codebooks_by_sequence,
)
from duet.evaluator_config import ComponentConfig, EvaluatorConfig
from duet.initialization import RandomInit
from duet.runner import DuetOptimizerConfig, run_duet_merfish, run_duet_ops

LAMBDAS = [0.0, 0.3, 1.0]


def _distinct_barcodes(n, length, rng):
    codes = rng.choice(4 ** length, size=n, replace=False)
    return ["".join("ACGT"[(c >> (2 * i)) & 3] for i in range(length)) for c in codes]


def _ops_table(n_groups=24, per_group=5, length=7, seed=0):
    rng = np.random.default_rng(seed)
    barcodes = _distinct_barcodes(n_groups * per_group + 1, length, rng)
    rows = [
        {"gene": f"g{g:02d}", "barcode": barcodes[g * per_group + k],
         "activity": float(rng.uniform(0.2, 1.0))}
        for g in range(n_groups) for k in range(per_group)
    ]
    # Two controls; one shares its sequence with a targeting guide (dedup path).
    rows.append({"gene": "ctrl", "barcode": rows[0]["barcode"], "activity": 1.0})
    rows.append({"gene": "ctrl", "barcode": barcodes[-1], "activity": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def ops_pool():
    return duet.CandidatePool.from_table(
        _ops_table(), group="gene", sequence="barcode", score="activity", quota=2,
    )


def _pep_config(channel, num_samples, seed, rule=None, n_jobs=2):
    noise, metric = channel.component_configs()
    return EvaluatorConfig(
        noise_channel=ComponentConfig.from_dict(noise),
        decoding_metric=ComponentConfig.from_dict(metric),
        decoding_rule=ComponentConfig.from_dict(rule or {"type": "unique_minimum"}),
        num_samples=num_samples, num_cpus=n_jobs, seed=seed,
    )


OPS_KW = dict(lambdas=LAMBDAS, seed=7, num_samples=300, eval_samples=200, device="cpu",
              workers=2, max_patience=200, verbose=False)
DESIGN_KW = {k: v for k, v in OPS_KW.items() if k != "lambdas"}
ACC_COLS = ["accuracy_mean", "accuracy_p10", "accuracy_p95_p5_ratio", "surrogate_accuracy",
            "n_codewords", "duplicate_codewords"]


@pytest.fixture(scope="module")
def ops_result(ops_pool):
    return duet.design_ops(ops_pool, duet.channels.symmetric(0.1), **OPS_KW)


@pytest.fixture(scope="module")
def decoding_result(ops_pool):
    return duet.design(ops_pool, duet.channels.symmetric(0.1), **DESIGN_KW)


@pytest.fixture(scope="module")
def ops_lambda1_result(ops_pool):
    return duet.design_ops(ops_pool, duet.channels.symmetric(0.1), lambdas=[1.0], **DESIGN_KW)


# ---------------------------------------------------------------------------
# Equivalence with the direct engine calls
# ---------------------------------------------------------------------------


class TestOpsEquivalence:
    def test_selections_match_run_duet_ops(self, ops_pool, ops_result):
        seeds = derive_stage_seeds(7)
        init, _ = RandomInit().get_initial_indices(ops_pool, seeds["init"])
        ch = duet.channels.symmetric(0.1)
        direct = run_duet_ops(
            candidates=ops_pool,
            pep_config=_pep_config(ch, 300, seeds["pep"]),
            optimizer_config=DuetOptimizerConfig(lambda_=LAMBDAS, max_patience=200, num_cpus=2),
            init=init, alphabet_size=4, seed=seeds["optimizer"], cache_dir=None, device="cpu",
        )
        for lam, idx in zip(direct["lambdas"], direct["best_indices"]):
            np.testing.assert_array_equal(ops_result.codebook(lam)["candidate"].to_numpy(), idx)
        np.testing.assert_array_equal(ops_result.codebook("initial")["candidate"].to_numpy(), init)

    def test_accuracies_equal_the_evaluation_function(self, ops_pool, ops_result):
        seeds = derive_stage_seeds(7)
        labels = ops_result.table["codebook"].tolist()
        assert labels == ["initial", "lambda=0", "lambda=0.3", "lambda=1", "max_score"]
        codebooks = [ops_result.codebook(lbl)["sequence"].tolist() for lbl in labels]
        results = evaluate_codebooks_by_sequence(
            codebooks, eval_config=_pep_config(duet.channels.symmetric(0.1), 200, seeds["evaluation"]),
            alphabet_size=4, n_jobs=1, device="cpu",
        )
        for lbl, res in zip(labels, results):
            np.testing.assert_array_equal(ops_result.codebook(lbl)["accuracy"].to_numpy(),
                                          res.codeword_accuracy)
            m = compute_metrics(res.codeword_accuracy)
            row = ops_result.table.set_index("codebook").loc[lbl]
            assert row["accuracy_mean"] == m["Mean decode accuracy"]
            assert row["accuracy_p10"] == m["10th percentile decode accuracy"]
            assert row["accuracy_p95_p5_ratio"] == m["95th/5th percentile ratio"]

    def test_duet_evaluate_reproduces_the_design_accuracies(self, ops_result):
        labels = ops_result.table["codebook"].tolist()
        acc = duet.evaluate({lbl: ops_result.codebook(lbl) for lbl in labels},
                            duet.channels.symmetric(0.1), num_samples=200, seed=7, device="cpu")
        for lbl in labels:
            got = acc.per_codeword.query("codebook == @lbl")["accuracy"].to_numpy()
            np.testing.assert_array_equal(got, ops_result.codebook(lbl)["accuracy"].to_numpy())
        assert acc.summary["accuracy_mean"].tolist() == ops_result.table["accuracy_mean"].tolist()

    def test_table_columns_and_secondary_objective(self, ops_pool, ops_result):
        t = ops_result.table
        for col in ("codebook", "kind", "lambda", "accuracy_mean", "accuracy_p10",
                    "accuracy_p95_p5_ratio", "surrogate_accuracy", "mean_score",
                    "n_codewords", "duplicate_codewords", "on_pareto_front"):
            assert col in t.columns
        for lbl in t["codebook"]:
            cb = ops_result.codebook(lbl)
            expect = float(np.mean(ops_pool.scores[cb["candidate"].to_numpy()]))
            assert t.set_index("codebook").loc[lbl, "mean_score"] == pytest.approx(expect, abs=1e-12)
        assert not t.loc[t["kind"] != "lambda", "on_pareto_front"].any()
        assert t.loc[t["kind"] == "lambda", "on_pareto_front"].any()

    def test_surrogate_matches_compute_duet_objective(self, ops_pool, ops_result):
        from duet.benchmark.metrics import compute_duet_objective

        seeds = derive_stage_seeds(7)
        run = run_duet_ops(
            candidates=ops_pool, pep_config=_pep_config(duet.channels.symmetric(0.1), 300, seeds["pep"]),
            optimizer_config=DuetOptimizerConfig(lambda_=[1.0], max_patience=1, max_iter=1),
            init=RandomInit().get_initial_indices(ops_pool, seeds["init"])[0], alphabet_size=4,
            cache_dir=None, device="cpu",
        )
        M = run["pep_count_matrix"].astype(float) / run["n_samples"]
        c2u = ops_pool.candidate_to_sequence_idx
        for lbl in ops_result.table["codebook"]:
            u = c2u[ops_result.codebook(lbl)["candidate"].to_numpy()]
            expect = compute_duet_objective(M[np.ix_(u, u)], np.arange(len(u)))
            got = ops_result.table.set_index("codebook").loc[lbl, "surrogate_accuracy"]
            assert got == pytest.approx(expect, rel=1e-12, abs=1e-12)


def _merfish_inputs(n_genes=10, n_bits=8, seed=3):
    rng = np.random.default_rng(seed)
    genes = pd.DataFrame({"gene": [f"G{i}" for i in range(n_genes)],
                          "cpm": rng.lognormal(3, 1, n_genes)})
    combos = list(itertools.combinations(range(n_bits), 4))
    weight4 = ["".join("1" if j in combos[i] else "0" for j in range(n_bits))
               for i in rng.permutation(len(combos))]
    init = pd.DataFrame({"gene": genes["gene"], "barcode": weight4[:n_genes]})
    published = pd.DataFrame({"gene": genes["gene"][:6], "barcode": weight4[n_genes:n_genes + 6]})
    return genes, init, published


MERFISH_CH = duet.channels.asymmetric([[0.96, 0.04], [0.1, 0.9]], alphabet="01")
MERFISH_KW = dict(gene="gene", expression="cpm", n_bits=8, hamming_weights={3: 20, 4: None},
                  candidates_per_gene=12, lambdas=LAMBDAS, seed=11, num_samples=300,
                  eval_samples=200, device="cpu", workers=2, max_patience=100,
                  avoid_duplicates=False, verbose=False)


@pytest.fixture(scope="module")
def merfish_result():
    genes, init, published = _merfish_inputs()
    return duet.design_merfish(genes, MERFISH_CH, init=init, include={"published": published},
                               **MERFISH_KW)


class TestMerfishEquivalence:
    def test_selections_match_run_duet_merfish(self, merfish_result):
        from duet.api import _build_merfish_pool
        from duet.initialization import _lookup_indices

        genes, init, published = _merfish_inputs()
        seeds = derive_stage_seeds(11)
        names = genes["gene"].tolist()
        pool = _build_merfish_pool(
            names, 8, {3: 20, 4: None}, 12, init["barcode"].tolist(),
            [("published", published["gene"].tolist(), published["barcode"].tolist())],
            seeds["pool"],
        )
        init_idx = _lookup_indices(init["barcode"].tolist(), pool.sequences)
        direct = run_duet_merfish(
            candidates=pool, pep_config=_pep_config(MERFISH_CH, 300, seeds["pep"]),
            optimizer_config=DuetOptimizerConfig(lambda_=LAMBDAS, max_patience=100,
                                                 avoid_duplicates=False, num_cpus=2),
            init=init_idx, codewords=pool.get_sequences_as_array(2).astype(np.float64),
            expression=genes["cpm"].to_numpy(dtype=np.float64), alphabet_size=2,
            seed=seeds["optimizer"], cache_dir=None, device="cpu",
        )
        for lam, idx in zip(direct["lambdas"], direct["best_indices"]):
            assert merfish_result.codebook(lam)["barcode"].tolist() == [pool.sequences[i] for i in idx]

    def test_table_and_codebooks(self, merfish_result):
        t = merfish_result.table
        assert t["codebook"].tolist() == ["initial", "lambda=0", "lambda=0.3", "lambda=1", "published"]
        assert t.set_index("codebook").loc["initial", "crowding"] == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(t.set_index("codebook").loc["published", "crowding"])  # covers 6 of 10 genes
        cb = merfish_result.codebook(0.3)
        assert list(cb.columns) == ["gene", "barcode", "expression", "accuracy"]
        assert cb["gene"].tolist() == [f"G{i}" for i in range(10)]
        assert len(merfish_result.codebook("published")) == 6
        assert merfish_result.provenance["merfish"]["crowding_anchor_C_S0"] > 0

    def test_accuracies_equal_the_evaluation_function(self, merfish_result):
        seeds = derive_stage_seeds(11)
        labels = merfish_result.table["codebook"].tolist()
        codebooks = [merfish_result.codebook(lbl)["barcode"].tolist() for lbl in labels]
        results = evaluate_codebooks_by_sequence(
            codebooks, eval_config=_pep_config(MERFISH_CH, 200, seeds["evaluation"]),
            alphabet_size=2, n_jobs=1, device="cpu",
        )
        for lbl, res in zip(labels, results):
            np.testing.assert_array_equal(merfish_result.codebook(lbl)["accuracy"].to_numpy(),
                                          res.codeword_accuracy)

    def test_operating_point_is_ops_only(self, merfish_result):
        with pytest.raises(ValueError, match="OPS"):
            merfish_result.operating_point()


# ---------------------------------------------------------------------------
# Storage, seeds, RNG hygiene
# ---------------------------------------------------------------------------


class TestStorageAndSeeds:
    def test_memory_and_mmap_storage_give_identical_selections(self, ops_pool, ops_result, tmp_path):
        kw = dict(OPS_KW, eval_samples=0)
        mem = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), pep_storage="memory", **kw)
        mm = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), pep_storage="mmap", **kw)
        cached = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **kw)
        assert mem.provenance["pep"]["storage"] == "memory"
        assert mm.provenance["pep"]["storage"] == "mmap"
        for lam in LAMBDAS:
            ref = ops_result.codebook(lam)["candidate"].tolist()
            assert mem.codebook(lam)["candidate"].tolist() == ref
            assert mm.codebook(lam)["candidate"].tolist() == ref
            assert cached.codebook(lam)["candidate"].tolist() == ref
        np.testing.assert_allclose(mm.table["surrogate_accuracy"], mem.table["surrogate_accuracy"],
                                   rtol=0, atol=1e-12)

    def test_cache_provenance_reports_the_computing_device(self, ops_pool, tmp_path):
        kw = dict(OPS_KW, eval_samples=0, lambdas=[1.0])
        first = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **kw)
        second = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **kw)
        assert first.provenance["pep"]["source"] == "computed"
        assert second.provenance["pep"]["source"] == "cache"
        assert second.provenance["pep"]["computed_on"] == "cpu"

    def test_stage_seeds_are_independent_and_recorded(self, ops_result):
        seeds = derive_stage_seeds(7)
        assert len(set(seeds.values())) == 5
        prov = ops_result.provenance
        assert prov["seed"] == 7 and prov["stage_seeds"] == seeds
        assert prov["stage_seed_order"] == ["pool", "init", "pep", "evaluation", "optimizer"]
        assert prov["devices"] == {"pep": "cpu", "lambda_sweep": "cpu", "evaluation": "cpu"}
        assert len(prov["pool"]["table_sha256"]) == 64

    def test_no_seed_draws_and_records_one(self, ops_pool):
        res = duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                              **dict(OPS_KW, seed=None, eval_samples=0, lambdas=[1.0]))
        again = duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                                **dict(OPS_KW, seed=res.provenance["seed"], eval_samples=0, lambdas=[1.0]))
        assert res.codebook(1.0)["candidate"].tolist() == again.codebook(1.0)["candidate"].tolist()

    def test_pep_and_evaluation_seeds_must_differ(self, ops_pool):
        with pytest.raises(ValueError, match="must not share a seed"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                            _stage_seeds={"pep": 5, "evaluation": 5}, **OPS_KW)

    def test_serial_run_leaves_the_callers_rng_untouched(self, ops_pool):
        np.random.seed(123)
        random.seed(456)
        expect_np, expect_py = np.random.random(), random.random()
        np.random.seed(123)
        random.seed(456)
        duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                        **dict(OPS_KW, workers=1, eval_samples=0))
        assert np.random.random() == expect_np
        assert random.random() == expect_py


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


class TestChannels:
    T_ACGT = np.array([
        [0.90, 0.05, 0.03, 0.02],
        [0.10, 0.80, 0.05, 0.05],
        [0.02, 0.03, 0.90, 0.05],
        [0.01, 0.01, 0.01, 0.97],
    ])

    def test_matrix_in_acgt_order_is_reordered_to_atcg(self):
        T = np.array(duet.channels.asymmetric(self.T_ACGT).params["channel_matrix"])
        perm = ["ACGT".index(s) for s in "ATCG"]
        np.testing.assert_array_equal(T, self.T_ACGT[np.ix_(perm, perm)])
        assert T[1, 1] == 0.97  # T -> T, internal index 1

    def test_labeled_dataframe_is_reordered_by_its_labels(self):
        df = pd.DataFrame(self.T_ACGT, index=list("ACGT"), columns=list("ACGT"))
        shuffled = df.loc[list("GTAC"), list("CATG")]
        a = duet.channels.asymmetric(df).params["channel_matrix"]
        b = duet.channels.asymmetric(shuffled).params["channel_matrix"]
        c = duet.channels.asymmetric(self.T_ACGT).params["channel_matrix"]
        assert a == b == c

    def test_alphabet_argument_sets_the_matrix_order(self):
        perm = ["ACGT".index(s) for s in "ATCG"]
        T_atcg = self.T_ACGT[np.ix_(perm, perm)]
        assert (duet.channels.asymmetric(T_atcg, alphabet="ATCG").params["channel_matrix"]
                == duet.channels.asymmetric(self.T_ACGT).params["channel_matrix"])

    def test_position_varying_asymmetric_reorders_every_position(self):
        tensor = np.stack([self.T_ACGT, self.T_ACGT.T / self.T_ACGT.T.sum(1, keepdims=True)])
        got = np.array(duet.channels.position_varying_asymmetric(tensor).params["channel_matrices"])
        perm = ["ACGT".index(s) for s in "ATCG"]
        for l in range(2):
            np.testing.assert_allclose(got[l], tensor[l][np.ix_(perm, perm)])

    def test_rectangular_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="only square channels"):
            duet.channels.asymmetric(np.full((4, 3), 1 / 3))
        with pytest.raises(ValueError, match="only square channels"):
            duet.channels.position_varying_asymmetric(np.full((5, 4, 3), 1 / 3))

    def test_matrix_rows_must_be_stochastic(self):
        with pytest.raises(ValueError, match="sum to 1"):
            duet.channels.asymmetric(self.T_ACGT * 0.5)

    def test_npy_path_is_accepted(self, tmp_path):
        np.save(tmp_path / "T.npy", self.T_ACGT)
        assert (duet.channels.asymmetric(tmp_path / "T.npy").params
                == duet.channels.asymmetric(self.T_ACGT).params)

    def test_metric_follows_the_channel(self):
        assert duet.channels.symmetric(0.1).component_configs()[1] == {"type": "hamming"}
        assert duet.channels.position_varying([0.1] * 3).component_configs()[1]["type"] == "position_varying_nll"
        assert duet.channels.asymmetric(self.T_ACGT).component_configs()[1]["type"] == "asymmetric_nll"


# ---------------------------------------------------------------------------
# CandidatePool.from_table and describe
# ---------------------------------------------------------------------------


class TestPool:
    def test_from_table_keeps_row_order(self):
        df = _ops_table(n_groups=3)
        pool = duet.CandidatePool.from_table(df, group="gene", sequence="barcode", score="activity", quota=2)
        assert pool.sequences == df["barcode"].tolist()
        assert list(pool.group_to_candidates) == list(dict.fromkeys(df["gene"]))

    def test_from_table_quota_forms(self):
        df = _ops_table(n_groups=3)
        by_col = duet.CandidatePool.from_table(df.assign(k=df["gene"].map({"g00": 1, "g01": 2, "g02": 3, "ctrl": 1})),
                                               group="gene", sequence="barcode", quota="k")
        by_map = duet.CandidatePool.from_table(df, group="gene", sequence="barcode",
                                               quota={"g00": 1, "g01": 2, "g02": 3, "ctrl": 1})
        assert by_col.quotas == by_map.quotas == {"g00": 1, "g01": 2, "g02": 3, "ctrl": 1}

    def test_seq_length_truncates(self):
        df = _ops_table(n_groups=3)
        pool = duet.CandidatePool.from_table(df, group="gene", sequence="barcode", seq_length=4)
        assert pool.seq_length == 4

    def test_describe_matches_the_pep_actually_built(self, ops_pool, tmp_path):
        d = ops_pool.describe(num_samples=300)
        U = len(ops_pool.unique_sequences)
        assert d.unique_codewords == U and d.duplicate_candidates == ops_pool.pool_size - U == 1
        assert d.groups == 25 and d.candidates == 122 and d.codebook_size == 50
        assert d.swaps_per_iteration == 24 * 2 * 4 + 2 * 1
        run = run_duet_ops(
            candidates=ops_pool, pep_config=_pep_config(duet.channels.symmetric(0.1), 300, 1),
            optimizer_config=DuetOptimizerConfig(lambda_=[1.0], max_patience=1, max_iter=1),
            init=RandomInit().get_initial_indices(ops_pool, 0)[0], alphabet_size=4,
            cache_dir=None, device="cpu",
        )
        assert d.pep_memory_bytes == run["pep_count_matrix"].nbytes
        run_duet_ops(
            candidates=ops_pool, pep_config=_pep_config(duet.channels.symmetric(0.1), 300, 1),
            optimizer_config=DuetOptimizerConfig(lambda_=[1.0], max_patience=1, max_iter=1),
            init=RandomInit().get_initial_indices(ops_pool, 0)[0], alphabet_size=4,
            cache_dir=tmp_path, use_mmap=True, device="cpu",
        )
        on_disk = sum(p.stat().st_size for p in tmp_path.iterdir() if p.suffix in (".npy", ".dat"))
        assert 0 <= on_disk - d.pep_disk_bytes <= 1024  # .npy header
        assert "unique codewords" in str(d)


# ---------------------------------------------------------------------------
# Validation: one test per error
# ---------------------------------------------------------------------------


class TestValidation:
    def test_symbol_outside_the_pool_alphabet(self):
        df = _ops_table(n_groups=2)
        df.loc[0, "barcode"] = "ACGTNNA"
        with pytest.raises(ValueError, match=r"symbols \['N'\] outside the alphabet"):
            duet.CandidatePool.from_table(df, group="gene", sequence="barcode")

    def test_symbol_outside_the_channel_alphabet(self, ops_pool):
        with pytest.raises(ValueError, match="must share an alphabet"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1, alphabet="01"), **OPS_KW)

    def test_mixed_sequence_lengths(self):
        df = _ops_table(n_groups=2)
        df.loc[0, "barcode"] = "ACG"
        with pytest.raises(ValueError, match="different lengths"):
            duet.CandidatePool.from_table(df, group="gene", sequence="barcode")

    def test_quota_above_group_size(self):
        with pytest.raises(ValueError, match="quota 6 but 5 candidates"):
            duet.CandidatePool.from_table(_ops_table(n_groups=2), group="gene", sequence="barcode",
                                          quota=6)

    @pytest.mark.parametrize("lambdas, message", [
        ([0.0, 1.5], r"must lie in \[0, 1\]"),
        ([0.5, 0.5], "repeats"),
        ([0.0, 0.0005], "share an optimizer seed"),
    ])
    def test_bad_lambdas(self, ops_pool, lambdas, message):
        with pytest.raises(ValueError, match=message):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), **dict(OPS_KW, lambdas=lambdas))

    def test_channel_length_must_match_truncated_sequences(self, ops_pool):
        with pytest.raises(ValueError, match="has 5 positions but the sequences have length 7"):
            duet.design_ops(ops_pool, duet.channels.position_varying([0.1] * 5), **OPS_KW)

    @pytest.mark.parametrize("key", ["num_samples", "eval_samples"])
    def test_samples_above_the_uint16_limit(self, ops_pool, key):
        with pytest.raises(ValueError, match="32,767"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), **dict(OPS_KW, **{key: 40_000}))

    def test_pep_that_does_not_fit_on_disk(self, ops_pool, tmp_path, monkeypatch):
        import shutil
        from collections import namedtuple

        Usage = namedtuple("Usage", "total used free")
        monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10, 10, 10))
        with pytest.raises(OSError, match=r"needs [0-9.]+ GB of free disk"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **OPS_KW)

    def test_pep_that_does_not_fit_in_memory(self, ops_pool, monkeypatch):
        import psutil

        class _VM:
            available = 1000

        monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
        with pytest.raises(MemoryError, match="pep_storage=\"mmap\""):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), pep_storage="memory", **OPS_KW)

    def test_initial_codebook_sequence_not_in_its_group(self, ops_pool, ops_result):
        init = ops_result.codebook(0.3)[["group", "sequence"]].copy()
        other = ops_pool.sequences[ops_pool.group_to_candidates["g01"][0]]
        init.loc[init["group"] == "g00", "sequence"] = other
        with pytest.raises(ValueError, match="is not a candidate of group 'g00'"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), init=init, **OPS_KW)

    def test_nan_scores_name_the_controls_rule(self):
        df = _ops_table(n_groups=2)
        df.loc[df["gene"] == "ctrl", "activity"] = np.nan
        with pytest.raises(ValueError, match="controls included.*1.0"):
            duet.CandidatePool.from_table(df, group="gene", sequence="barcode", score="activity")

    def test_gpu_device_without_cupy_names_the_extra(self, ops_pool, monkeypatch):
        monkeypatch.setattr("duet.gpu_utils.HAS_CUPY", False)
        with pytest.raises(ImportError, match=r"duet-codebook\[gpu\]"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), **dict(OPS_KW, device="gpu:all"))

    def test_auto_device_warns_about_large_problems_on_cpu(self, ops_pool, monkeypatch):
        monkeypatch.setattr("duet.gpu_utils.has_gpu", lambda: False)
        monkeypatch.setattr("duet.api.CPU_WARN_UNIQUE_CODEWORDS", 10)
        with pytest.warns(UserWarning, match=r"U = 121 unique codewords.*\[gpu\]"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                            **dict(OPS_KW, device="auto", eval_samples=0, lambdas=[1.0]))

    def test_pool_without_scores_warns_and_points_to_design(self):
        pool = duet.CandidatePool.from_table(_ops_table(n_groups=3), group="gene",
                                             sequence="barcode", quota=2, seq_length=7)
        with pytest.warns(UserWarning, match=r"same mean score.*duet\.design"):
            duet.design_ops(pool, duet.channels.symmetric(0.1),
                            **dict(OPS_KW, eval_samples=0, lambdas=[1.0]))

    def test_scores_constant_within_every_group_warn(self):
        df = _ops_table(n_groups=3)
        df["activity"] = df.groupby("gene")["activity"].transform("first")
        pool = duet.CandidatePool.from_table(df, group="gene", sequence="barcode",
                                             score="activity", quota=2, seq_length=7)
        assert np.unique(pool.scores).size > 1  # the groups differ; within each it is constant
        with pytest.warns(UserWarning, match=r"same mean score"):
            duet.design_ops(pool, duet.channels.symmetric(0.1),
                            **dict(OPS_KW, eval_samples=0, lambdas=[1.0]))

    def test_varying_scores_do_not_warn(self, ops_pool):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                            **dict(OPS_KW, eval_samples=0, lambdas=[1.0]))
        assert not [w for w in caught if "same mean score" in str(w.message)]

    def test_initial_codebook_from_a_result_is_accepted(self, ops_pool, ops_result):
        init = ops_result.codebook(1.0)
        res = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), init=init,
                              **dict(OPS_KW, eval_samples=0, lambdas=[1.0]))
        assert res.codebook("initial")["candidate"].tolist() == init["candidate"].tolist()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class TestResults:
    def test_save_and_load_round_trip(self, ops_result, tmp_path):
        ops_result.save(tmp_path / "out")
        back = duet.load(tmp_path / "out")
        pd.testing.assert_frame_equal(back.table, ops_result.table)
        for lbl in ops_result.table["codebook"]:
            pd.testing.assert_frame_equal(back.codebook(lbl), ops_result.codebook(lbl),
                                          check_dtype=False)
        assert back.provenance == ops_result.provenance
        assert back.pareto_front()["codebook"].tolist() == ops_result.pareto_front()["codebook"].tolist()

    def test_merfish_round_trip_keeps_barcodes_as_strings(self, merfish_result, tmp_path):
        merfish_result.save(tmp_path)
        back = duet.load(tmp_path)
        assert back.codebook(0.3)["barcode"].tolist() == merfish_result.codebook(0.3)["barcode"].tolist()

    def test_pareto_front_is_non_dominated(self, ops_result):
        front = ops_result.pareto_front()
        lam = ops_result.table[ops_result.table["kind"] == "lambda"]
        for _, r in front.iterrows():
            dominated = ((lam["accuracy_mean"] >= r["accuracy_mean"]) & (lam["mean_score"] >= r["mean_score"])
                         & ((lam["accuracy_mean"] > r["accuracy_mean"]) | (lam["mean_score"] > r["mean_score"])))
            assert not dominated.any()

    def test_operating_point_rule(self, ops_result):
        t = ops_result.table
        ref = float(t.loc[t["kind"] == "max_score", "mean_score"].iloc[0])
        lam = t[t["kind"] == "lambda"]
        eligible = lam[lam["mean_score"] >= 0.5 * ref]
        best = eligible.sort_values(["accuracy_mean", "lambda"], ascending=[False, True]).iloc[0]
        assert ops_result.operating_point(0.5)["codebook"] == best["codebook"]

    def test_plot_leaves_global_style_alone(self, ops_result):
        import matplotlib as mpl

        before = dict(mpl.rcParams)
        ax = ops_result.plot()
        assert ax.get_ylabel() == "Decoding accuracy"
        assert dict(mpl.rcParams) == before

    def test_non_dominated_helper(self):
        from duet.results import non_dominated

        assert non_dominated([0.9, 0.8, 0.7, 0.9], [0.1, 0.3, 0.2, 0.1]).tolist() == [True, True, False, True]


def test_evaluate_accepts_sequence_lists_and_tables():
    seqs = ["ACGTAC", "TTGACA", "GGGCCA", "CATCAT"]
    acc = duet.evaluate({"list": seqs, "table": pd.DataFrame({"gene": list("abcd"), "barcode": seqs})},
                        duet.channels.symmetric(0.1), num_samples=100, seed=1, device="cpu")
    a = acc.per_codeword.query("codebook == 'list'")["accuracy"].to_numpy()
    b = acc.per_codeword.query("codebook == 'table'")["accuracy"].to_numpy()
    np.testing.assert_array_equal(a, b)  # identical codebooks share reads
    assert acc.summary["n_codewords"].tolist() == [4, 4]
    assert "group" in acc.per_codeword.columns


def test_evaluate_rejects_symbols_outside_the_channel_alphabet():
    with pytest.raises(ValueError, match="outside the channel's alphabet"):
        duet.evaluate({"x": ["0101", "1100"]}, duet.channels.symmetric(0.1), num_samples=10, device="cpu")


# ---------------------------------------------------------------------------
# Review follow-ups (2026-09-24): storage above 256 codewords, forwarding of
# settings, independent checks of the secondary objective, front and rule
# ---------------------------------------------------------------------------


def _big_codebook_pool():
    """300 groups x 4 candidates, quota 1: |S| = 300 > 256, U = 1,200 (small PEP)."""
    rng = np.random.default_rng(4)
    barcodes = _distinct_barcodes(1200, 8, rng)
    df = pd.DataFrame({"gene": [f"g{i // 4:03d}" for i in range(1200)], "barcode": barcodes,
                       "activity": rng.uniform(0.2, 1.0, 1200)})
    return duet.CandidatePool.from_table(df, group="gene", sequence="barcode", score="activity")


class TestStorageAboveOneBatch:
    KW = dict(lambdas=[0.8, 0.95, 1.0], seed=0, num_samples=20, eval_samples=0, device="cpu",
              workers=3, verbose=False)

    def test_auto_uses_the_memory_mapped_path_and_memory_is_refused(self):
        pool = _big_codebook_pool()
        ch = duet.channels.symmetric(0.2)
        auto = duet.design_ops(pool, ch, **self.KW)
        mm = duet.design_ops(pool, ch, pep_storage="mmap", **self.KW)
        assert auto.provenance["pep"]["storage"] == "mmap"
        for lam in self.KW["lambdas"]:
            assert auto.codebook(lam)["candidate"].tolist() == mm.codebook(lam)["candidate"].tolist()
        with pytest.raises(ValueError, match="at most 256 codewords"):
            duet.design_ops(pool, ch, pep_storage="memory", **self.KW)

    def test_selections_equal_run_duet_ops_on_the_memory_mapped_path(self, tmp_path):
        pool = _big_codebook_pool()
        ch = duet.channels.symmetric(0.2)
        seeds = derive_stage_seeds(0)
        res = duet.design_ops(pool, ch, **self.KW)
        direct = run_duet_ops(
            candidates=pool, pep_config=_pep_config(ch, 20, seeds["pep"], n_jobs=3),
            optimizer_config=DuetOptimizerConfig(lambda_=self.KW["lambdas"], num_cpus=3),
            init=RandomInit().get_initial_indices(pool, seeds["init"])[0], alphabet_size=4,
            seed=seeds["optimizer"], cache_dir=tmp_path, use_mmap=True, device="cpu",
        )
        for lam, idx in zip(direct["lambdas"], direct["best_indices"]):
            np.testing.assert_array_equal(res.codebook(lam)["candidate"].to_numpy(), idx)


ENGINE_MARK = "<run_duet_ops called>"


def _spy_on_the_engine(monkeypatch):
    """Record the keywords of the OPS engine and evaluation calls; the engine spy prints ENGINE_MARK."""
    import duet.benchmark.metrics as metrics
    import duet.runner.ops as ops

    seen = {}
    real_run, real_eval = ops.run_duet_ops, metrics.evaluate_codebooks_by_sequence

    def spy_run(**kw):
        seen["run"] = kw
        print(ENGINE_MARK)
        return real_run(**kw)

    def spy_eval(codebooks, **kw):
        seen["eval"] = kw
        return real_eval(codebooks, **kw)

    monkeypatch.setattr(ops, "run_duet_ops", spy_run)
    monkeypatch.setattr(metrics, "evaluate_codebooks_by_sequence", spy_eval)
    return seen


class TestSettingsAreForwarded:
    def test_optimizer_rule_and_eval_channel_reach_the_engine(self, ops_pool, monkeypatch):
        seen = _spy_on_the_engine(monkeypatch)
        duet.design_ops(ops_pool, duet.channels.symmetric(0.1),
                        eval_channel=duet.channels.symmetric(0.2),
                        rule=duet.MarginDecoding(0.5), lambdas=[0.5], seed=3, num_samples=50,
                        eval_samples=40, device="cpu", workers=2, max_patience=17, max_iter=999,
                        avoid_duplicates=False, duplicate_penalty=3.5, temperature=0.25,
                        verbose=False)
        cfg = seen["run"]["optimizer_config"]
        assert (cfg.max_patience, cfg.max_iter, cfg.avoid_duplicates, cfg.duplicate_penalty,
                cfg.temperature, cfg.lambda_) == (17, 999, False, 3.5, 0.25, [0.5])
        assert seen["run"]["pep_config"].decoding_rule.to_dict() == {"type": "margin", "margin": 0.5}
        assert seen["run"]["pep_config"].num_samples == 50
        ev = seen["eval"]["eval_config"]
        assert ev.noise_channel.to_dict() == {"type": "symmetric", "epsilon": 0.2}
        assert ev.decoding_rule.to_dict() == {"type": "margin", "margin": 0.5}
        assert ev.num_samples == 40

    def test_binding_settings_match_the_direct_call(self, ops_pool):
        ch, rule = duet.channels.symmetric(0.1), duet.MarginDecoding(1.0)
        seeds = derive_stage_seeds(9)
        res = duet.design_ops(ops_pool, ch, rule=rule, lambdas=[0.2, 1.0], seed=9, num_samples=200,
                              eval_samples=0, device="cpu", workers=2, max_patience=5,
                              verbose=False)
        direct = run_duet_ops(
            candidates=ops_pool,
            pep_config=_pep_config(ch, 200, seeds["pep"], rule={"type": "margin", "margin": 1.0}),
            optimizer_config=DuetOptimizerConfig(lambda_=[0.2, 1.0], max_patience=5, num_cpus=2),
            init=RandomInit().get_initial_indices(ops_pool, seeds["init"])[0], alphabet_size=4,
            seed=seeds["optimizer"], cache_dir=None, device="cpu",
        )
        for lam, idx in zip(direct["lambdas"], direct["best_indices"]):
            np.testing.assert_array_equal(res.codebook(lam)["candidate"].to_numpy(), idx)

    def test_eval_channel_accuracies_equal_the_evaluation_function(self, ops_pool):
        design, judge = duet.channels.symmetric(0.1), duet.channels.symmetric(0.25)
        res = duet.design_ops(ops_pool, design, eval_channel=judge,
                              **dict(OPS_KW, lambdas=[1.0], max_patience=20))
        labels = res.table["codebook"].tolist()
        results = evaluate_codebooks_by_sequence(
            [res.codebook(lbl)["sequence"].tolist() for lbl in labels],
            eval_config=_pep_config(judge, 200, derive_stage_seeds(7)["evaluation"]),
            alphabet_size=4, n_jobs=1, device="cpu",
        )
        for lbl, r in zip(labels, results):
            np.testing.assert_array_equal(res.codebook(lbl)["accuracy"].to_numpy(), r.codeword_accuracy)


def test_merfish_crowding_equals_one_minus_c_over_c0(merfish_result):
    genes, init, _ = _merfish_inputs()
    expr = genes["cpm"].to_numpy(dtype=float)

    def C(barcodes):
        bits = np.array([[int(b) for b in bc] for bc in barcodes], dtype=float)
        tev = (expr[:, None] * bits).sum(axis=0)
        return float(np.sum(tev ** 2))

    c0 = C(init["barcode"])
    assert merfish_result.provenance["merfish"]["crowding_anchor_C_S0"] == pytest.approx(c0, rel=1e-12)
    rows = merfish_result.table.set_index("codebook")
    for lam in LAMBDAS:
        expect = 1.0 - C(merfish_result.codebook(lam)["barcode"]) / c0
        assert rows.loc[f"lambda={lam:g}", "crowding"] == pytest.approx(expect, rel=1e-9, abs=1e-12)


def test_front_flags_are_the_non_dominated_lambda_codebooks(ops_result, merfish_result):
    for res, sec in ((ops_result, "mean_score"), (merfish_result, "crowding")):
        lam = res.table[res.table["kind"] == "lambda"]
        acc, y = lam["accuracy_mean"].to_numpy(), lam[sec].to_numpy()
        expect = [not any((acc[j] >= acc[i]) and (y[j] >= y[i]) and ((acc[j] > acc[i]) or (y[j] > y[i]))
                          for j in range(len(acc))) for i in range(len(acc))]
        assert lam["on_pareto_front"].tolist() == expect


def _hand_made_result(lambdas, acc, score, ref_score):
    from duet.results import DesignResult

    rows = [{"codebook": f"lambda={l:g}", "kind": "lambda", "lambda": l, "accuracy_mean": a,
             "mean_score": s, "on_pareto_front": False} for l, a, s in zip(lambdas, acc, score)]
    rows.append({"codebook": "max_score", "kind": "max_score", "lambda": np.nan,
                 "accuracy_mean": 0.5, "mean_score": ref_score, "on_pareto_front": False})
    table = pd.DataFrame(rows)
    return DesignResult(kind="ops", table=table, codebooks={r: pd.DataFrame() for r in table["codebook"]},
                        provenance={})


class TestOperatingPoint:
    def test_score_fraction_excludes_more_accurate_codebooks(self):
        res = _hand_made_result([0.0, 0.1, 0.5, 1.0], [0.70, 0.80, 0.90, 0.95], [1.0, 0.98, 0.96, 0.5], 1.0)
        assert res.operating_point(0.975)["lambda"] == 0.1
        assert res.operating_point(0.95)["lambda"] == 0.5
        assert res.operating_point(0.0)["lambda"] == 1.0

    def test_ties_go_to_the_smallest_lambda(self):
        res = _hand_made_result([0.3, 0.1, 0.2], [0.9, 0.9, 0.8], [1.0, 1.0, 1.0], 1.0)
        assert res.operating_point(0.975)["lambda"] == 0.1

    def test_no_qualifying_codebook_raises(self):
        res = _hand_made_result([0.5, 1.0], [0.9, 0.95], [0.5, 0.4], 1.0)
        with pytest.raises(ValueError, match="no lambda codebook keeps"):
            res.operating_point(0.975)


def test_evaluate_keeps_the_given_codebook_order():
    zeta = ["ACGTAC", "TTGACA", "GGGCCA"]
    alpha = ["TTGACA", "CATCAT", "ACGAAC", "GGGCCA"]
    ch = duet.channels.symmetric(0.15)
    acc = duet.evaluate({"zeta": zeta, "alpha": alpha}, ch, num_samples=300, seed=2, device="cpu")
    assert acc.summary["codebook"].tolist() == ["zeta", "alpha"]
    ref = evaluate_codebooks_by_sequence(
        [zeta, alpha], eval_config=_pep_config(ch, 300, derive_stage_seeds(2)["evaluation"]),
        alphabet_size=4, n_jobs=1, device="cpu")
    for name, r in zip(["zeta", "alpha"], ref):
        np.testing.assert_array_equal(acc.per_codeword.query("codebook == @name")["accuracy"], r.codeword_accuracy)


def test_cached_pep_reports_the_device_that_computed_it(ops_pool, tmp_path):
    import json

    kw = dict(OPS_KW, eval_samples=0, lambdas=[1.0], pep_storage="mmap")
    first = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **kw)
    for meta in tmp_path.glob("*.json"):
        d = json.loads(meta.read_text())
        d["device"] = "gpu:all"  # as if another run had computed the counts on GPUs
        meta.write_text(json.dumps(d))
    second = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **kw)
    assert first.provenance["pep"]["computed_on"] == "cpu"
    assert second.provenance["pep"]["source"] == "cache"
    assert second.provenance["pep"]["computed_on"] == "gpu:all"
    assert second.provenance["devices"]["pep"] == "cpu"


def test_save_load_keeps_text_labels_and_the_codebook_is_reusable(tmp_path):
    df = _ops_table(n_groups=4)
    df["gene"] = df["gene"].map({"g00": "007", "g01": "NA", "g02": "1e3", "g03": "True", "ctrl": "ctrl"})
    pool = duet.CandidatePool.from_table(df, group="gene", sequence="barcode", score="activity", quota=1)
    res = duet.design_ops(pool, duet.channels.symmetric(0.1), **dict(OPS_KW, eval_samples=50))
    res.save(tmp_path)
    back = duet.load(tmp_path)
    assert back.codebook(1.0)["group"].tolist() == res.codebook(1.0)["group"].tolist()
    assert back.codebook(1.0)["group"].tolist()[:4] == ["007", "NA", "1e3", "True"]
    again = duet.design_ops(pool, duet.channels.symmetric(0.1), init=back.codebook(1.0),
                            **dict(OPS_KW, eval_samples=0, lambdas=[1.0]))
    assert again.codebook("initial")["candidate"].tolist() == res.codebook(1.0)["candidate"].tolist()


class TestInitialCodebookTable:
    def _pool(self):
        df = pd.DataFrame({
            "gene": ["A", "A", "A", "B", "B", "B"],
            "guide": ["ACGTACGTACGGGGGGGGGG", "ACGTACGTACTTTTTTTTTT", "CCCCAAAATTGGGGAAAACC",
                      "TTTTGGGGCCAAAACCCCGG", "GGGGTTTTAACCCCAAAATT", "AAAACCCCGGTTTTGGGGCC"],
            "activity": [0.2, 0.9, 0.5, 0.4, 0.6, 0.7],
        })
        return df, duet.CandidatePool.from_table(df, group="gene", sequence="guide", score="activity",
                                                 quota=1, seq_length=10)

    def test_full_length_sequences_pick_the_right_guide(self):
        df, pool = self._pool()
        res = duet.design_ops(pool, duet.channels.symmetric(0.1), init=df.iloc[[1, 4]],
                              **dict(OPS_KW, eval_samples=0, lambdas=[1.0]))
        assert res.codebook("initial")["candidate"].tolist() == [1, 4]

    def test_an_ambiguous_truncated_codeword_is_refused(self):
        df, pool = self._pool()
        init = pd.DataFrame({"gene": ["A", "B"], "guide": ["ACGTACGTAC", "GGGGTTTTAA"]})
        with pytest.raises(ValueError, match="matches several candidates"):
            duet.design_ops(pool, duet.channels.symmetric(0.1), init=init, **OPS_KW)

    def test_candidate_and_sequence_must_agree(self):
        df, pool = self._pool()
        init = df.iloc[[1, 4]].assign(candidate=[2, 4])
        with pytest.raises(ValueError, match="different candidate pool"):
            duet.design_ops(pool, duet.channels.symmetric(0.1), init=init, **OPS_KW)


class TestMoreValidation:
    def test_infinite_scores(self):
        df = _ops_table(n_groups=2)
        df.loc[0, "activity"] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            duet.CandidatePool.from_table(df, group="gene", sequence="barcode", score="activity")

    def test_all_quotas_zero(self):
        pool = duet.CandidatePool.from_table(_ops_table(n_groups=2), group="gene", sequence="barcode",
                                             quota=0)
        with pytest.raises(ValueError, match="codebook would be empty"):
            duet.design_ops(pool, duet.channels.symmetric(0.1), **OPS_KW)

    def test_negative_margin(self, ops_pool):
        with pytest.raises(ValueError, match="margin k must be finite and >= 0"):
            duet.design_ops(ops_pool, duet.channels.symmetric(0.1), rule=duet.MarginDecoding(-1.0), **OPS_KW)

    def test_bare_string_codebook(self):
        with pytest.raises(ValueError, match="single string"):
            duet.evaluate({"x": "ACGT"}, duet.channels.symmetric(0.1), num_samples=10, device="cpu")

    def test_dataframe_channel_with_integer_labels(self):
        T = pd.DataFrame([[0.9, 0.1], [0.2, 0.8]], index=[0, 1], columns=[0, 1])
        assert duet.channels.asymmetric(T, alphabet="01").params["channel_matrix"] == [[0.9, 0.1], [0.2, 0.8]]


# ---------------------------------------------------------------------------
# Decoding-only design (duet.design, 0.2.0)
# ---------------------------------------------------------------------------


def _cands(res, label):
    return res.codebook(label)["candidate"].to_numpy()


class TestDecodingDesign:
    def test_selections_and_surrogate_equal_design_ops_at_lambda_1(self, decoding_result,
                                                                    ops_lambda1_result, ops_result):
        np.testing.assert_array_equal(_cands(decoding_result, "initial"),
                                      _cands(ops_lambda1_result, "initial"))
        np.testing.assert_array_equal(_cands(decoding_result, "designed"),
                                      _cands(ops_lambda1_result, "lambda=1"))
        d = decoding_result.table.set_index("codebook")
        o = ops_lambda1_result.table.set_index("codebook")
        assert d.loc["initial", "surrogate_accuracy"] == o.loc["initial", "surrogate_accuracy"]
        assert d.loc["designed", "surrogate_accuracy"] == o.loc["lambda=1", "surrogate_accuracy"]
        # Also the lambda = 1 row of a 3-lambda sweep: holds because the
        # per-lambda seed is seed + int(1000 * lambda) and the OPS search uses
        # no BLAS. Observed and kept, but not promised in the docs.
        np.testing.assert_array_equal(_cands(decoding_result, "designed"), _cands(ops_result, 1.0))

    def test_selections_match_run_duet_ops(self, ops_pool, decoding_result):
        seeds = derive_stage_seeds(7)
        init, _ = RandomInit().get_initial_indices(ops_pool, seeds["init"])
        direct = run_duet_ops(
            candidates=ops_pool,
            pep_config=_pep_config(duet.channels.symmetric(0.1), 300, seeds["pep"]),
            optimizer_config=DuetOptimizerConfig(lambda_=[1.0], max_patience=200, num_cpus=1),
            init=init, alphabet_size=4, seed=seeds["optimizer"], cache_dir=None, device="cpu",
        )
        np.testing.assert_array_equal(_cands(decoding_result, "designed"), direct["best_indices"][0])
        np.testing.assert_array_equal(_cands(decoding_result, "initial"), init)

    def test_accuracies_equal_design_ops_when_the_evaluation_union_agrees(self, decoding_result,
                                                                          ops_lambda1_result):
        mine = [decoding_result.codebook(lbl)["sequence"].tolist() for lbl in ("initial", "designed")]
        theirs = [ops_lambda1_result.codebook(lbl)["sequence"].tolist()
                  for lbl in ("initial", "lambda=1", "max_score")]
        u_mine, _ = _build_multiset_union(mine)
        u_theirs, _ = _build_multiset_union(theirs)
        # Reads are seeded per position of the evaluation union; the accuracies
        # agree only when max_score adds no codeword copy ahead of the shared part.
        assert u_theirs[:len(u_mine)] == u_mine, "precondition: max_score must not shift the evaluation union"
        d = decoding_result.table.set_index("codebook").loc[["initial", "designed"], ACC_COLS]
        o = ops_lambda1_result.table.set_index("codebook").loc[["initial", "lambda=1"], ACC_COLS]
        o.index = d.index
        pd.testing.assert_frame_equal(d, o, check_exact=True)
        for a, b in (("initial", "initial"), ("designed", "lambda=1")):
            np.testing.assert_array_equal(decoding_result.codebook(a)["accuracy"].to_numpy(),
                                          ops_lambda1_result.codebook(b)["accuracy"].to_numpy())

    def test_accuracies_equal_the_evaluation_function(self, decoding_result):
        order = decoding_result.provenance["evaluation"]["codebook_order"]
        assert order == ["initial", "designed"]
        results = evaluate_codebooks_by_sequence(
            [decoding_result.codebook(lbl)["sequence"].tolist() for lbl in order],
            eval_config=_pep_config(duet.channels.symmetric(0.1), 200, derive_stage_seeds(7)["evaluation"]),
            alphabet_size=4, n_jobs=1, device="cpu",
        )
        t = decoding_result.table.set_index("codebook")
        for lbl, r in zip(order, results):
            np.testing.assert_array_equal(decoding_result.codebook(lbl)["accuracy"].to_numpy(),
                                          r.codeword_accuracy)
            assert t.loc[lbl, "accuracy_mean"] == compute_metrics(r.codeword_accuracy)["Mean decode accuracy"]
        ev = duet.evaluate({lbl: decoding_result.codebook(lbl) for lbl in order},
                           duet.channels.symmetric(0.1), num_samples=200, seed=7, device="cpu")
        assert ev.summary["accuracy_mean"].tolist() == decoding_result.table["accuracy_mean"].tolist()

    def test_table_and_codebooks(self, decoding_result):
        t = decoding_result.table
        assert list(t.columns) == ["codebook", "kind", "accuracy_mean", "accuracy_p10",
                                   "accuracy_p95_p5_ratio", "surrogate_accuracy", "n_codewords",
                                   "duplicate_codewords"]
        assert t["codebook"].tolist() == t["kind"].tolist() == ["initial", "designed"]
        assert list(decoding_result.codebook().columns) == ["group", "sequence", "accuracy", "candidate"]
        pd.testing.assert_frame_equal(decoding_result.codebook(), decoding_result.codebook("designed"))
        assert decoding_result.kind == "decoding"
        assert decoding_result.secondary_objective is None and decoding_result.front_axis is None
        rows = t.set_index("codebook")
        assert rows.loc["designed", "surrogate_accuracy"] >= rows.loc["initial", "surrogate_accuracy"]

    def test_provenance(self, decoding_result, ops_result):
        p = decoding_result.provenance
        assert list(p) == list(ops_result.provenance)
        assert p["design"] == "decoding" and p["lambdas"] == [1.0] and p["front_axis"] is None
        assert p["stage_seeds"] == derive_stage_seeds(7)
        assert p["workers"] == {"lambda_sweep": 1, "cpu_processes": 2}
        assert p["init"] == {"type": "random", "seed": derive_stage_seeds(7)["init"]}

    def test_sweep_helpers_are_refused(self, decoding_result):
        for method in ("pareto_front", "operating_point", "plot"):
            with pytest.raises(ValueError, match=r"decoding-only design .*res\.codebook\(\)"):
                getattr(decoding_result, method)()
        with pytest.raises(KeyError, match="no lambda values"):
            decoding_result.codebook(1.0)
        with pytest.raises(KeyError, match="no codebook labeled 'lambda=1'"):
            decoding_result.codebook("lambda=1")

    def test_sweep_results_still_need_which(self, ops_result, merfish_result):
        for res in (ops_result, merfish_result):
            with pytest.raises(TypeError, match="needs a lambda value or a row label"):
                res.codebook()
        assert ops_result.secondary_objective == "mean_score"
        assert merfish_result.secondary_objective == "crowding"
        with pytest.raises(ValueError, match="OPS"):
            merfish_result.operating_point()

    def test_existing_tables_keep_their_columns(self, ops_result, merfish_result):
        base = ["codebook", "kind", "lambda", "accuracy_mean", "accuracy_p10", "accuracy_p95_p5_ratio",
                "surrogate_accuracy"]
        tail = ["n_codewords", "duplicate_codewords", "on_pareto_front"]
        assert list(ops_result.table.columns) == base + ["mean_score"] + tail
        assert list(merfish_result.table.columns) == base + ["crowding"] + tail
        for res in (ops_result, merfish_result):
            assert res.front_axis == res.provenance["front_axis"] == "accuracy_mean"

    def test_save_and_load_round_trip(self, decoding_result, tmp_path):
        decoding_result.save(tmp_path / "out")
        assert json.loads((tmp_path / "out" / "settings.json").read_text())["kind"] == "decoding"
        back = duet.load(tmp_path / "out")
        assert back.kind == "decoding"
        pd.testing.assert_frame_equal(back.table, decoding_result.table)
        for lbl in ("initial", "designed"):
            pd.testing.assert_frame_equal(back.codebook(lbl), decoding_result.codebook(lbl),
                                          check_dtype=False)
        pd.testing.assert_frame_equal(back.codebook(), decoding_result.codebook(), check_dtype=False)
        assert back.provenance == decoding_result.provenance
        assert back.secondary_objective is None
        with pytest.raises(ValueError, match="decoding-only design"):
            back.pareto_front()

    def test_init_from_a_table(self, ops_pool, decoding_result):
        init = decoding_result.codebook()
        kw = dict(DESIGN_KW, eval_samples=0)
        res = duet.design(ops_pool, duet.channels.symmetric(0.1), init=init, **kw)
        assert res.codebook("initial")["candidate"].tolist() == init["candidate"].tolist()
        assert res.table[["accuracy_mean", "accuracy_p10", "accuracy_p95_p5_ratio"]].isna().all().all()
        assert res.provenance["init"] == {"type": "table", "rows": 50}
        again = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), init=init, lambdas=[1.0], **kw)
        assert again.codebook("initial")["candidate"].tolist() == init["candidate"].tolist()

    def test_fixed_candidates_stay(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"set": ["in_use"] * 4 + ["new"] * 60,
                           "barcode": _distinct_barcodes(64, 6, rng)})
        pool = duet.CandidatePool.from_table(df, group="set", sequence="barcode",
                                             quota={"in_use": 4, "new": 8})
        res = duet.design(pool, duet.channels.symmetric(0.1), **dict(DESIGN_KW, eval_samples=0))
        book = res.codebook()
        assert sorted(book.loc[book["group"] == "in_use", "candidate"]) == [0, 1, 2, 3]

    def test_binary_pool(self, tmp_path):
        rng = np.random.default_rng(2)
        codes = rng.choice(2 ** 10, size=40, replace=False)
        df = pd.DataFrame({"set": [f"s{i // 4}" for i in range(40)],
                           "barcode": [format(int(c), "010b") for c in codes]})
        pool = duet.CandidatePool.from_table(df, group="set", sequence="barcode", quota=1, alphabet="01")
        ch = duet.channels.asymmetric([[0.96, 0.04], [0.1, 0.9]], alphabet="01")
        kw = dict(DESIGN_KW, num_samples=200)
        res = duet.design(pool, ch, **kw)
        with pytest.warns(UserWarning, match="same mean score"):  # the pool has no score
            ref = duet.design_ops(pool, ch, lambdas=[1.0], **kw)
        assert res.codebook()["candidate"].tolist() == ref.codebook(1.0)["candidate"].tolist()
        res.save(tmp_path)
        assert duet.load(tmp_path).codebook()["sequence"].tolist() == res.codebook()["sequence"].tolist()

    def test_memory_mapped_path_above_one_batch(self):
        pool = _big_codebook_pool()
        ch = duet.channels.symmetric(0.2)
        kw = dict(seed=0, num_samples=20, eval_samples=0, device="cpu", workers=3, verbose=False)
        res = duet.design(pool, ch, **kw)
        ref = duet.design_ops(pool, ch, lambdas=[1.0], **kw)
        assert res.provenance["pep"]["storage"] == "mmap"
        assert res.codebook()["candidate"].tolist() == ref.codebook(1.0)["candidate"].tolist()

    @pytest.mark.parametrize("init, message", [
        ("best_score", "not offered"),
        ("other", r'"random" or a codebook table'),
        (np.arange(50), r'"random" or a codebook table.*ndarray'),
    ])
    def test_init_must_be_random_or_a_table(self, ops_pool, init, message):
        with pytest.raises(ValueError, match=message):
            duet.design(ops_pool, duet.channels.symmetric(0.1), init=init, **DESIGN_KW)

    def test_lambdas_is_not_a_keyword(self, ops_pool):
        with pytest.raises(TypeError, match="unexpected keyword argument 'lambdas'"):
            duet.design(ops_pool, duet.channels.symmetric(0.1), **OPS_KW)

    def test_keywords_are_those_of_design_ops_without_lambdas(self):
        def params(f):
            return [(p.name, p.kind, p.default) for p in inspect.signature(f).parameters.values()]

        assert params(duet.design) == [p for p in params(duet.design_ops) if p[0] != "lambdas"]

    @pytest.mark.parametrize("verbose", [False, True])
    def test_settings_reach_the_engine(self, ops_pool, monkeypatch, capsys, verbose):
        seen = _spy_on_the_engine(monkeypatch)
        res = duet.design(ops_pool, duet.channels.symmetric(0.1),
                          eval_channel=duet.channels.symmetric(0.2),
                          rule=duet.MarginDecoding(0.5), seed=3, num_samples=50, eval_samples=40,
                          device="cpu", workers=2, max_patience=17, max_iter=999,
                          avoid_duplicates=False, duplicate_penalty=3.5, temperature=0.25,
                          pep_storage="mmap", verbose=verbose)
        cfg = seen["run"]["optimizer_config"]
        assert (cfg.max_patience, cfg.max_iter, cfg.avoid_duplicates, cfg.duplicate_penalty,
                cfg.temperature, cfg.lambda_, cfg.num_cpus) == (17, 999, False, 3.5, 0.25, [1.0], 1)
        pep = seen["run"]["pep_config"]
        assert pep.decoding_rule.to_dict() == {"type": "margin", "margin": 0.5}
        assert (pep.noise_channel.to_dict(), pep.num_samples, pep.num_cpus) == (
            {"type": "symmetric", "epsilon": 0.1}, 50, 2)
        assert seen["run"]["seed"] == derive_stage_seeds(3)["optimizer"]
        assert seen["run"]["use_mmap"] is True and res.provenance["pep"]["storage"] == "mmap"
        ev = seen["eval"]["eval_config"]
        assert ev.noise_channel.to_dict() == {"type": "symmetric", "epsilon": 0.2}
        assert ev.decoding_rule.to_dict() == {"type": "margin", "margin": 0.5}
        assert ev.num_samples == 40
        # verbose=False sends the engine's output to os.devnull.
        assert (ENGINE_MARK in capsys.readouterr().out) is verbose

    def test_validation_as_in_design_ops(self, ops_pool):
        ch = duet.channels.symmetric(0.1)
        with pytest.raises(TypeError, match="CandidatePool"):
            duet.design(_ops_table(), ch, **DESIGN_KW)
        with pytest.raises(ValueError, match="must share an alphabet"):
            duet.design(ops_pool, duet.channels.symmetric(0.1, alphabet="01"), **DESIGN_KW)
        # A bad design channel with a good eval_channel: the design channel's own check.
        with pytest.raises(ValueError, match="must share an alphabet"):
            duet.design(ops_pool, duet.channels.symmetric(0.1, alphabet="01"), eval_channel=ch,
                        **DESIGN_KW)
        with pytest.raises(ValueError, match="must share an alphabet"):
            duet.design(ops_pool, ch, eval_channel=duet.channels.symmetric(0.1, alphabet="01"),
                        **DESIGN_KW)
        # from_table refuses non-finite scores and the constructor refuses NaN; an
        # infinite score in a pool built with the constructor reaches design's own check.
        scores = np.array(ops_pool.scores, dtype=float)
        scores[0] = np.inf
        inf_pool = duet.CandidatePool(sequences=ops_pool.sequences,
                                      group_to_candidates=ops_pool.group_to_candidates,
                                      quotas=ops_pool.quotas, scores=scores)
        with pytest.raises(ValueError, match="scores must be finite"):
            duet.design(inf_pool, ch, **DESIGN_KW)
        for key in ("num_samples", "eval_samples"):
            with pytest.raises(ValueError, match="32,767"):
                duet.design(ops_pool, ch, **dict(DESIGN_KW, **{key: 40_000}))
        empty = duet.CandidatePool.from_table(_ops_table(n_groups=2), group="gene", sequence="barcode",
                                              quota=0)
        with pytest.raises(ValueError, match="codebook would be empty"):
            duet.design(empty, ch, **DESIGN_KW)
        with pytest.raises(ValueError, match="must not share a seed"):
            duet.design(ops_pool, ch, _stage_seeds={"pep": 5, "evaluation": 5}, **DESIGN_KW)

    def test_serial_run_leaves_the_callers_rng_untouched(self, ops_pool):
        np.random.seed(123)
        random.seed(456)
        expect_np, expect_py = np.random.random(), random.random()
        np.random.seed(123)
        random.seed(456)
        duet.design(ops_pool, duet.channels.symmetric(0.1), **dict(DESIGN_KW, eval_samples=0))
        assert np.random.random() == expect_np
        assert random.random() == expect_py

    def test_pep_cache_is_shared_with_design_ops(self, ops_pool, decoding_result, tmp_path):
        kw = dict(DESIGN_KW, eval_samples=0)
        first = duet.design(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path, **kw)
        second = duet.design_ops(ops_pool, duet.channels.symmetric(0.1), cache_dir=tmp_path,
                                 lambdas=[1.0], **kw)
        assert first.provenance["pep"]["source"] == "computed"
        assert second.provenance["pep"]["source"] == "cache"
        assert second.codebook(1.0)["candidate"].tolist() == _cands(decoding_result, "designed").tolist()

    def test_design_is_public(self):
        import duet.api

        assert "design" in duet.__all__ and "design" in duet.api.__all__
        assert duet.design is duet.api.design


@pytest.mark.parametrize("module", ["duet.api", "duet.channels", "duet.results", "duet.candidate_pool"])
def test_docstring_examples_run(module):
    import doctest
    import importlib

    result = doctest.testmod(importlib.import_module(module), extraglobs={"duet": duet},
                             optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
    assert result.failed == 0 and result.attempted > 0

"""Tests for CRISPickFactory score_method parameter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from duet.crispick_factory import CRISPickFactory


@pytest.fixture
def crispick_csv(tmp_path):
    """Create a minimal CRISPick CSV with known values.

    Gene GENE_A: 3 guides, Picking Rounds 1, 1, 3, Pick Orders 1, 2, 3
    Gene GENE_B: 3 guides, Picking Rounds 1, 2, 3, Pick Orders 1, 2, 3
    Control NO_SITE: 2 guides, NaN Picking Round / Pick Order
    """
    rows = [
        # GENE_A: two round-1 guides, one round-3 guide
        ("GENE_A", "AAAAAAAAAA", 1, 1),
        ("GENE_A", "AAAAAAAAAC", 2, 1),
        ("GENE_A", "AAAAAAAAAG", 3, 3),
        # GENE_B: one guide per round 1, 2, 3
        ("GENE_B", "CCCCCCCCCC", 1, 1),
        ("GENE_B", "CCCCCCCCCA", 2, 2),
        ("GENE_B", "CCCCCCCCCG", 3, 3),
        # Controls
        ("NO_SITE", "TTTTTTTTTT", float("nan"), float("nan")),
        ("NO_SITE", "TTTTTTTTTC", float("nan"), float("nan")),
    ]
    df = pd.DataFrame(rows, columns=[
        "Target Gene ID", "sgRNA Sequence", "Pick Order", "Picking Round",
    ])
    path = tmp_path / "crispick_test.csv"
    df.to_csv(path, index=False)
    return path


class TestPickingRoundScoring:
    """Tests for the default picking_round score method."""

    def test_default_score_method_is_picking_round(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, control_quotas={"NO_SITE": 2},
        )
        assert factory.score_method == "picking_round"

    def test_picking_round_scores(self, crispick_csv):
        """Verify linear inversion: score = (max_round + 1 - round) / max_round."""
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        df = pool.to_dataframe()

        # max_round = 3, so round 1 -> (3+1-1)/3 = 1.0, round 2 -> 2/3, round 3 -> 1/3
        gene_a = df[df["Group"] == "GENE_A"].sort_values("Score", ascending=False)
        np.testing.assert_allclose(gene_a["Score"].values, [1.0, 1.0, 1 / 3], rtol=1e-10)

        gene_b = df[df["Group"] == "GENE_B"].sort_values("Score", ascending=False)
        np.testing.assert_allclose(gene_b["Score"].values, [1.0, 2 / 3, 1 / 3], rtol=1e-10)

    def test_control_scores_filled(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        df = pool.to_dataframe()
        ctrl = df[df["Group"] == "NO_SITE"]
        np.testing.assert_allclose(ctrl["Score"].values, [1.0, 1.0])


class TestNormalizedRankScoring:
    """Tests for the normalized_rank score method."""

    def test_normalized_rank_scores(self, crispick_csv):
        """Verify score = (min_rank - rank) / (min_rank - 1)."""
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, score_method="normalized_rank",
            control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        df = pool.to_dataframe()

        # min_rank=3: rank 1 -> (3-1)/(3-1)=1.0, rank 2 -> (3-2)/2=0.5, rank 3 -> 0.0
        gene_a = df[df["Group"] == "GENE_A"].sort_values("Score", ascending=False)
        np.testing.assert_allclose(gene_a["Score"].values, [1.0, 0.5, 0.0])

        gene_b = df[df["Group"] == "GENE_B"].sort_values("Score", ascending=False)
        np.testing.assert_allclose(gene_b["Score"].values, [1.0, 0.5, 0.0])

    def test_normalized_rank_control_scores_filled(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, score_method="normalized_rank",
            control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        df = pool.to_dataframe()
        ctrl = df[df["Group"] == "NO_SITE"]
        np.testing.assert_allclose(ctrl["Score"].values, [1.0, 1.0])

    def test_min_rank_one_gives_all_ones(self, tmp_path):
        """Edge case: min_rank=1 means all guides are rank 1 -> score 1.0."""
        rows = [
            ("GENE_A", "AAAAAAAAAA", 1, 1),
            ("GENE_B", "CCCCCCCCCC", 1, 1),
        ]
        df = pd.DataFrame(rows, columns=[
            "Target Gene ID", "sgRNA Sequence", "Pick Order", "Picking Round",
        ])
        path = tmp_path / "crispick_min1.csv"
        df.to_csv(path, index=False)

        factory = CRISPickFactory(
            csv_path=path, seq_rounds=10, quota=1,
            min_rank=1, score_method="normalized_rank",
            control_quotas={},
        )
        pool = factory.create()
        np.testing.assert_allclose(pool.scores, [1.0, 1.0])

    def test_scores_evenly_spaced(self, crispick_csv):
        """Normalized rank scores should be evenly spaced within [0, 1]."""
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, score_method="normalized_rank",
            control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        df = pool.to_dataframe()
        gene_scores = df[df["Group"] == "GENE_A"]["Score"].sort_values(ascending=False).values
        diffs = np.diff(gene_scores)
        np.testing.assert_allclose(diffs, diffs[0])


class TestScoreMethodValidation:
    """Tests for invalid score_method values."""

    def test_invalid_score_method_raises(self, crispick_csv):
        with pytest.raises(ValueError, match="score_method"):
            CRISPickFactory(
                csv_path=crispick_csv, seq_rounds=10, quota=2,
                min_rank=3, score_method="invalid",
                control_quotas={"NO_SITE": 2},
            )


class TestMetadata:
    """Test that score_method is stored in metadata."""

    def test_score_method_in_metadata(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, score_method="normalized_rank",
            control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        assert pool.metadata["score_method"] == "normalized_rank"

    def test_default_score_method_in_metadata(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        assert pool.metadata["score_method"] == "picking_round"

    def test_control_groups_metadata(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, control_quotas={"NO_SITE": 2},
        )
        pool = factory.create()
        assert pool.metadata["control_groups"] == {"NO_SITE"}

    def test_control_groups_metadata_multiple(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2,
            min_rank=3, control_quotas={"NO_SITE": 1, "ONE_SITE_INTERGENIC": 1},
        )
        pool = factory.create()
        assert pool.metadata["control_groups"] == {"NO_SITE", "ONE_SITE_INTERGENIC"}


class TestFeldmanSourceDataframe:
    """The Feldman et al. input keeps both control groups, ranked first."""

    def test_control_groups_keep_their_names(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=10, quota=2, min_rank=3,
            control_quotas={"NO_SITE": 1},
        )
        pool = factory.create()
        df_full = pool.metadata["source_dataframe"]
        assert "negative_control" not in set(df_full["Gene"])
        assert set(df_full["Gene"]) == set(pool.quotas)

    def test_control_rank_is_one_and_gene_ranks_unchanged(self, crispick_csv):
        factory = CRISPickFactory(
            csv_path=crispick_csv, seq_rounds=8, quota=2, min_rank=3,
            control_quotas={"NO_SITE": 1},
        )
        pool = factory.create()
        df_full = pool.metadata["source_dataframe"]
        ctrl = df_full["Gene"] == "NO_SITE"
        assert (df_full.loc[ctrl, "Rank"] == 1).all()
        assert df_full.loc[~ctrl, "Rank"].tolist() == [1, 2, 3, 1, 2, 3]
        # Feldman gets the full 10-mers; the pool's codewords are 8-mers.
        assert set(df_full["Sequence"].str.len()) == {10}
        assert set(pool.to_dataframe()["Sequence"].str.len()) == {8}

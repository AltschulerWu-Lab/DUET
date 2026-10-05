"""Tests for synthetic source in candidate pool factory."""
import numpy as np
import pytest

from duet.candidate_pool_factory import create_pool_from_source


class TestSyntheticSource:
    """Test create_pool_from_source with source='synthetic'."""

    def test_basic_creation(self):
        pool = create_pool_from_source(
            source="synthetic",
            seq_rounds=7,
            quota=1,
            num_controls=0,
            num_groups=10,
            candidates_per_group=5,
            alphabet_size=4,
            seed=42,
        )
        assert pool.pool_size == 50
        assert pool.num_groups == 10
        assert pool.seq_length == 7
        assert pool.total_selections == 10

    def test_sequences_are_dna_strings(self):
        pool = create_pool_from_source(
            source="synthetic",
            seq_rounds=5,
            quota=1,
            num_controls=0,
            num_groups=3,
            candidates_per_group=4,
            alphabet_size=4,
            seed=42,
        )
        assert isinstance(pool.sequences, list)
        assert all(isinstance(s, str) for s in pool.sequences)
        valid_chars = set("ATCG")
        for seq in pool.sequences:
            assert len(seq) == 5
            assert set(seq).issubset(valid_chars), f"Invalid chars in {seq}"

    def test_csv_roundtrip(self):
        """Verify create -> to_dataframe -> from_dataframe roundtrip."""
        pool = create_pool_from_source(
            source="synthetic",
            seq_rounds=5,
            quota=1,
            num_controls=0,
            num_groups=3,
            candidates_per_group=4,
            alphabet_size=4,
            seed=42,
        )
        df = pool.to_dataframe()
        pool2 = pool.from_dataframe(df)
        assert pool2.pool_size == pool.pool_size
        assert pool2.sequences == pool.sequences
        assert pool2.group_to_candidates == pool.group_to_candidates

    def test_dna_encoder_compatible(self):
        """Verify sequences can be encoded by DNAEncoder."""
        from duet.codebook_evaluator import get_encoder

        pool = create_pool_from_source(
            source="synthetic",
            seq_rounds=7,
            quota=1,
            num_controls=0,
            num_groups=5,
            candidates_per_group=3,
            alphabet_size=4,
            seed=42,
        )
        encoder = get_encoder(alphabet_size=4)
        encoded = encoder.encode(pool.sequences)
        assert encoded.shape == (15, 7)
        assert encoded.min() >= 0
        assert encoded.max() < 4

    def test_reproducible_with_same_seed(self):
        kwargs = dict(
            source="synthetic", seq_rounds=5, quota=1, num_controls=0,
            num_groups=3, candidates_per_group=4, alphabet_size=4, seed=99,
        )
        pool1 = create_pool_from_source(**kwargs)
        pool2 = create_pool_from_source(**kwargs)
        assert pool1.sequences == pool2.sequences

    def test_missing_candidates_per_group_raises(self):
        with pytest.raises(ValueError, match="candidates_per_group"):
            create_pool_from_source(
                source="synthetic",
                seq_rounds=5,
                quota=1,
                num_controls=0,
                num_groups=3,
                alphabet_size=4,
                seed=42,
            )

    def test_missing_num_groups_raises(self):
        with pytest.raises(ValueError, match="num_groups"):
            create_pool_from_source(
                source="synthetic",
                seq_rounds=5,
                quota=1,
                num_controls=0,
                candidates_per_group=4,
                alphabet_size=4,
                seed=42,
            )

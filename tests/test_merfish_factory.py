# tests/test_merfish_factory.py
"""Tests for MERFISHFactory candidate generation."""
import pytest

from duet.merfish_factory import MERFISHFactory


class TestEnumerationOrder:
    """Lock down the combinatorial enumeration order so we can write
    deterministic expectations for subsampling tests."""

    def test_seq_rounds_5_hw_3_enumeration_order(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
        )
        pool = factory.create()
        assert list(pool.sequences) == [
            "11100", "11010", "11001", "10110", "10101", "10011",
            "01110", "01101", "01011", "00111",
        ]

    def test_multiple_weights_concatenated_in_sorted_order(self):
        factory = MERFISHFactory(
            seq_rounds=4, codebook_size=2, hamming_weights=[3, 1],
        )
        pool = factory.create()
        # sorted: HW=1 first (4 codewords), then HW=3 (4 codewords)
        assert list(pool.sequences) == [
            "1000", "0100", "0010", "0001",  # HW=1
            "1110", "1101", "1011", "0111",  # HW=3
        ]


class TestHammingWeightLimitsValidation:
    """Validation of hamming_weight_limits at construction time."""

    def test_limit_key_outside_hamming_weights_raises(self):
        with pytest.raises(ValueError, match="hamming_weight_limits keys"):
            MERFISHFactory(
                seq_rounds=8, codebook_size=4,
                hamming_weights=[3, 4],
                hamming_weight_limits={5: 10},
            )

    def test_zero_cap_raises(self):
        with pytest.raises(ValueError, match="must be a positive int"):
            MERFISHFactory(
                seq_rounds=8, codebook_size=4,
                hamming_weights=[3, 4],
                hamming_weight_limits={4: 0},
            )

    def test_negative_cap_raises(self):
        with pytest.raises(ValueError, match="must be a positive int"):
            MERFISHFactory(
                seq_rounds=8, codebook_size=4,
                hamming_weights=[3, 4],
                hamming_weight_limits={4: -1},
            )

    def test_valid_cap_accepted(self):
        # Should not raise.
        MERFISHFactory(
            seq_rounds=8, codebook_size=4,
            hamming_weights=[3, 4],
            hamming_weight_limits={4: 10},
        )


class TestRequiredCodewordsValidation:
    """Validation of required_codewords at construction time."""

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="not a length-5 binary string"):
            MERFISHFactory(
                seq_rounds=5, codebook_size=2,
                hamming_weights=[3],
                required_codewords=[{"111"}, set()],  # "111" len=3, expected 5
            )

    def test_non_binary_raises(self):
        with pytest.raises(ValueError, match="not a length-5 binary string"):
            MERFISHFactory(
                seq_rounds=5, codebook_size=2,
                hamming_weights=[3],
                required_codewords=[{"11210"}, set()],  # contains "2"
            )

    def test_hw_not_in_hamming_weights_raises(self):
        with pytest.raises(ValueError, match=r"has HW=2, not in hamming_weights"):
            MERFISHFactory(
                seq_rounds=5, codebook_size=2,
                hamming_weights=[3, 4],
                required_codewords=[{"11000"}, set()],  # HW=2
            )

    def test_length_mismatch_raises(self):
        """required_codewords list length must equal codebook_size."""
        with pytest.raises(ValueError, match="length 3 != codebook_size 2"):
            MERFISHFactory(
                seq_rounds=5, codebook_size=2,
                hamming_weights=[3],
                required_codewords=[set(), set(), set()],  # length 3
            )

    def test_valid_required_indexed_by_weight(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=3,
            hamming_weights=[3, 4],
            required_codewords=[{"11100"}, {"11110"}, {"11010"}],
        )
        # _required_by_weight is an implementation detail but is the contract
        # the enumeration code depends on; assert directly here.
        assert factory._required_by_weight[3] == {"11100", "11010"}
        assert factory._required_by_weight[4] == {"11110"}

    def test_valid_required_indexed_by_position(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=3,
            hamming_weights=[3, 4],
            required_codewords=[{"11100"}, {"11110"}, {"11010"}],
        )
        assert factory._required_by_position == [{"11100"}, {"11110"}, {"11010"}]
        assert factory._required_flat == {"11100", "11110", "11010"}

    def test_none_required_codewords_is_empty(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2,
            hamming_weights=[3],
        )
        assert factory._required_by_position == [set(), set()]
        assert factory._required_flat == set()
        assert len(factory._required_by_weight) == 0

    def test_required_exceeds_cap_raises_at_construction(self):
        """Fail-fast: the cap-vs-required check fires in __init__, not in
        .create(), so misconfigured fixtures are caught before any
        enumeration work happens."""
        with pytest.raises(ValueError, match="required codewords exceed cap"):
            MERFISHFactory(
                seq_rounds=5, codebook_size=3,
                hamming_weights=[3],
                hamming_weight_limits={3: 2},
                required_codewords=[{"00111"}, {"01011"}, {"01101"}],  # 3 distinct > cap=2
            )


class TestSubsampling:
    """Subsampling behavior in _enumerate_codewords."""

    def test_no_limits_unchanged(self):
        """With no caps, enumeration matches the baseline test in
        TestEnumerationOrder."""
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
        )
        pool = factory.create()
        assert len(pool.sequences) == 10  # C(5,3) = 10

    def test_cap_binding_returns_exactly_cap(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 4},
            subsample_seed=42,
        )
        pool = factory.create()
        assert len(pool.sequences) == 4

    def test_cap_binding_is_deterministic_with_same_seed(self):
        seqs_1 = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 4}, subsample_seed=42,
        ).create().sequences
        seqs_2 = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 4}, subsample_seed=42,
        ).create().sequences
        assert list(seqs_1) == list(seqs_2)

    def test_cap_binding_differs_with_different_seed(self):
        seqs_a = list(MERFISHFactory(
            seq_rounds=6, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 5}, subsample_seed=1,
        ).create().sequences)
        seqs_b = list(MERFISHFactory(
            seq_rounds=6, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 5}, subsample_seed=999,
        ).create().sequences)
        # C(6,3) = 20; with 5/20 sampled, the chance two seeds produce
        # the same subset by coincidence is vanishingly small.
        assert seqs_a != seqs_b

    def test_cap_non_binding_uses_all(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 100},  # cap > C(5,3)=10
            subsample_seed=42,
        )
        pool = factory.create()
        assert len(pool.sequences) == 10

    def test_subsample_preserves_combinatorial_order(self):
        """idx.sort() ensures the subsample is in the same order as the
        underlying combinatorial enumeration."""
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 4},
            subsample_seed=42,
        )
        pool = factory.create()
        full_order = [
            "11100", "11010", "11001", "10110", "10101", "10011",
            "01110", "01101", "01011", "00111",
        ]
        # The subsample should preserve relative order of the underlying
        # enumeration (the order in `full_order` above).
        subsample_positions = [full_order.index(s) for s in pool.sequences]
        assert subsample_positions == sorted(subsample_positions)

    def test_required_codewords_preserved_when_cap_binding(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 4},
            required_codewords=[{"00111"}, {"01011"}],  # last two in enumeration
            subsample_seed=42,
        )
        pool = factory.create()
        seqs = set(pool.sequences)
        assert "00111" in seqs
        assert "01011" in seqs
        assert len(pool.sequences) == 4

    def test_required_codewords_preserved_with_non_binding_cap(self):
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 100},
            required_codewords=[{"00111"}, set()],
            subsample_seed=42,
        )
        pool = factory.create()
        assert "00111" in set(pool.sequences)
        assert len(pool.sequences) == 10  # all C(5,3)

    def test_required_codewords_no_limits(self):
        """Required codewords are also present when no caps are configured."""
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            required_codewords=[{"00111"}, set()],
        )
        pool = factory.create()
        assert "00111" in set(pool.sequences)
        assert len(pool.sequences) == 10

    def test_estimate_candidate_count_accounts_for_caps(self):
        """_estimate_candidate_count must match the post-cap pool size so
        the large-pool warning in create() is meaningful when caps shrink
        the actual enumeration substantially."""
        factory = MERFISHFactory(
            seq_rounds=5, codebook_size=2, hamming_weights=[3],
            hamming_weight_limits={3: 4},
            subsample_seed=42,
        )
        pool = factory.create()
        # C(5,3) = 10; cap = 4. Estimate must reflect cap, not full count.
        assert factory._estimate_candidate_count() == 4
        assert len(pool.sequences) == 4

import numpy as np
import pytest

from duet.merfish_factory import MERFISHFactory


def test_per_codeword_sample_size_creates_one_group_per_codeword():
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=10,
        hamming_weights=[4],
        per_codeword_sample_size=20,
        sample_seed=42,
    )
    pool = f.create()
    # 10 codewords -> 10 groups
    assert len(pool.group_to_candidates) == 10
    # Each group has exactly per_codeword_sample_size candidates
    for cands in pool.group_to_candidates.values():
        assert len(cands) == 20
    # Each group has quota=1
    for q in pool.quotas.values():
        assert q == 1
    # codeword_to_group is ["pos_0", ..., "pos_9"]
    assert pool.codeword_to_group == [f"pos_{i}" for i in range(10)]


def test_per_codeword_sample_size_none_falls_back_to_single_pseudo_group():
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=10,
        hamming_weights=[4],
        per_codeword_sample_size=None,
    )
    pool = f.create()
    # Single pseudo-group named "MERFISH" with quota=codebook_size
    assert list(pool.group_to_candidates.keys()) == ["MERFISH"]
    assert pool.quotas == {"MERFISH": 10}


def test_per_codeword_sampling_is_reproducible():
    f1 = MERFISHFactory(
        seq_rounds=8, codebook_size=10, hamming_weights=[4],
        per_codeword_sample_size=20, sample_seed=42,
    )
    f2 = MERFISHFactory(
        seq_rounds=8, codebook_size=10, hamming_weights=[4],
        per_codeword_sample_size=20, sample_seed=42,
    )
    p1 = f1.create()
    p2 = f2.create()
    for k in p1.group_to_candidates:
        assert p1.group_to_candidates[k] == p2.group_to_candidates[k]


def test_per_codeword_sample_size_anchors_required_at_each_position():
    """When required_codewords[i] is non-empty, pos_i's sample must contain
    every sequence listed there."""
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=3,
        hamming_weights=[4],
        per_codeword_sample_size=20,
        sample_seed=42,
        required_codewords=[{"11110000"}, {"00001111"}, {"11000011"}],
    )
    pool = f.create()
    seq_to_idx = {s: i for i, s in enumerate(pool.sequences)}
    for i, anchor in enumerate(["11110000", "00001111", "11000011"]):
        assert anchor in seq_to_idx, f"pos_{i} anchor {anchor} missing from pool"
        anchor_idx = seq_to_idx[anchor]
        group_name = f"pos_{i}"
        assert anchor_idx in pool.group_to_candidates[group_name], (
            f"pos_{i} anchor {anchor} not in pos_{i}'s sample"
        )


# Captured from pre-refactor code (Step 2.2a). If this changes, any cached
# PEP matrix built from this fixture by earlier runs becomes
# invalid for the K=20 sample_seed=42 fixture; investigate before updating.
# We lock in pool.sequences[:20] (the actual seed-derived codewords). Using
# pos_0's renumbered indices would be a tautology — the old_to_new loop
# always renumbers the first group's draws to [0..K-1] in encounter order
# regardless of RNG state.
EXPECTED_FIRST_20_SEQS_NO_ANCHOR_SEED_42 = [
    '01101010', '00100111', '11001001', '00110011', '00010111',
    '00101011', '11010001', '11100001', '10000111', '00011101',
    '01000111', '00111010', '10001110', '01010110', '11011000',
    '01011010', '01001110', '10100101', '00111001', '10001101',
]


def test_no_anchor_sample_is_unchanged_by_refactor():
    """Bit-for-bit equivalence between the pre-refactor rng.choice(int) call
    and the post-refactor rng.choice(np.arange(int)) call. NumPy documents
    these as equivalent, but the assertion locks it in so cached PEP
    matrices stay valid."""
    f = MERFISHFactory(
        seq_rounds=8, codebook_size=10, hamming_weights=[4],
        per_codeword_sample_size=20, sample_seed=42,
    )
    pool = f.create()
    assert list(pool.sequences[:20]) == EXPECTED_FIRST_20_SEQS_NO_ANCHOR_SEED_42


def test_per_codeword_sample_size_anchor_count_exceeds_K_raises():
    """If a position requires more sequences than K, that's unsatisfiable
    and must fail loudly at construction (not at sample-build time)."""
    with pytest.raises(ValueError, match="anchor count.*exceeds per_codeword_sample_size"):
        MERFISHFactory(
            seq_rounds=8,
            codebook_size=2,
            hamming_weights=[4],
            per_codeword_sample_size=2,
            sample_seed=42,
            required_codewords=[{"11110000", "00001111", "11000011"}, set()],
        )


def test_per_codeword_sample_size_multi_anchor_per_position():
    """A position may require multiple sequences (e.g., warm-start says X
    but a baseline disagrees and says Y); both must end up in pos_i's
    sample."""
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=2,
        hamming_weights=[4],
        per_codeword_sample_size=10,
        sample_seed=42,
        required_codewords=[{"11110000", "00001111"}, {"11000011"}],
    )
    pool = f.create()
    seq_to_idx = {s: i for i, s in enumerate(pool.sequences)}
    pos_0_indices = set(pool.group_to_candidates["pos_0"])
    assert seq_to_idx["11110000"] in pos_0_indices
    assert seq_to_idx["00001111"] in pos_0_indices


def test_per_codeword_sample_size_pool_coverage_with_anchors():
    """Structural property: every flat-required sequence appears in the
    pruned pool when per-codeword sampling is active."""
    required = [{"11110000"}, {"00001111"}, {"11000011"}]
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=3,
        hamming_weights=[4],
        per_codeword_sample_size=5,
        sample_seed=42,
        required_codewords=required,
    )
    pool = f.create()
    pool_seqs = set(pool.sequences)
    flat = set().union(*required)
    assert flat.issubset(pool_seqs)


def test_per_codeword_sample_size_sample_size_K_preserved_with_anchors():
    """Each pos_i's sample is exactly K candidates, regardless of how many
    anchors it carries (anchors count toward K, not beyond it)."""
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=2,
        hamming_weights=[4],
        per_codeword_sample_size=5,
        sample_seed=42,
        required_codewords=[{"11110000", "00001111"}, set()],  # 2 anchors at pos_0
    )
    pool = f.create()
    assert len(pool.group_to_candidates["pos_0"]) == 5
    assert len(pool.group_to_candidates["pos_1"]) == 5


def test_warm_start_compatible_with_per_codeword_sampling():
    """End-to-end: factory + CodebookWarmStart on a per-codeword-sampled pool.

    Confirms that CodebookWarmStart.get_initial_indices succeeds, and each
    returned index lies in its position's candidate sample (a property the
    DUET optimizer relies on)."""
    import tempfile
    import pandas as pd
    from pathlib import Path
    from duet.initialization import CodebookWarmStart

    warm_seqs = ["11110000", "00001111", "11001100"]  # 3 codewords, HW=4
    f = MERFISHFactory(
        seq_rounds=8,
        codebook_size=3,
        hamming_weights=[4],
        per_codeword_sample_size=10,
        sample_seed=42,
        required_codewords=[{s} for s in warm_seqs],
    )
    pool = f.create()

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "warm.csv"
        pd.DataFrame({"Sequence": warm_seqs}).to_csv(csv_path, index=False)
        ws = CodebookWarmStart(codebook_path=csv_path, sequence_col="Sequence")
        init, meta = ws.get_initial_indices(pool, seed=0)

    # Each init[i] must be in pos_i's candidate sample (DUET optimizer
    # invariant).
    for i, idx in enumerate(init):
        group_name = f"pos_{i}"
        assert int(idx) in pool.group_to_candidates[group_name], (
            f"warm-start index {idx} for pos_{i} not in pos_{i}'s sample"
        )
        # And the sequence at that index is the warm-start sequence.
        assert pool.sequences[int(idx)] == warm_seqs[i]

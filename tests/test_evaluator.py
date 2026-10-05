"""
Tests for CodebookEvaluator and PEP matrix construction.

This module tests the core correctness of PEP matrix computation, including:
1. Noise channels - sample generation and distribution verification
2. Decoding metrics - cost computation correctness
3. Decoding rules - competitor identification
4. CodebookEvaluator - PEP matrix and accuracy computation
5. Encoders - sequence encoding/decoding
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from typing import List

from duet.codebook_evaluator import (
    # Noise channels
    SymmetricEpsilon,
    PositionVaryingEpsilon,
    AsymmetricChannel,
    PositionVaryingAsymmetricChannel,
    # Decoding metrics
    HammingDistance,
    WeightedHammingDistance,
    SymmetricNLL,
    PositionVaryingNLL,
    AsymmetricNLL,
    PositionVaryingAsymmetricNLL,
    # Decoding rules
    UniqueMinimum,
    MarginDecoding,
    PairwisePosteriorThreshold,
    ApproximatePosteriorThreshold,
    # Evaluator
    CodebookEvaluator,
    # Encoders
    DNAEncoder,
    HexEncoder,
    BinaryEncoder,
    get_encoder,
    register_encoder,
    # Helpers
    build_channel_matrix_from_symbol_epsilons,
    build_positional_channel_matrices_from_epsilons,
    # Convenience
    create_evaluator_for_dna,
)


# =============================================================================
# Tests for Encoders
# =============================================================================


class TestDNAEncoder:
    """Tests for DNA sequence encoding."""

    def test_encode_basic(self, dna_encoder):
        """Basic DNA encoding works correctly."""
        sequences = ["ATCG", "AAAA", "TTTT"]
        encoded = dna_encoder.encode(sequences)

        expected = np.array([
            [0, 1, 2, 3],  # ATCG
            [0, 0, 0, 0],  # AAAA
            [1, 1, 1, 1],  # TTTT
        ], dtype=np.int8)

        np.testing.assert_array_equal(encoded, expected)

    def test_encode_decode_roundtrip(self, dna_encoder):
        """Encode then decode returns original sequences."""
        sequences = ["ATCG", "GCTA", "AAGG", "TTCC"]
        encoded = dna_encoder.encode(sequences)
        decoded = dna_encoder.decode(encoded)

        assert decoded == sequences

    def test_encode_invalid_character(self, dna_encoder):
        """Invalid characters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid base"):
            dna_encoder.encode(["ATCX"])

    def test_encode_different_lengths(self, dna_encoder):
        """Different length sequences raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            dna_encoder.encode(["ATCG", "AT"])

    def test_encode_empty_list(self, dna_encoder):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            dna_encoder.encode([])

    def test_alphabet_properties(self, dna_encoder):
        """Encoder has correct alphabet properties."""
        assert dna_encoder.alphabet_size == 4
        assert dna_encoder.alphabet == "ATCG"

    def test_validate_valid_sequence(self, dna_encoder):
        """Valid sequence passes validation."""
        assert dna_encoder.validate("ATCGATCG")

    def test_validate_invalid_sequence(self, dna_encoder):
        """Invalid sequence fails validation."""
        assert not dna_encoder.validate("ATCX")


class TestHexEncoder:
    """Tests for hexadecimal encoding (16-symbol alphabet)."""

    def test_encode_basic(self, hex_encoder):
        """Basic hex encoding works correctly."""
        sequences = ["0F", "A5", "FF"]
        encoded = hex_encoder.encode(sequences)

        expected = np.array([
            [0, 15],   # 0F
            [10, 5],   # A5
            [15, 15],  # FF
        ], dtype=np.int8)

        np.testing.assert_array_equal(encoded, expected)

    def test_encode_decode_roundtrip(self, hex_encoder):
        """Encode then decode returns original (uppercase)."""
        sequences = ["0123", "ABCD", "9F8E"]
        encoded = hex_encoder.encode(sequences)
        decoded = hex_encoder.decode(encoded)

        assert decoded == sequences

    def test_case_insensitive(self, hex_encoder):
        """Lowercase and uppercase encode the same."""
        upper = hex_encoder.encode(["ABCD"])
        lower = hex_encoder.encode(["abcd"])

        np.testing.assert_array_equal(upper, lower)

    def test_alphabet_properties(self, hex_encoder):
        """Encoder has correct alphabet properties."""
        assert hex_encoder.alphabet_size == 16
        assert hex_encoder.alphabet == "0123456789ABCDEF"


class TestBinaryEncoder:
    """Tests for binary encoding."""

    def test_encode_basic(self):
        """Basic binary encoding works correctly."""
        encoder = BinaryEncoder()
        sequences = ["0110", "1001", "0000"]
        encoded = encoder.encode(sequences)

        expected = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.int8)

        np.testing.assert_array_equal(encoded, expected)

    def test_encode_decode_roundtrip(self):
        """Encode then decode returns original."""
        encoder = BinaryEncoder()
        sequences = ["0110", "1001"]
        encoded = encoder.encode(sequences)
        decoded = encoder.decode(encoded)

        assert decoded == sequences


class TestEncoderRegistry:
    """Tests for encoder registry functions."""

    def test_get_encoder_dna(self):
        """get_encoder(4) returns DNAEncoder."""
        encoder = get_encoder(4)
        assert isinstance(encoder, DNAEncoder)

    def test_get_encoder_hex(self):
        """get_encoder(16) returns HexEncoder."""
        encoder = get_encoder(16)
        assert isinstance(encoder, HexEncoder)

    def test_get_encoder_binary(self):
        """get_encoder(2) returns BinaryEncoder."""
        encoder = get_encoder(2)
        assert isinstance(encoder, BinaryEncoder)

    def test_get_encoder_invalid(self):
        """Invalid alphabet size raises ValueError."""
        with pytest.raises(ValueError, match="No encoder registered"):
            get_encoder(5)

    def test_get_encoder_default(self):
        """Default get_encoder() returns DNAEncoder."""
        encoder = get_encoder()
        assert isinstance(encoder, DNAEncoder)


# =============================================================================
# Tests for Noise Channels
# =============================================================================


class TestSymmetricEpsilon:
    """Tests for symmetric error noise channel."""

    def test_generate_shape(self, symmetric_noise_medium):
        """Generated samples have correct shape."""
        rng = np.random.default_rng(42)
        sequence = np.array([0, 1, 2, 3], dtype=np.int8)

        samples = symmetric_noise_medium.generate(sequence, n_samples=100, rng=rng)

        assert samples.shape == (100, 4)

    def test_average_error_rate(self):
        """Average mutation rate matches epsilon."""
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        rng = np.random.default_rng(42)
        sequence = np.array([0, 1, 2, 3] * 10, dtype=np.int8)  # Length 40

        samples = noise.generate(sequence, n_samples=50000, rng=rng)

        # Count mutations
        mutation_rate = (samples != sequence).mean()

        # Should be close to epsilon
        assert abs(mutation_rate - 0.1) < 0.01, \
            f"Mutation rate {mutation_rate} too far from 0.1"

    def test_no_self_mutation(self):
        """Mutations always change the symbol."""
        noise = SymmetricEpsilon(epsilon=0.5, alphabet_size=4)
        rng = np.random.default_rng(42)
        sequence = np.array([0, 0, 0, 0], dtype=np.int8)

        samples = noise.generate(sequence, n_samples=10000, rng=rng)

        # Find positions where mutation occurred
        mutated = samples != sequence

        # At mutated positions, value should not equal original
        for i in range(len(sequence)):
            mutated_values = samples[mutated[:, i], i]
            assert not np.any(mutated_values == sequence[i]), \
                "Self-mutation detected"

    def test_uniform_distribution_among_others(self):
        """Mutations are uniform over other q-1 symbols."""
        noise = SymmetricEpsilon(epsilon=0.9, alphabet_size=4)
        rng = np.random.default_rng(42)
        # Start with all 0s
        sequence = np.array([0, 0, 0, 0], dtype=np.int8)

        samples = noise.generate(sequence, n_samples=100000, rng=rng)

        # Look at mutated positions
        mutated_mask = samples[:, 0] != 0
        mutated_values = samples[mutated_mask, 0]

        # Count occurrences of 1, 2, 3
        counts = np.bincount(mutated_values, minlength=4)

        # Should be roughly equal (each gets 1/3 of mutations)
        total_mutations = counts[1] + counts[2] + counts[3]
        expected_each = total_mutations / 3

        for val in [1, 2, 3]:
            ratio = counts[val] / expected_each
            assert 0.95 < ratio < 1.05, \
                f"Value {val} has count {counts[val]}, expected ~{expected_each}"

    def test_epsilon_zero_no_mutations(self):
        """With epsilon=0, no mutations occur."""
        noise = SymmetricEpsilon(epsilon=0.0, alphabet_size=4)
        rng = np.random.default_rng(42)
        sequence = np.array([0, 1, 2, 3], dtype=np.int8)

        samples = noise.generate(sequence, n_samples=1000, rng=rng)

        # All samples should equal original
        np.testing.assert_array_equal(samples, np.tile(sequence, (1000, 1)))

    def test_epsilon_one_all_mutations(self):
        """With epsilon=1, all positions mutate."""
        noise = SymmetricEpsilon(epsilon=1.0, alphabet_size=4)
        rng = np.random.default_rng(42)
        sequence = np.array([0, 1, 2, 3], dtype=np.int8)

        samples = noise.generate(sequence, n_samples=1000, rng=rng)

        # All positions should be different from original
        assert (samples != sequence).all()

    def test_seed_reproducibility(self):
        """Same seed produces same samples."""
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        sequence = np.array([0, 1, 2, 3], dtype=np.int8)

        rng1 = np.random.default_rng(12345)
        samples1 = noise.generate(sequence, n_samples=100, rng=rng1)

        rng2 = np.random.default_rng(12345)
        samples2 = noise.generate(sequence, n_samples=100, rng=rng2)

        np.testing.assert_array_equal(samples1, samples2)

    def test_invalid_epsilon(self):
        """Invalid epsilon raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be in"):
            SymmetricEpsilon(epsilon=-0.1, alphabet_size=4)

        with pytest.raises(ValueError, match="epsilon must be in"):
            SymmetricEpsilon(epsilon=1.5, alphabet_size=4)

    def test_invalid_alphabet_size(self):
        """Invalid alphabet size raises ValueError."""
        with pytest.raises(ValueError, match="alphabet_size must be at least"):
            SymmetricEpsilon(epsilon=0.1, alphabet_size=1)


class TestPositionVaryingEpsilon:
    """Tests for position-varying noise channel."""

    def test_per_position_rates(self):
        """Each position has its specified error rate."""
        # Position 0: 0%, Position 1: 50%, Position 2: 100%
        epsilons = np.array([0.0, 0.5, 1.0])
        noise = PositionVaryingEpsilon(epsilons=epsilons, alphabet_size=4)
        rng = np.random.default_rng(42)

        sequence = np.array([0, 0, 0], dtype=np.int8)
        samples = noise.generate(sequence, n_samples=10000, rng=rng)

        # Position 0: no mutations
        assert (samples[:, 0] == 0).all()

        # Position 1: ~50% mutations
        rate_1 = (samples[:, 1] != 0).mean()
        assert 0.48 < rate_1 < 0.52, f"Position 1 rate: {rate_1}"

        # Position 2: all mutations
        assert (samples[:, 2] != 0).all()

    def test_length_mismatch_raises(self):
        """Sequence length != epsilons length raises ValueError."""
        epsilons = np.array([0.1, 0.2, 0.3])  # Length 3
        noise = PositionVaryingEpsilon(epsilons=epsilons, alphabet_size=4)
        rng = np.random.default_rng(42)

        sequence = np.array([0, 1, 2, 3], dtype=np.int8)  # Length 4

        with pytest.raises(ValueError, match="must match sequence length"):
            noise.generate(sequence, n_samples=100, rng=rng)


class TestAsymmetricChannel:
    """Tests for asymmetric noise channel."""

    def test_follows_channel_distribution(self):
        """Output distribution matches channel matrix."""
        # Simple channel: symbol 0 stays, symbol 1 always becomes 0
        channel_matrix = np.array([
            [1.0, 0.0],  # 0 -> always 0
            [1.0, 0.0],  # 1 -> always 0
        ])
        noise = AsymmetricChannel(channel_matrix)
        rng = np.random.default_rng(42)

        sequence = np.array([0, 1], dtype=np.int8)
        samples = noise.generate(sequence, n_samples=100, rng=rng)

        # All outputs should be 0
        assert (samples == 0).all()

    def test_probabilistic_channel(self):
        """Probabilistic channel produces correct distribution."""
        # 70% stay same, 30% flip
        channel_matrix = np.array([
            [0.7, 0.3],
            [0.3, 0.7],
        ])
        noise = AsymmetricChannel(channel_matrix)
        rng = np.random.default_rng(42)

        sequence = np.array([0, 0, 0, 0], dtype=np.int8)
        samples = noise.generate(sequence, n_samples=50000, rng=rng)

        # Count 0s in output
        stay_rate = (samples == 0).mean()
        assert 0.68 < stay_rate < 0.72, f"Stay rate: {stay_rate}"

    def test_from_symbol_epsilons(self):
        """Factory method creates correct channel matrix."""
        epsilons = np.array([0.1, 0.2, 0.15, 0.1])
        noise = AsymmetricChannel.from_symbol_epsilons(epsilons)

        # Check diagonal
        for i, eps in enumerate(epsilons):
            assert np.isclose(noise.channel_matrix[i, i], 1 - eps)

        # Check off-diagonal sums
        for i, eps in enumerate(epsilons):
            off_diag_sum = noise.channel_matrix[i].sum() - noise.channel_matrix[i, i]
            assert np.isclose(off_diag_sum, eps)

    def test_invalid_channel_matrix(self):
        """Non-stochastic matrix raises ValueError."""
        # Rows don't sum to 1
        bad_matrix = np.array([
            [0.5, 0.3],
            [0.4, 0.4],
        ])
        with pytest.raises(ValueError, match="must sum to 1"):
            AsymmetricChannel(bad_matrix)


class TestPositionVaryingAsymmetricChannel:
    """Tests for position-varying, asymmetric noise channel."""

    def test_per_position_channels(self):
        """Each position uses its own channel matrix."""
        # Position 0: identity (no noise)
        # Position 1: flip everything
        channel_matrices = np.array([
            [[1.0, 0.0], [0.0, 1.0]],  # Position 0: identity
            [[0.0, 1.0], [1.0, 0.0]],  # Position 1: flip
        ])
        noise = PositionVaryingAsymmetricChannel(channel_matrices)
        rng = np.random.default_rng(42)

        sequence = np.array([0, 0], dtype=np.int8)
        samples = noise.generate(sequence, n_samples=100, rng=rng)

        # Position 0 should stay 0
        assert (samples[:, 0] == 0).all()

        # Position 1 should all be 1 (flipped)
        assert (samples[:, 1] == 1).all()


# =============================================================================
# Tests for Decoding Metrics
# =============================================================================


class TestHammingDistance:
    """Tests for Hamming distance computation."""

    def test_basic_cases(self, hamming_distance):
        """Basic Hamming distance computation."""
        observed = np.array([[0, 1, 2, 3]])
        transmitted = np.array([
            [0, 1, 2, 3],  # Distance 0
            [0, 0, 0, 0],  # Distance 3
            [3, 2, 1, 0],  # Distance 4
        ])

        distances = hamming_distance.compute(observed, transmitted)

        expected = np.array([[0, 3, 4]])
        np.testing.assert_array_equal(distances, expected)

    def test_self_distance_zero(self, hamming_distance):
        """Distance to self is always 0."""
        sequences = np.array([
            [0, 1, 2, 3],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ])

        for seq in sequences:
            d = hamming_distance.compute(seq.reshape(1, -1), seq.reshape(1, -1))
            assert d[0, 0] == 0

    def test_symmetric(self, hamming_distance):
        """Hamming distance is symmetric."""
        seq1 = np.array([[0, 1, 2, 3]])
        seq2 = np.array([[0, 0, 0, 0]])

        d12 = hamming_distance.compute(seq1, seq2)[0, 0]
        d21 = hamming_distance.compute(seq2, seq1)[0, 0]

        assert d12 == d21

    def test_compute_pairwise(self, hamming_distance):
        """Pairwise computation matches diagonal of full matrix."""
        observed = np.array([
            [0, 1, 2, 3],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ])
        transmitted = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ])

        pairwise = hamming_distance.compute_pairwise(observed, transmitted)

        expected = np.array([3, 4, 4])
        np.testing.assert_array_equal(pairwise, expected)


class TestWeightedHammingDistance:
    """Tests for weighted Hamming distance."""

    def test_weights_applied(self):
        """Weights are applied correctly."""
        weights = np.array([1.0, 2.0, 0.0, 1.0])
        decoding_metric = WeightedHammingDistance(weights)

        observed = np.array([[0, 0, 0, 0]])
        transmitted = np.array([[1, 1, 1, 1]])  # All different

        d = decoding_metric.compute(observed, transmitted)

        # Position 0: 1*1=1, Position 1: 1*2=2, Position 2: 1*0=0, Position 3: 1*1=1
        # Total: 4
        assert d[0, 0] == 4.0

    def test_zero_weight_ignored(self):
        """Positions with zero weight don't contribute."""
        weights = np.array([0.0, 0.0, 0.0, 0.0])
        decoding_metric = WeightedHammingDistance(weights)

        observed = np.array([[0, 0, 0, 0]])
        transmitted = np.array([[1, 1, 1, 1]])

        d = decoding_metric.compute(observed, transmitted)

        assert d[0, 0] == 0.0

    def test_invalid_negative_weights(self):
        """Negative weights raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            WeightedHammingDistance(np.array([1.0, -1.0]))


class TestSymmetricNLL:
    """Tests for symmetric NLL decoding metric."""

    def test_weight_formula(self):
        """Weight is correctly computed from epsilon."""
        epsilon = 0.1
        alphabet_size = 4
        decoding_metric = SymmetricNLL(epsilon, alphabet_size)

        expected_weight = np.log((alphabet_size - 1) * (1 - epsilon) / epsilon)
        assert np.isclose(decoding_metric.weight, expected_weight)

    def test_proportional_to_hamming(self):
        """NLL is proportional to Hamming distance."""
        decoding_metric = SymmetricNLL(epsilon=0.1, alphabet_size=4)
        hamming = HammingDistance()

        observed = np.array([[0, 1, 2, 3]])
        transmitted = np.array([
            [0, 1, 2, 3],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ])

        nll = decoding_metric.compute(observed, transmitted)
        ham = hamming.compute(observed, transmitted)

        # NLL = weight * Hamming
        expected = decoding_metric.weight * ham
        np.testing.assert_array_almost_equal(nll, expected)

    def test_zero_epsilon_handling(self):
        """Zero epsilon (infinite weight) handled correctly."""
        decoding_metric = SymmetricNLL(epsilon=0.0, alphabet_size=4)

        observed = np.array([[0, 1, 2, 3]])

        # Same sequence -> distance 0
        transmitted_same = np.array([[0, 1, 2, 3]])
        d_same = decoding_metric.compute(observed, transmitted_same)
        assert d_same[0, 0] == 0.0  # 0 * inf = 0

        # Different sequence -> distance inf
        transmitted_diff = np.array([[1, 1, 1, 1]])
        d_diff = decoding_metric.compute(observed, transmitted_diff)
        assert np.isinf(d_diff[0, 0])


class TestPositionVaryingNLL:
    """Tests for position-varying NLL decoding metric."""

    def test_per_position_weights(self):
        """Weights are computed per-position."""
        epsilons = np.array([0.1, 0.2, 0.3, 0.4])
        decoding_metric = PositionVaryingNLL(epsilons, alphabet_size=4)

        expected_weights = np.log(3 * (1 - epsilons) / epsilons)
        np.testing.assert_array_almost_equal(decoding_metric.weights, expected_weights)

    def test_weighted_sum(self):
        """Decoding metric is weighted sum of mismatches."""
        epsilons = np.array([0.1, 0.2, 0.1, 0.2])
        decoding_metric = PositionVaryingNLL(epsilons, alphabet_size=4)

        # Mismatch only at position 1
        observed = np.array([[0, 0, 0, 0]])
        transmitted = np.array([[0, 1, 0, 0]])

        d = decoding_metric.compute(observed, transmitted)

        # Should equal weight at position 1
        assert np.isclose(d[0, 0], decoding_metric.weights[1])


class TestAsymmetricNLL:
    """Tests for asymmetric NLL decoding metric."""

    def test_negative_log_probability(self):
        """Decoding metric is -log(P(obs|tx))."""
        # Deterministic channel: 0->0, 1->1
        channel_matrix = np.eye(2)
        decoding_metric = AsymmetricNLL(channel_matrix)

        # Transmitted [0], observed [0] -> P=1, -log(1)=0
        observed = np.array([[0]])
        transmitted = np.array([[0]])
        d = decoding_metric.compute(observed, transmitted)
        assert np.isclose(d[0, 0], 0.0)

        # Transmitted [0], observed [1] -> P=0, -log(0)=inf
        observed = np.array([[1]])
        d = decoding_metric.compute(observed, transmitted)
        assert np.isinf(d[0, 0])

    def test_from_symbol_epsilons(self):
        """Factory method creates correct decoding metric."""
        epsilons = np.array([0.1, 0.2, 0.15, 0.1])
        decoding_metric = AsymmetricNLL.from_symbol_epsilons(epsilons)

        # Check dimensions
        assert decoding_metric.channel_matrix.shape == (4, 4)
        assert decoding_metric.cost_matrix.shape == (4, 4)


# =============================================================================
# Tests for Decoding Rules
# =============================================================================


class TestUniqueMinimum:
    """Tests for unique minimum decoding rule."""

    def test_identifies_competitors(self, unique_minimum_rule):
        """Competitors are codewords with cost <= transmitted cost."""
        # 3 samples, 4 codewords
        cost_matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],  # Sample 0: costs to codewords
            [2.0, 1.0, 1.0, 3.0],  # Sample 1: tie between codewords 1,2
            [0.5, 0.5, 0.5, 0.5],  # Sample 2: all tied
        ])
        transmitted_costs = np.array([1.0, 1.0, 0.5])  # Costs to transmitted codeword

        competitors = unique_minimum_rule.identify_competitors(cost_matrix, transmitted_costs)

        # Sample 0: only codeword 0 (cost 1.0 <= 1.0)
        assert competitors[0, 0] == True
        assert competitors[0, 1] == False

        # Sample 1: codewords 1 and 2 (both cost 1.0 <= 1.0)
        assert competitors[1, 1] == True
        assert competitors[1, 2] == True
        assert competitors[1, 0] == False

        # Sample 2: all codewords (all costs 0.5 <= 0.5)
        assert competitors[2, :].toarray().all()

    def test_returns_sparse_matrix(self, unique_minimum_rule):
        """Returns csr_matrix."""
        cost_matrix = np.array([[1.0, 2.0]])
        transmitted_costs = np.array([1.0])

        competitors = unique_minimum_rule.identify_competitors(cost_matrix, transmitted_costs)

        assert isinstance(competitors, csr_matrix)


class TestMarginDecoding:
    """Tests for margin-based decoding."""

    def test_margin_expands_competitors(self):
        """Larger margin includes more competitors."""
        rule_k0 = MarginDecoding(k=0.0)
        rule_k1 = MarginDecoding(k=1.0)

        cost_matrix = np.array([[1.0, 1.5, 2.0, 3.0]])
        transmitted_costs = np.array([1.0])

        competitors_k0 = rule_k0.identify_competitors(cost_matrix, transmitted_costs)
        competitors_k1 = rule_k1.identify_competitors(cost_matrix, transmitted_costs)

        # k=0: cost <= 1.0, so only codeword 0
        assert competitors_k0[0, 0] == True
        assert competitors_k0[0, 1] == False

        # k=1: cost <= 2.0, so codewords 0, 1, 2
        assert competitors_k1[0, 0] == True
        assert competitors_k1[0, 1] == True
        assert competitors_k1[0, 2] == True
        assert competitors_k1[0, 3] == False

    def test_k0_equals_unique_minimum(self, unique_minimum_rule):
        """MarginDecoding(k=0) equals UniqueMinimum."""
        rule_margin = MarginDecoding(k=0.0)

        cost_matrix = np.array([
            [1.0, 2.0, 1.5],
            [2.0, 1.0, 3.0],
        ])
        transmitted_costs = np.array([1.0, 1.0])

        competitors_margin = rule_margin.identify_competitors(cost_matrix, transmitted_costs)
        competitors_unique = unique_minimum_rule.identify_competitors(cost_matrix, transmitted_costs)

        np.testing.assert_array_equal(
            competitors_margin.toarray(),
            competitors_unique.toarray()
        )


class TestPairwisePosteriorThreshold:
    """Tests for pairwise posterior threshold decoding."""

    def test_threshold_05_equals_unique_minimum(self, unique_minimum_rule):
        """Threshold 0.5 equals unique minimum decoding."""
        rule_posterior = PairwisePosteriorThreshold(threshold=0.5)

        cost_matrix = np.array([
            [1.0, 2.0, 1.5],
            [2.0, 1.0, 3.0],
        ])
        transmitted_costs = np.array([1.0, 1.0])

        competitors_posterior = rule_posterior.identify_competitors(cost_matrix, transmitted_costs)
        competitors_unique = unique_minimum_rule.identify_competitors(cost_matrix, transmitted_costs)

        np.testing.assert_array_equal(
            competitors_posterior.toarray(),
            competitors_unique.toarray()
        )

    def test_higher_threshold_fewer_competitors(self):
        """Higher threshold is more conservative (more competitors)."""
        rule_low = PairwisePosteriorThreshold(threshold=0.5)
        rule_high = PairwisePosteriorThreshold(threshold=0.9)

        cost_matrix = np.array([[1.0, 1.5, 2.0, 5.0]])
        transmitted_costs = np.array([1.0])

        comp_low = rule_low.identify_competitors(cost_matrix, transmitted_costs)
        comp_high = rule_high.identify_competitors(cost_matrix, transmitted_costs)

        # Higher threshold should have >= competitors
        assert comp_high.toarray().sum() >= comp_low.toarray().sum()

    def test_invalid_threshold(self):
        """Invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be in"):
            PairwisePosteriorThreshold(threshold=0.0)

        with pytest.raises(ValueError, match="threshold must be in"):
            PairwisePosteriorThreshold(threshold=1.0)


# =============================================================================
# Tests for Helper Functions
# =============================================================================


class TestBuildChannelMatrix:
    """Tests for channel matrix construction helpers."""

    def test_from_symbol_epsilons_shape(self):
        """Output has correct shape."""
        epsilons = np.array([0.1, 0.2, 0.15, 0.1])
        matrix = build_channel_matrix_from_symbol_epsilons(epsilons)

        assert matrix.shape == (4, 4)

    def test_from_symbol_epsilons_row_stochastic(self):
        """Rows sum to 1."""
        epsilons = np.array([0.1, 0.2, 0.15, 0.1])
        matrix = build_channel_matrix_from_symbol_epsilons(epsilons)

        row_sums = matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))

    def test_from_symbol_epsilons_diagonal(self):
        """Diagonal is 1 - epsilon."""
        epsilons = np.array([0.1, 0.2, 0.15, 0.1])
        matrix = build_channel_matrix_from_symbol_epsilons(epsilons)

        expected_diag = 1 - epsilons
        np.testing.assert_array_almost_equal(np.diag(matrix), expected_diag)

    def test_positional_matrices_shape(self):
        """Positional matrices have correct shape."""
        epsilons = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
        ])
        matrices = build_positional_channel_matrices_from_epsilons(epsilons)

        assert matrices.shape == (2, 4, 4)


# =============================================================================
# Tests for CodebookEvaluator
# =============================================================================


class TestCodebookEvaluatorConstruction:
    """Tests for CodebookEvaluator construction."""

    def test_from_dna_list(self, simple_evaluator_setup):
        """Can create evaluator from DNA list."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        assert evaluator.num_codewords == 4
        assert evaluator.seq_length == 4

    def test_from_sequence_list(self, simple_evaluator_setup):
        """Can create evaluator from generic sequence list."""
        encoder = DNAEncoder()
        evaluator = CodebookEvaluator.from_sequence_list(
            sequence_list=simple_evaluator_setup["library"],
            encoder=encoder,
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        assert evaluator.num_codewords == 4

    def test_from_encoded_array(self, simple_evaluator_setup):
        """Can create evaluator from pre-encoded array."""
        encoder = DNAEncoder()
        codebook = encoder.encode(simple_evaluator_setup["library"])

        evaluator = CodebookEvaluator(
            codebook=codebook,
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        assert evaluator.num_codewords == 4

    def test_invalid_codebook_dtype(self, simple_evaluator_setup):
        """Non-integer codebook raises TypeError."""
        codebook = np.array([[0.1, 0.2], [0.3, 0.4]])  # Float

        with pytest.raises(TypeError, match="integer dtype"):
            CodebookEvaluator(
                codebook=codebook,
                noise_channel=simple_evaluator_setup["noise_channel"],
                decoding_metric=simple_evaluator_setup["decoding_metric"],
                decoding_rule=simple_evaluator_setup["decoding_rule"],
                n_samples=100,
            )


class TestCodebookEvaluatorPEPMatrix:
    """Tests for PEP matrix computation."""

    def test_pep_matrix_shape(self, simple_evaluator_setup):
        """PEP matrix has correct shape."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.shape == (4, 4)

    def test_pep_matrix_diagonal_is_n_samples(self, simple_evaluator_setup):
        """Diagonal of PEP count matrix equals n_samples (self always wins)."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        np.testing.assert_array_almost_equal(np.diag(pep), np.full(4, _n_samples))

    def test_pep_matrix_range(self, simple_evaluator_setup):
        """All PEP count values are in [0, n_samples]."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.dtype == np.uint16
        assert np.all(pep >= 0)
        assert np.all(pep <= _n_samples)

    def test_pep_matrix_no_nan(self, simple_evaluator_setup):
        """PEP matrix dtype is uint16 (integer, no NaN/Inf possible)."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.dtype == np.uint16

    def test_pep_identical_sequences_is_one(self):
        """PEP[i,j] = 1.0 for identical sequences."""
        library = ["ATCG", "ATCG", "GCTA", "GCTG"]  # 0 and 1 identical
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=library,
            noise_channel=noise,
            decoding_metric=decoding_metric,
            decoding_rule=rule,
            n_samples=1000,
            seed=42,
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        # Identical sequences should have count = n_samples
        assert pep[0, 1] == _n_samples
        assert pep[1, 0] == _n_samples

    def test_pep_distant_sequences_low(self):
        """PEP[i,j] is low for very different sequences."""
        library = ["AAAA", "TTTT", "CCCC", "GGGG"]  # All maximally distant
        noise = SymmetricEpsilon(epsilon=0.05, alphabet_size=4)  # Low noise
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=library,
            noise_channel=noise,
            decoding_metric=decoding_metric,
            decoding_rule=rule,
            n_samples=5000,
            seed=42,
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        # Off-diagonal should be very low (< 10% of n_samples)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert pep[i, j] < 0.1 * _n_samples, \
                        f"PEP[{i},{j}] = {pep[i,j]}/{_n_samples} too high for distant sequences"

    def test_seed_reproducibility(self, simple_evaluator_setup):
        """Same seed produces identical PEP matrix."""
        evaluator1 = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=1000,
            seed=12345,
        )

        evaluator2 = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=1000,
            seed=12345,
        )

        pep1, _ = evaluator1.compute_pep_matrix(n_jobs=1)
        pep2, _ = evaluator2.compute_pep_matrix(n_jobs=1)

        np.testing.assert_array_equal(pep1, pep2)

    def test_parallel_equals_serial(self, simple_evaluator_setup):
        """Parallel computation produces same result as serial."""
        evaluator_serial = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=1000,
            seed=42,
        )

        evaluator_parallel = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=1000,
            seed=42,
        )

        pep_serial, _ = evaluator_serial.compute_pep_matrix(n_jobs=1)
        pep_parallel, _ = evaluator_parallel.compute_pep_matrix(n_jobs=2)

        np.testing.assert_array_equal(pep_serial, pep_parallel)


class TestCodebookEvaluatorAccuracy:
    """Tests for accuracy computation."""

    def test_accuracy_range(self, simple_evaluator_setup):
        """Accuracy is in [0, 1]."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        evaluator.initialize_cache(n_jobs=1)
        accuracy = evaluator.get_accuracy()

        assert 0.0 <= accuracy <= 1.0

    def test_perfect_accuracy_no_noise(self):
        """With no noise, accuracy is 1.0 (for distinct sequences)."""
        library = ["AAAA", "TTTT", "CCCC", "GGGG"]
        noise = SymmetricEpsilon(epsilon=0.0, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=library,
            noise_channel=noise,
            decoding_metric=decoding_metric,
            decoding_rule=rule,
            n_samples=100,
            seed=42,
        )

        evaluator.initialize_cache(n_jobs=1)
        accuracy = evaluator.get_accuracy()

        assert np.isclose(accuracy, 1.0)

    def test_identical_sequences_low_accuracy(self):
        """With identical sequences, accuracy is low."""
        library = ["ATCG", "ATCG", "ATCG", "ATCG"]  # All identical
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=library,
            noise_channel=noise,
            decoding_metric=decoding_metric,
            decoding_rule=rule,
            n_samples=1000,
            seed=42,
        )

        evaluator.initialize_cache(n_jobs=1)
        accuracy = evaluator.get_accuracy()

        # With 4 identical sequences, accuracy should be ~1/4 = 0.25
        assert accuracy < 0.3

    def test_codeword_accuracy_length(self, simple_evaluator_setup):
        """Per-codeword accuracy has correct length."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        evaluator.initialize_cache(n_jobs=1)

        # All codewords
        accuracies = evaluator.get_codeword_accuracy()
        assert len(accuracies) == 4

        # Subset
        accuracies_subset = evaluator.get_codeword_accuracy(np.array([0, 2]))
        assert len(accuracies_subset) == 2

    def test_accuracy_subset(self, simple_evaluator_setup):
        """Accuracy can be computed for subset of codewords."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        evaluator.initialize_cache(n_jobs=1)

        indices = np.array([0, 2])
        accuracy = evaluator.get_accuracy(indices)

        assert 0.0 <= accuracy <= 1.0


class TestCodebookEvaluatorCache:
    """Tests for cache initialization and management."""

    def test_is_initialized_before_cache(self, simple_evaluator_setup):
        """is_initialized is False before initialize_cache."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        assert not evaluator.is_initialized

    def test_is_initialized_after_cache(self, simple_evaluator_setup):
        """is_initialized is True after initialize_cache."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        evaluator.initialize_cache(n_jobs=1)

        assert evaluator.is_initialized

    def test_accuracy_requires_cache(self, simple_evaluator_setup):
        """get_accuracy raises error if cache not initialized."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        with pytest.raises(RuntimeError, match="Cache must be initialized"):
            evaluator.get_accuracy()

    def test_pep_from_cache_requires_cache(self, simple_evaluator_setup):
        """get_pairwise_error_from_cache raises error if not initialized."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        with pytest.raises(RuntimeError, match="Cache must be initialized"):
            evaluator.get_pairwise_error_from_cache()


class TestCodebookEvaluatorStats:
    """Tests for evaluator statistics."""

    def test_get_stats_before_cache(self, simple_evaluator_setup):
        """get_stats works before cache initialization."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        stats = evaluator.get_stats()

        assert stats['num_codewords'] == 4
        assert stats['seq_length'] == 4
        assert stats['cache_initialized'] == False

    def test_get_stats_after_cache(self, simple_evaluator_setup):
        """get_stats includes cache info after initialization."""
        evaluator = CodebookEvaluator.from_dna_list(
            dna_list=simple_evaluator_setup["library"],
            noise_channel=simple_evaluator_setup["noise_channel"],
            decoding_metric=simple_evaluator_setup["decoding_metric"],
            decoding_rule=simple_evaluator_setup["decoding_rule"],
            n_samples=simple_evaluator_setup["n_samples"],
            seed=simple_evaluator_setup["seed"],
        )

        evaluator.initialize_cache(n_jobs=1)
        stats = evaluator.get_stats()

        assert stats['cache_initialized'] == True
        assert 'matrix_shape' in stats
        assert 'matrix_nnz' in stats


class TestCreateEvaluatorForDNA:
    """Tests for convenience function."""

    def test_creates_evaluator(self):
        """create_evaluator_for_dna creates working evaluator."""
        library = ["ATCG", "GCTA", "TTAA", "AAGG"]
        evaluator = create_evaluator_for_dna(
            dna_list=library,
            epsilon=0.1,
            n_samples=100,
            seed=42,
        )

        assert evaluator.num_codewords == 4
        assert evaluator.seq_length == 4

        # Can compute PEP matrix (returns tuple now)
        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)
        assert pep.shape == (4, 4)
        assert pep.dtype == np.uint16
        assert _n_samples == 100


# =============================================================================
# Tests for PEP Matrix Convergence
# =============================================================================


class TestPEPConvergence:
    """Tests for Monte Carlo convergence of PEP estimates."""

    def test_variance_decreases_with_samples(self):
        """Variance of PEP estimates decreases with more samples."""
        library = ["ATCG", "ATGG", "GCTA", "GCTG"]
        noise = SymmetricEpsilon(epsilon=0.15, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        # Low samples
        evaluator_low = CodebookEvaluator.from_dna_list(
            library, noise, decoding_metric, rule, n_samples=100, seed=42
        )
        pep_low, _ns_low = evaluator_low.compute_pep_matrix(n_jobs=1)

        # High samples
        evaluator_high = CodebookEvaluator.from_dna_list(
            library, noise, decoding_metric, rule, n_samples=5000, seed=42
        )
        pep_high, _ns_high = evaluator_high.compute_pep_matrix(n_jobs=1)

        # Run multiple times to estimate variance
        pep_runs_low = []
        pep_runs_high = []

        for seed in range(5):
            eval_low = CodebookEvaluator.from_dna_list(
                library, noise, decoding_metric, rule, n_samples=100, seed=seed
            )
            pep_runs_low.append(eval_low.compute_pep_matrix(n_jobs=1)[0])

            eval_high = CodebookEvaluator.from_dna_list(
                library, noise, decoding_metric, rule, n_samples=2000, seed=seed
            )
            pep_runs_high.append(eval_high.compute_pep_matrix(n_jobs=1)[0])

        # Compute variance (normalize counts to probabilities first)
        pep_probs_low = [p.astype(np.float64) / 100 for p in pep_runs_low]
        pep_probs_high = [p.astype(np.float64) / 2000 for p in pep_runs_high]
        var_low = np.var(pep_probs_low, axis=0).mean()
        var_high = np.var(pep_probs_high, axis=0).mean()

        assert var_high < var_low, \
            f"High sample variance ({var_high}) should be less than low ({var_low})"


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_codeword(self):
        """Single codeword evaluator works."""
        library = ["ATCG"]
        evaluator = create_evaluator_for_dna(library, epsilon=0.1, n_samples=100, seed=42)

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.shape == (1, 1)
        assert pep[0, 0] == _n_samples

    def test_length_one_sequences(self):
        """Length-1 sequences work."""
        library = ["A", "T", "C", "G"]
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            library, noise, decoding_metric, rule, n_samples=1000, seed=42
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.shape == (4, 4)
        assert pep.dtype == np.uint16

    def test_long_sequences(self):
        """Long sequences work."""
        np.random.seed(42)
        alphabet = "ATCG"
        library = [
            ''.join(np.random.choice(list(alphabet), size=100))
            for _ in range(4)
        ]

        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            library, noise, decoding_metric, rule, n_samples=500, seed=42
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.shape == (4, 4)
        assert pep.dtype == np.uint16

    def test_high_epsilon(self):
        """High epsilon (near 0.5) works."""
        library = ["ATCG", "GCTA", "TTAA", "AAGG"]
        noise = SymmetricEpsilon(epsilon=0.49, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            library, noise, decoding_metric, rule, n_samples=1000, seed=42
        )

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.dtype == np.uint16
        assert np.all(pep >= 0)
        assert np.all(pep <= _n_samples)

    def test_two_codewords(self):
        """Two codeword evaluator works."""
        library = ["AAAA", "TTTT"]
        evaluator = create_evaluator_for_dna(library, epsilon=0.1, n_samples=1000, seed=42)

        pep, _n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.shape == (2, 2)
        np.testing.assert_array_almost_equal(np.diag(pep), [_n_samples, _n_samples])


# =============================================================================
# Tests for Uint16 Count Matrix Output
# =============================================================================


class TestUint16CountOutput:
    """Tests for compute_pep_matrix returning uint16 counts."""

    def test_compute_pep_matrix_returns_uint16_counts(self):
        """Verify dtype is uint16, shape correct, values in [0, n_samples]."""
        library = ["ATCG", "GCTA", "TTAA", "AAGG"]
        evaluator = create_evaluator_for_dna(library, epsilon=0.1, n_samples=1000, seed=42)

        pep, n_samples = evaluator.compute_pep_matrix(n_jobs=1)

        assert pep.dtype == np.uint16
        assert pep.shape == (4, 4)
        assert n_samples == 1000
        assert np.all(pep >= 0)
        assert np.all(pep <= n_samples)
        # Diagonal should equal n_samples (self always decodes correctly)
        np.testing.assert_array_equal(np.diag(pep), np.full(4, n_samples))

    def test_compute_pep_matrix_overflow_validation(self):
        """Verify NotImplementedError when n_samples > 32767."""
        library = ["ATCG", "GCTA"]
        noise = SymmetricEpsilon(epsilon=0.1, alphabet_size=4)
        decoding_metric = HammingDistance()
        rule = UniqueMinimum()

        evaluator = CodebookEvaluator.from_dna_list(
            library, noise, decoding_metric, rule, n_samples=40000, seed=42
        )

        with pytest.raises(NotImplementedError, match="n_samples=40000 exceeds maximum"):
            evaluator.compute_pep_matrix(n_jobs=1)

"""Tests for the 2-D synthetic benchmark's noise-channel factory.

Verifies the invariants introduced in the 2026-05-10 redesign:
- Per-position, per-symbol eps stays ≤ 0.5 across the configured sweep.
- Mean per-bit error rate (uniform symbol distribution) equals the
  configured `error_rate`.

Also pins down the corner values of each new channel so future regressions
don't silently break the design.
"""
from __future__ import annotations

from _optional_deps import require_benchmark_extra

require_benchmark_extra()

import numpy as np
import pytest

from duet.codebook_evaluator import (
    AsymmetricChannel,
    PositionVaryingAsymmetricChannel,
    PositionVaryingEpsilon,
    SymmetricEpsilon,
)
from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
    _create_noise_and_decoding_metric,
)


ERROR_RATES = [0.05, 0.1, 0.2]
NOISE_TYPES = ["symmetric", "position_varying", "asymmetric", "position_varying_asymmetric"]
SEQ_LENGTH = 8
ALPHABET_SIZES = [2, 3, 4]  # binary (production), odd, even


def _channel_matrices(noise, q: int, L: int) -> np.ndarray:
    """Return channel matrices [L, q, q] for any NoiseChannel variant.

    Channels with uniform structure (SymmetricEpsilon, PositionVaryingEpsilon) are
    expanded into the full position-by-symbol form so a single test can
    inspect eps for all four channel types uniformly.
    """
    if isinstance(noise, SymmetricEpsilon):
        eps = float(noise.epsilon)
        m = np.full((q, q), eps / (q - 1))
        np.fill_diagonal(m, 1 - eps)
        return np.broadcast_to(m, (L, q, q)).copy()
    if isinstance(noise, PositionVaryingEpsilon):
        out = np.zeros((L, q, q))
        for p in range(L):
            eps = float(noise.epsilons[p])
            m = np.full((q, q), eps / (q - 1))
            np.fill_diagonal(m, 1 - eps)
            out[p] = m
        return out
    if isinstance(noise, AsymmetricChannel):
        return np.broadcast_to(noise.channel_matrix, (L, q, q)).copy()
    if isinstance(noise, PositionVaryingAsymmetricChannel):
        return np.asarray(noise.channel_matrices)
    raise TypeError(f"Unsupported noise channel type: {type(noise).__name__}")


@pytest.mark.parametrize("noise_type", NOISE_TYPES)
@pytest.mark.parametrize("error_rate", ERROR_RATES)
@pytest.mark.parametrize("alphabet_size", ALPHABET_SIZES)
def test_eps_at_most_half(noise_type, error_rate, alphabet_size):
    """Per-(position, symbol) eps must stay ≤ 0.5 across the sweep."""
    noise, _ = _create_noise_and_decoding_metric(
        noise_type=noise_type,
        error_rate=error_rate,
        seq_length=SEQ_LENGTH,
        alphabet_size=alphabet_size,
    )
    chs = _channel_matrices(noise, alphabet_size, SEQ_LENGTH)
    eps = 1.0 - np.diagonal(chs, axis1=1, axis2=2)  # (L, q)
    assert eps.max() <= 0.5 + 1e-9, (
        f"eps exceeded 0.5: max={eps.max():.4f} for "
        f"{noise_type=} {error_rate=} {alphabet_size=}"
    )


@pytest.mark.parametrize("noise_type", NOISE_TYPES)
@pytest.mark.parametrize("error_rate", ERROR_RATES)
@pytest.mark.parametrize("alphabet_size", ALPHABET_SIZES)
def test_mean_per_bit_error_rate(noise_type, error_rate, alphabet_size):
    """Mean per-bit error (uniform symbol distribution) equals error_rate.

    Per-bit error at position p = (1/q) Σ_a eps(p, a) = (1/q) Σ_a (1 - P(a→a)).
    Mean over positions of that equals error_rate by construction.
    """
    noise, _ = _create_noise_and_decoding_metric(
        noise_type=noise_type,
        error_rate=error_rate,
        seq_length=SEQ_LENGTH,
        alphabet_size=alphabet_size,
    )
    chs = _channel_matrices(noise, alphabet_size, SEQ_LENGTH)
    eps = 1.0 - np.diagonal(chs, axis1=1, axis2=2)  # (L, q)
    mean_eps = float(eps.mean())
    assert mean_eps == pytest.approx(error_rate, abs=1e-9), (
        f"mean per-bit error={mean_eps:.6f}, expected={error_rate} "
        f"for {noise_type=} {alphabet_size=}"
    )


def test_symmetric_unchanged():
    """Symmetric is unaffected by the redesign: epsilon == error_rate exactly."""
    x = 0.1
    noise, _ = _create_noise_and_decoding_metric(
        noise_type="symmetric", error_rate=x, seq_length=8, alphabet_size=2,
    )
    assert isinstance(noise, SymmetricEpsilon)
    assert noise.epsilon == pytest.approx(x)


def test_position_varying_ramp_endpoints_binary():
    """Position-varying ramp: eps[0] = 0.5x, eps[L-1] = 1.5x."""
    x = 0.2
    L = 8
    noise, _ = _create_noise_and_decoding_metric(
        noise_type="position_varying", error_rate=x, seq_length=L, alphabet_size=2,
    )
    assert isinstance(noise, PositionVaryingEpsilon)
    assert float(noise.epsilons[0]) == pytest.approx(0.5 * x)
    assert float(noise.epsilons[-1]) == pytest.approx(1.5 * x)


def test_asymmetric_binary_partition():
    """Binary channel at x=0.2: symbol 0 clean (eps=0.5x), symbol 1 noisy (eps=1.5x)."""
    x = 0.2
    noise, _ = _create_noise_and_decoding_metric(
        noise_type="asymmetric", error_rate=x, seq_length=8, alphabet_size=2,
    )
    assert isinstance(noise, AsymmetricChannel)
    m = noise.channel_matrix
    assert m[0, 0] == pytest.approx(1 - 0.5 * x)   # row 0: clean
    assert m[1, 1] == pytest.approx(1 - 1.5 * x)   # row 1: noisy


def test_asymmetric_odd_q_partition():
    """Odd q=3 channel: 1 clean, 1 normal, 1 noisy."""
    x = 0.1
    noise, _ = _create_noise_and_decoding_metric(
        noise_type="asymmetric", error_rate=x, seq_length=8, alphabet_size=3,
    )
    m = noise.channel_matrix
    assert m[0, 0] == pytest.approx(1 - 0.5 * x)   # clean
    assert m[1, 1] == pytest.approx(1 - 1.0 * x)   # normal
    assert m[2, 2] == pytest.approx(1 - 1.5 * x)   # noisy


def test_position_varying_asymmetric_binary_corners():
    """Position-varying asymmetric q=2 at x=0.2: corner eps match the multiplicative design."""
    x = 0.2
    L = 8
    noise, _ = _create_noise_and_decoding_metric(
        noise_type="position_varying_asymmetric", error_rate=x, seq_length=L, alphabet_size=2,
    )
    assert isinstance(noise, PositionVaryingAsymmetricChannel)
    chs = noise.channel_matrices  # (L, 2, 2)

    # position_factor ramps 0.5 → 1.5; class_factor = 0.5 (clean) / 1.5 (noisy).
    # Position 0 (position_factor=0.5):
    assert (1 - chs[0, 0, 0]) == pytest.approx(0.5 * 0.5 * x)  # 0.05
    assert (1 - chs[0, 1, 1]) == pytest.approx(1.5 * 0.5 * x)  # 0.15
    # Position L-1=7 (position_factor=1.5):
    assert (1 - chs[L - 1, 0, 0]) == pytest.approx(0.5 * 1.5 * x)  # 0.15
    assert (1 - chs[L - 1, 1, 1]) == pytest.approx(1.5 * 1.5 * x)  # 0.45


@pytest.mark.parametrize("noise_type", NOISE_TYPES)
@pytest.mark.parametrize("error_rate", ERROR_RATES)
@pytest.mark.parametrize("alphabet_size", ALPHABET_SIZES)
def test_factory_matches_evaluator_config(noise_type, error_rate, alphabet_size):
    """`_build_evaluator_config` must materialize bit-identical eps to
    `_create_noise_and_decoding_metric`.

    The runner builds a local NoiseChannel via the first function and hands
    DUET an EvaluatorConfig built via the second function. DUET reconstructs
    a NoiseChannel from that config and uses it for PEP. If the two paths
    drift, DUET and the per-codebook re-evaluator see different channels
    and HV/IGD become apples-to-oranges in a way no other test catches.
    """
    from duet.evaluator_config import create_noise_channel

    from scripts.benchmark.synthetic.run_2d_synthetic_benchmark import (
        _build_evaluator_config,
    )

    direct_noise, _ = _create_noise_and_decoding_metric(
        noise_type=noise_type,
        error_rate=error_rate,
        seq_length=SEQ_LENGTH,
        alphabet_size=alphabet_size,
    )
    eval_cfg = _build_evaluator_config(
        noise_type=noise_type,
        error_rate=error_rate,
        seq_length=SEQ_LENGTH,
        alphabet_size=alphabet_size,
        num_samples=1,
        num_cpus=1,
        seed=0,
    )
    config_noise = create_noise_channel(
        eval_cfg.noise_channel,
        seq_length=SEQ_LENGTH,
        alphabet_size=alphabet_size,
    )

    direct_chs = _channel_matrices(direct_noise, alphabet_size, SEQ_LENGTH)
    config_chs = _channel_matrices(config_noise, alphabet_size, SEQ_LENGTH)
    np.testing.assert_allclose(
        direct_chs, config_chs, atol=1e-12,
        err_msg=(
            f"Runner and DUET noise channels drifted for {noise_type=} "
            f"{error_rate=} {alphabet_size=}"
        ),
    )

"""
Configuration classes and factory functions for CodebookEvaluator components.

This module provides:
- ComponentConfig: Generic configuration for noise channels, decoding metrics, and decoding rules
- EvaluatorConfig: Complete evaluator specification
- Factory functions to instantiate components from configuration

The configuration system uses a type + params pattern where each component
specifies its type and a dictionary of type-specific parameters. This allows
for extensibility without modifying configuration classes.

Component Hierarchy (from simplest to most general):
----------------------------------------------------
1. symmetric: Scalar epsilon - same error rate for all positions and symbols
2. position_varying: 1D array epsilon[i] - per-position error rate (same for all symbols)
3. asymmetric: 1D array epsilon[a] - per-symbol error rate (same at all positions)
4. position_varying_asymmetric: 2D array epsilon[i,a] - per-position per-symbol error rates

Choose the simplest model that captures your noise characteristics.

Example YAML configuration:
    noise_channel:
      type: position_varying
      epsilon: [0.1, 0.15, 0.2, 0.25, 0.3]
    decoding_metric:
      type: position_varying_nll
      epsilon: [0.1, 0.15, 0.2, 0.25, 0.3]
    decoding_rule:
      type: unique_minimum
    num_samples: 5000
    num_cpus: 8

Arrays and matrices can be specified inline or loaded from .npy files:
    noise_channel:
      type: position_varying
      epsilon_path: /path/to/epsilon.npy
    decoding_metric:
      type: position_varying_nll
      epsilon_path: /path/to/epsilon.npy
    decoding_metric:
      type: weighted_hamming
      weights_path: /path/to/weights.npy
    noise_channel:
      type: asymmetric
      epsilon: [0.08, 0.12, 0.10, 0.09]  # Per-symbol error rates (A, T, C, G)
    decoding_metric:
      type: position_varying_asymmetric_nll
      epsilon_path: /path/to/epsilon_2d.npy  # Shape (seq_length, alphabet_size)

Decoding rule examples:
    decoding_rule:
      type: unique_minimum

    decoding_rule:
      type: margin
      margin: 2.0

    decoding_rule:
      type: pairwise_posterior_threshold
      threshold: 0.9

    decoding_rule:
      type: approximate_posterior_threshold
      threshold: 0.9
      n_eff: 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from duet.codebook_evaluator import (
    CodebookEvaluator,
    NoiseChannel,
    SymmetricEpsilon,
    PositionVaryingEpsilon,
    AsymmetricChannel,
    PositionVaryingAsymmetricChannel,
    DecodingMetric,
    HammingDistance,
    WeightedHammingDistance,
    SymmetricNLL,
    PositionVaryingNLL,
    AsymmetricNLL,
    PositionVaryingAsymmetricNLL,
    DecodingRule,
    UniqueMinimum,
    MarginDecoding,
    PairwisePosteriorThreshold,
    ApproximatePosteriorThreshold,
    SequenceEncoder,
    get_encoder,
)


# =============================================================================
# Generic Component Configuration
# =============================================================================


@dataclass
class ComponentConfig:
    """Generic configuration for a strategy component.

    This class provides a flexible way to configure noise channels, decoding metrics,
    and decoding rules. The `type` field identifies the component class, and
    `params` holds type-specific parameters as a dictionary.

    This design allows new component types to be added without modifying
    the configuration class itself - only the factory functions need updates.

    Attributes:
        type: Component type name (e.g., "symmetric", "position_varying", "hamming").
        params: Dictionary of type-specific parameters.

    Example:
        >>> # Symmetric noise channel (scalar epsilon)
        >>> config = ComponentConfig(type="symmetric", params={"epsilon": 0.1})

        >>> # Position-varying noise channel (1D array: per-position)
        >>> config = ComponentConfig(type="position_varying", params={"epsilon": [0.1, 0.15, 0.2]})

        >>> # Asymmetric noise (1D array: per-symbol)
        >>> config = ComponentConfig(type="asymmetric", params={"epsilon": [0.08, 0.12, 0.10, 0.09]})

        >>> # Position-varying asymmetric (2D array: per-position per-symbol)
        >>> config = ComponentConfig(type="position_varying_asymmetric", params={
        ...     "epsilon": [[0.1, 0.1, 0.1, 0.1], [0.2, 0.15, 0.15, 0.2]]
        ... })

        >>> # Symmetric NLL decoding metric (scalar epsilon)
        >>> config = ComponentConfig(type="symmetric_nll", params={"epsilon": 0.1})

        >>> # Position-varying NLL decoding metric (1D array: per-position)
        >>> config = ComponentConfig(type="position_varying_nll", params={"epsilon": [0.1, 0.15, 0.2]})

        >>> # Margin decoding
        >>> config = ComponentConfig(type="margin", params={"margin": 2.0})

        >>> # Pairwise posterior threshold decoding
        >>> config = ComponentConfig(type="pairwise_posterior_threshold", params={
        ...     "threshold": 0.9
        ... })

        >>> # Approximate posterior threshold decoding
        >>> config = ComponentConfig(type="approximate_posterior_threshold", params={
        ...     "threshold": 0.9, "n_eff": 3
        ... })

        >>> # Asymmetric with explicit matrix (from file)
        >>> config = ComponentConfig(type="asymmetric", params={
        ...     "channel_matrix_path": "/path/to/matrix.npy"
        ... })
    """

    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComponentConfig":
        """Create ComponentConfig from dictionary.

        The 'type' key is extracted and all remaining keys become params.

        Args:
            d: Dictionary with 'type' key and optional additional parameters.

        Returns:
            ComponentConfig instance.

        Raises:
            KeyError: If 'type' key is missing.
        """
        d = d.copy()
        type_ = d.pop("type")
        return cls(type=type_, params=d)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with 'type' and all params as top-level keys.
        """
        result = {"type": self.type}
        result.update(self.params)
        return result


# =============================================================================
# Main Evaluator Configuration
# =============================================================================


@dataclass
class EvaluatorConfig:
    """Complete configuration for CodebookEvaluator.

    This dataclass holds all configuration needed to create a CodebookEvaluator
    instance, including noise channel, decoding metric, decoding rule, and
    Monte Carlo sampling parameters.

    Note: alphabet_size is NOT stored here - it is derived from chemistry
    at the call site and passed explicitly to create_evaluator().

    Attributes:
        noise_channel: Configuration for noise generation.
        decoding_metric: Configuration for cost/decoding metric computation.
        decoding_rule: Configuration for decoding decisions.
        num_samples: Number of Monte Carlo samples per codeword.
        num_cpus: Number of CPUs for parallel computation.
        seed: Random seed for reproducibility.

    Example:
        >>> config = EvaluatorConfig(
        ...     noise_channel=ComponentConfig(type="position_varying", params={"epsilon": [0.1, 0.1]}),
        ...     decoding_metric=ComponentConfig(type="position_varying_nll", params={"epsilon": [0.1, 0.1]}),
        ...     decoding_rule=ComponentConfig(type="unique_minimum"),
        ...     num_samples=5000,
        ...     num_cpus=8,
        ... )
    """

    noise_channel: ComponentConfig
    decoding_metric: ComponentConfig
    decoding_rule: ComponentConfig
    num_samples: int = 5000
    num_cpus: int = 8
    seed: int | None = None
    scratch_dir: str | None = None
    device: str | None = None        # runtime placement; NOT part of the PEP fingerprint
    mem_budget_gb: float | None = None  # per-batch ceiling; NOT part of the fingerprint

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluatorConfig":
        """Create EvaluatorConfig from dictionary.

        Args:
            d: Dictionary with component configurations.

        Returns:
            EvaluatorConfig instance.
        """
        decoding_metric_config = d.get("decoding_metric")
        if decoding_metric_config is None:
            raise KeyError("Configuration must include 'decoding_metric' key")

        return cls(
            noise_channel=ComponentConfig.from_dict(d["noise_channel"]),
            decoding_metric=ComponentConfig.from_dict(decoding_metric_config),
            decoding_rule=ComponentConfig.from_dict(d["decoding_rule"]),
            num_samples=d.get("num_samples", 5000),
            num_cpus=d.get("num_cpus", 8),
            seed=d.get("seed"),
            scratch_dir=d.get("scratch_dir"),
            device=d.get("device"),
            mem_budget_gb=d.get("mem_budget_gb"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # NOTE: device/mem_budget_gb/scratch_dir are runtime knobs and are
        # intentionally excluded — they must not affect the PEP cache fingerprint.
        return {
            "noise_channel": self.noise_channel.to_dict(),
            "decoding_metric": self.decoding_metric.to_dict(),
            "decoding_rule": self.decoding_rule.to_dict(),
            "num_samples": self.num_samples,
            "num_cpus": self.num_cpus,
            "seed": self.seed,
        }

    @classmethod
    def default_symmetric(
        cls,
        epsilon: float,
        num_samples: int = 5000,
        num_cpus: int = 8,
        seed: int | None = None,
    ) -> "EvaluatorConfig":
        """Create default config with symmetric noise and Hamming distance.

        Args:
            epsilon: Uniform error probability.
            num_samples: Number of Monte Carlo samples.
            num_cpus: Number of CPUs for parallel computation.
            seed: Random seed.

        Returns:
            EvaluatorConfig with symmetric noise channel and Hamming distance.
        """
        return cls(
            noise_channel=ComponentConfig(type="symmetric", params={"epsilon": epsilon}),
            decoding_metric=ComponentConfig(type="hamming"),
            decoding_rule=ComponentConfig(type="unique_minimum"),
            num_samples=num_samples,
            num_cpus=num_cpus,
            seed=seed,
        )

    @classmethod
    def default_position_varying(
        cls,
        epsilon: List[float],
        num_samples: int = 5000,
        num_cpus: int = 8,
        seed: int | None = None,
    ) -> "EvaluatorConfig":
        """Create default config with position-varying noise and position-varying NLL decoding metric.

        Args:
            epsilon: Per-position error probabilities (1D array of length seq_length).
            num_samples: Number of Monte Carlo samples.
            num_cpus: Number of CPUs for parallel computation.
            seed: Random seed.

        Returns:
            EvaluatorConfig with position-varying noise and position-varying NLL decoding metric.
        """
        return cls(
            noise_channel=ComponentConfig(type="position_varying", params={"epsilon": epsilon}),
            decoding_metric=ComponentConfig(type="position_varying_nll", params={"epsilon": epsilon}),
            decoding_rule=ComponentConfig(type="unique_minimum"),
            num_samples=num_samples,
            num_cpus=num_cpus,
            seed=seed,
        )

    @classmethod
    def default_position_varying_pairwise(
        cls,
        epsilon: List[float],
        threshold: float = 0.9,
        num_samples: int = 5000,
        num_cpus: int = 8,
        seed: int | None = None,
    ) -> "EvaluatorConfig":
        """Create config with position-varying noise, NLL decoding metric, and pairwise posterior decoding.

        This configuration uses pairwise posterior threshold decoding, which only
        assigns a read to a codeword if ALL pairwise posteriors exceed the threshold.

        Args:
            epsilon: Per-position error probabilities (1D array of length seq_length).
            threshold: Minimum pairwise posterior probability (default 0.9).
            num_samples: Number of Monte Carlo samples.
            num_cpus: Number of CPUs for parallel computation.
            seed: Random seed.

        Returns:
            EvaluatorConfig with position-varying noise, NLL decoding metric, and
            pairwise posterior threshold decoding.
        """
        return cls(
            noise_channel=ComponentConfig(type="position_varying", params={"epsilon": epsilon}),
            decoding_metric=ComponentConfig(type="position_varying_nll", params={"epsilon": epsilon}),
            decoding_rule=ComponentConfig(type="pairwise_posterior_threshold",
                                          params={"threshold": threshold}),
            num_samples=num_samples,
            num_cpus=num_cpus,
            seed=seed,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def _load_array_param(
    params: Dict[str, Any],
    inline_key: str,
    path_key: str,
) -> np.ndarray | None:
    """Load an array parameter from inline value or file path.

    Args:
        params: Parameter dictionary from ComponentConfig.
        inline_key: Key for inline array value (e.g., "epsilon", "channel_matrix").
        path_key: Key for file path (e.g., "epsilon_path", "channel_matrix_path").

    Returns:
        Loaded array as numpy array, or None if neither key is present.
        Priority: inline_key > path_key

    Raises:
        FileNotFoundError: If path is specified but file doesn't exist.
        ValueError: If file cannot be loaded as numpy array.
    """
    # Check for inline value first (takes priority)
    inline_value = params.get(inline_key)
    if inline_value is not None:
        return np.asarray(inline_value, dtype=float)

    # Check for file path
    path = params.get(path_key)
    if path is not None:
        try:
            return np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Array file not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load array from {path}: {e}")

    return None


def _validate_1d_array(arr: np.ndarray, name: str, expected_length: int | None = None) -> np.ndarray:
    """Validate that array is 1D with optional length check.

    Args:
        arr: Array to validate.
        name: Parameter name for error messages.
        expected_length: If provided, verify array has this length.

    Returns:
        Validated array.

    Raises:
        ValueError: If validation fails.
    """
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1D array, got {arr.ndim}D array with shape {arr.shape}. "
            f"For scalar values, use 'symmetric' type instead."
        )
    if expected_length is not None and len(arr) != expected_length:
        raise ValueError(
            f"{name} length ({len(arr)}) must match expected length ({expected_length})"
        )
    return arr


def _validate_2d_array(
    arr: np.ndarray,
    name: str,
    expected_shape: tuple[int | None, int | None] | None = None
) -> np.ndarray:
    """Validate that array is 2D with optional shape check.

    Args:
        arr: Array to validate.
        name: Parameter name for error messages.
        expected_shape: If provided, tuple of (expected_rows, expected_cols).
                       Use None for dimensions that shouldn't be checked.

    Returns:
        Validated array.

    Raises:
        ValueError: If validation fails.
    """
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D array, got {arr.ndim}D array with shape {arr.shape}. "
            f"For 1D arrays, use 'asymmetric' type instead."
        )
    if expected_shape is not None:
        expected_rows, expected_cols = expected_shape
        if expected_rows is not None and arr.shape[0] != expected_rows:
            raise ValueError(
                f"{name} has {arr.shape[0]} rows, expected {expected_rows}"
            )
        if expected_cols is not None and arr.shape[1] != expected_cols:
            raise ValueError(
                f"{name} has {arr.shape[1]} columns, expected {expected_cols}"
            )
    return arr


def create_noise_channel(config: ComponentConfig, seq_length: int, alphabet_size: int = 4) -> NoiseChannel:
    """Create NoiseChannel instance from configuration.

    Supported types and their epsilon requirements:
        - "symmetric": Scalar epsilon (same for all positions and symbols)
            Params: epsilon (float), alphabet_size (int, optional - overrides config)

        - "position_varying": 1D array epsilon[i] (per-position, same for all symbols)
            Params: epsilon or epsilon_path (1D array of length seq_length),
            alphabet_size (int, optional - overrides config)

        - "asymmetric": 1D array epsilon[a] (per-symbol, same at all positions)
            Params: epsilon or epsilon_path (1D array of length alphabet_size),
            OR channel_matrix / channel_matrix_path (2D array of shape (tx_size, obs_size))

        - "position_varying_asymmetric": 2D array epsilon[i,a] (per-position per-symbol)
            Params: epsilon or epsilon_path (2D array of shape (seq_length, alphabet_size)),
            OR channel_matrices / channel_matrices_path (3D array of shape (L, tx_size, obs_size))

    Args:
        config: Component configuration with type and params.
        seq_length: Expected sequence length (for validation).
        alphabet_size: Default alphabet size from EvaluatorConfig.

    Returns:
        NoiseChannel instance.

    Raises:
        ValueError: If type is unknown, required params are missing, or array shapes are wrong.
    """
    # Allow per-component alphabet_size override
    effective_alphabet_size = config.params.get("alphabet_size", alphabet_size)

    if config.type == "symmetric":
        epsilon = config.params.get("epsilon")
        if epsilon is None:
            raise ValueError("SymmetricEpsilon requires 'epsilon' parameter")
        # Ensure it's a scalar
        epsilon_arr = np.asarray(epsilon)
        if epsilon_arr.ndim != 0:
            raise ValueError(
                f"SymmetricEpsilon requires scalar epsilon, got array with shape {epsilon_arr.shape}. "
                f"For per-position error rates, use 'position_varying' type. "
                f"For per-symbol error rates, use 'asymmetric' type."
            )
        return SymmetricEpsilon(epsilon=float(epsilon), alphabet_size=effective_alphabet_size)

    elif config.type == "position_varying":
        epsilon = _load_array_param(config.params, "epsilon", "epsilon_path")
        if epsilon is None:
            raise ValueError("PositionVaryingEpsilon requires 'epsilon' or 'epsilon_path' parameter")
        epsilon = _validate_1d_array(epsilon, "PositionVaryingEpsilon epsilon", expected_length=seq_length)
        return PositionVaryingEpsilon(epsilons=epsilon, alphabet_size=effective_alphabet_size)

    elif config.type == "asymmetric":
        # First check for explicit channel matrix
        channel_matrix = _load_array_param(
            config.params, "channel_matrix", "channel_matrix_path"
        )
        if channel_matrix is not None:
            return AsymmetricChannel(channel_matrix=channel_matrix)

        # Otherwise, build from per-symbol epsilon array
        epsilon = _load_array_param(config.params, "epsilon", "epsilon_path")
        if epsilon is None:
            raise ValueError(
                "AsymmetricChannel requires 'channel_matrix', 'channel_matrix_path', "
                "'epsilon', or 'epsilon_path' parameter"
            )
        epsilon = _validate_1d_array(epsilon, "AsymmetricChannel epsilon")
        # Note: alphabet_size is inferred from epsilon length
        return AsymmetricChannel.from_symbol_epsilons(epsilon)

    elif config.type == "position_varying_asymmetric":
        # First check for explicit channel matrices
        channel_matrices = _load_array_param(
            config.params, "channel_matrices", "channel_matrices_path"
        )
        if channel_matrices is not None:
            if channel_matrices.ndim != 3:
                raise ValueError(
                    f"channel_matrices must be 3D, got {channel_matrices.ndim}D"
                )
            if channel_matrices.shape[0] != seq_length:
                raise ValueError(
                    f"PositionVaryingAsymmetricChannel channel_matrices has "
                    f"{channel_matrices.shape[0]} positions, expected {seq_length}"
                )
            return PositionVaryingAsymmetricChannel(channel_matrices=channel_matrices)

        # Otherwise, build from per-position per-symbol epsilon array
        epsilon = _load_array_param(config.params, "epsilon", "epsilon_path")
        if epsilon is None:
            raise ValueError(
                "PositionVaryingAsymmetricChannel requires 'channel_matrices', "
                "'channel_matrices_path', 'epsilon', or 'epsilon_path' parameter"
            )
        epsilon = _validate_2d_array(epsilon, "PositionVaryingAsymmetricChannel epsilon",
                                     expected_shape=(seq_length, None))
        # Note: alphabet_size is inferred from epsilon shape
        return PositionVaryingAsymmetricChannel.from_positional_symbol_epsilons(epsilon)

    else:
        raise ValueError(f"Unknown noise channel type: {config.type}")


def create_decoding_metric(config: ComponentConfig, seq_length: int, alphabet_size: int = 4) -> DecodingMetric:
    """Create DecodingMetric instance from configuration.

    Supported types and their epsilon requirements:
        - "hamming": No params required.

        - "weighted_hamming": 1D array weights[i] (per-position)
            Params: weights or weights_path (1D array of length seq_length)

        - "symmetric_nll": Scalar epsilon (same for all positions)
            Params: epsilon (float), alphabet_size (int, optional - overrides config)
            This is a simplified version of position_varying_nll where all positions
            share the same error probability.

        - "position_varying_nll": 1D array epsilon[i] (per-position, same for all symbols)
            Params: epsilon or epsilon_path (1D array of length seq_length),
            alphabet_size (int, optional - overrides config)

        - "asymmetric_nll": 1D array epsilon[a] (per-symbol, same at all positions)
            Params: epsilon or epsilon_path (1D array of length alphabet_size),
            OR channel_matrix / channel_matrix_path (2D array of shape (tx_size, obs_size))

        - "position_varying_asymmetric_nll": 2D array epsilon[i,a] (per-position per-symbol)
            Params: epsilon or epsilon_path (2D array of shape (seq_length, alphabet_size)),
            OR channel_matrices / channel_matrices_path (3D array of shape (L, tx_size, obs_size))

    Args:
        config: Component configuration with type and params.
        seq_length: Expected sequence length (for validation).
        alphabet_size: Default alphabet size from EvaluatorConfig.

    Returns:
        DecodingMetric instance.

    Raises:
        ValueError: If type is unknown, required params are missing, or array shapes are wrong.
    """
    # Allow per-component alphabet_size override
    effective_alphabet_size = config.params.get("alphabet_size", alphabet_size)

    if config.type == "hamming":
        return HammingDistance(alphabet_size=effective_alphabet_size)

    elif config.type == "weighted_hamming":
        weights = _load_array_param(config.params, "weights", "weights_path")
        if weights is None:
            raise ValueError("WeightedHammingDistance requires 'weights' or 'weights_path' parameter")
        weights = _validate_1d_array(weights, "WeightedHammingDistance weights",
                                     expected_length=seq_length)
        return WeightedHammingDistance(weights=weights, alphabet_size=effective_alphabet_size)

    elif config.type == "symmetric_nll":
        epsilon = config.params.get("epsilon")
        if epsilon is None:
            raise ValueError("SymmetricNLL requires 'epsilon' parameter")
        # Ensure it's a scalar
        epsilon_arr = np.asarray(epsilon)
        if epsilon_arr.ndim != 0:
            raise ValueError(
                f"SymmetricNLL requires scalar epsilon, got array with shape {epsilon_arr.shape}. "
                f"For per-position error rates, use 'position_varying_nll' type."
            )
        return SymmetricNLL(epsilon=float(epsilon), alphabet_size=effective_alphabet_size)

    elif config.type == "position_varying_nll":
        epsilon = _load_array_param(config.params, "epsilon", "epsilon_path")
        if epsilon is None:
            raise ValueError("PositionVaryingNLL requires 'epsilon' or 'epsilon_path' parameter")
        epsilon = _validate_1d_array(epsilon, "PositionVaryingNLL epsilon",
                                     expected_length=seq_length)
        return PositionVaryingNLL(epsilons=epsilon, alphabet_size=effective_alphabet_size)

    elif config.type == "asymmetric_nll":
        # First check for explicit channel matrix
        channel_matrix = _load_array_param(
            config.params, "channel_matrix", "channel_matrix_path"
        )
        if channel_matrix is not None:
            return AsymmetricNLL(channel_matrix=channel_matrix)

        # Otherwise, build from per-symbol epsilon array
        epsilon = _load_array_param(config.params, "epsilon", "epsilon_path")
        if epsilon is None:
            raise ValueError(
                "AsymmetricNLL requires 'channel_matrix', 'channel_matrix_path', "
                "'epsilon', or 'epsilon_path' parameter"
            )
        epsilon = _validate_1d_array(epsilon, "AsymmetricNLL epsilon")
        # Note: alphabet_size is inferred from epsilon length
        return AsymmetricNLL.from_symbol_epsilons(epsilon)

    elif config.type == "position_varying_asymmetric_nll":
        # First check for explicit channel matrices
        channel_matrices = _load_array_param(
            config.params, "channel_matrices", "channel_matrices_path"
        )
        if channel_matrices is not None:
            if channel_matrices.ndim != 3:
                raise ValueError(
                    f"channel_matrices must be 3D, got {channel_matrices.ndim}D"
                )
            if channel_matrices.shape[0] != seq_length:
                raise ValueError(
                    f"PositionVaryingAsymmetricNLL channel_matrices has "
                    f"{channel_matrices.shape[0]} positions, expected {seq_length}"
                )
            return PositionVaryingAsymmetricNLL(channel_matrices=channel_matrices)

        # Otherwise, build from per-position per-symbol epsilon array
        epsilon = _load_array_param(config.params, "epsilon", "epsilon_path")
        if epsilon is None:
            raise ValueError(
                "PositionVaryingAsymmetricNLL requires 'channel_matrices', "
                "'channel_matrices_path', 'epsilon', or 'epsilon_path' parameter"
            )
        epsilon = _validate_2d_array(epsilon, "PositionVaryingAsymmetricNLL epsilon",
                                     expected_shape=(seq_length, None))
        # Note: alphabet_size is inferred from epsilon shape
        return PositionVaryingAsymmetricNLL.from_positional_symbol_epsilons(epsilon)

    else:
        raise ValueError(f"Unknown decoding metric type: {config.type}")


def create_decoding_rule(config: ComponentConfig) -> DecodingRule:
    """Create DecodingRule instance from configuration.

    Supported types:
        - "unique_minimum": UniqueMinimum
            No params required.
            Decodes to the codeword with uniquely minimum cost; rejects ties.
            Equivalent to pairwise_posterior_threshold with threshold=0.5.

        - "margin": MarginDecoding
            Params: margin (float, default 1.0)
            A codeword competes if cost <= transmitted_cost + margin.
            Equivalent to pairwise_posterior_threshold with threshold = σ(margin).

        - "pairwise_posterior_threshold": PairwisePosteriorThreshold
            Params: threshold (float, default 0.5)
            Decodes only if ALL pairwise posteriors exceed threshold.
            This is the recommended decoding rule for use with PEP matrices.

        - "approximate_posterior_threshold": ApproximatePosteriorThreshold
            Params: threshold (float, default 0.5), n_eff (int, default 3)
            Approximates full posterior threshold by converting to pairwise.
            WARNING: This is an approximation; see class docstring for details.

    Args:
        config: Component configuration with type and params.

    Returns:
        DecodingRule instance.

    Raises:
        ValueError: If type is unknown or parameters are invalid.

    Example YAML configurations:
        # Simple unique minimum (default)
        decoding_rule:
          type: unique_minimum

        # Margin-based with k=2
        decoding_rule:
          type: margin
          margin: 2.0

        # Pairwise posterior threshold
        decoding_rule:
          type: pairwise_posterior_threshold
          threshold: 0.9

        # Approximate full posterior threshold
        decoding_rule:
          type: approximate_posterior_threshold
          threshold: 0.9
          n_eff: 3
    """
    if config.type == "unique_minimum":
        return UniqueMinimum()

    elif config.type == "margin":
        margin = config.params.get("margin", 1.0)
        return MarginDecoding(k=float(margin))

    elif config.type == "pairwise_posterior_threshold":
        threshold = config.params.get("threshold", 0.5)
        return PairwisePosteriorThreshold(threshold=float(threshold))

    elif config.type == "approximate_posterior_threshold":
        threshold = config.params.get("threshold", 0.5)
        n_eff = config.params.get("n_eff", 3)
        return ApproximatePosteriorThreshold(threshold=float(threshold), n_eff=int(n_eff))

    else:
        raise ValueError(f"Unknown decoding rule type: {config.type}")


def create_evaluator(
    library,
    config: EvaluatorConfig,
    alphabet_size: int,
) -> CodebookEvaluator:
    """Create fully configured CodebookEvaluator from sequences and configuration.

    This is the generic factory function that creates a complete CodebookEvaluator
    instance from a list of string sequences (or a pre-encoded numpy array) and
    an EvaluatorConfig. It automatically selects the appropriate encoder based on
    alphabet_size when the library is a list of strings.

    Args:
        library: List of string sequences (DNA, hex, binary, etc.) or a 2-D numpy
            integer array of shape (N, L) containing pre-encoded sequences.  When
            a numpy array is supplied the encoder step is skipped and the array is
            used directly as the codebook.
        config: Complete evaluator configuration.
        alphabet_size: Alphabet size for encoding (e.g., 4 for DNA, 16 for hex).
                      This should be derived from chemistry at the call site.

    Returns:
        CodebookEvaluator instance ready for use.

    Raises:
        ValueError: If library is empty, configuration is invalid, or no encoder
                   is registered for the given alphabet_size (string path only).

    Example:
        >>> # DNA sequences
        >>> config = EvaluatorConfig.default_position_varying(epsilon=[0.1, 0.1, 0.1])
        >>> evaluator = create_evaluator(["ATG", "CGT", "TAC"], config, alphabet_size=4)

        >>> # Hex sequences (dual-guide)
        >>> config = EvaluatorConfig.default_position_varying(epsilon=[0.1, 0.1, 0.1])
        >>> evaluator = create_evaluator(["0A5", "F3B", "C7E"], config, alphabet_size=16)

        >>> # Pre-encoded integer array (synthetic benchmarks)
        >>> evaluator = create_evaluator(np.array([[0,1],[1,0]], dtype=np.int8), config, alphabet_size=2)
    """
    if isinstance(library, np.ndarray):
        # Pre-encoded integer sequences: bypass encoder, build components directly.
        if library.ndim != 2:
            raise ValueError(
                f"library array must be 2-D (N, L), got {library.ndim}-D"
            )
        if library.size == 0:
            raise ValueError("Cannot create evaluator with empty library")
        seq_length = library.shape[1]
        noise_channel = create_noise_channel(config.noise_channel, seq_length, alphabet_size)
        decoding_metric = create_decoding_metric(
            config.decoding_metric, seq_length, alphabet_size
        )
        decoding_rule = create_decoding_rule(config.decoding_rule)
        return CodebookEvaluator(
            codebook=library,
            noise_channel=noise_channel,
            decoding_metric=decoding_metric,
            decoding_rule=decoding_rule,
            n_samples=config.num_samples,
            seed=config.seed,
            scratch_dir=config.scratch_dir,
        )

    if not library:
        raise ValueError("Cannot create evaluator with empty library")

    # Get appropriate encoder
    encoder = get_encoder(alphabet_size)

    seq_length = len(library[0])

    # Create components with alphabet_size passed through
    noise_channel = create_noise_channel(config.noise_channel, seq_length, alphabet_size)
    decoding_metric = create_decoding_metric(
        config.decoding_metric, seq_length, alphabet_size
    )
    decoding_rule = create_decoding_rule(config.decoding_rule)

    return CodebookEvaluator.from_sequence_list(
        library,
        encoder=encoder,
        noise_channel=noise_channel,
        decoding_metric=decoding_metric,
        decoding_rule=decoding_rule,
        n_samples=config.num_samples,
        seed=config.seed,
        scratch_dir=config.scratch_dir,
    )


def create_dna_evaluator(library: List[str], config: EvaluatorConfig) -> CodebookEvaluator:
    """Create fully configured CodebookEvaluator from DNA sequences and configuration.

    This is a convenience wrapper around create_evaluator that forces DNA encoding
    (alphabet_size=4) regardless of config.alphabet_size. Use this when you know
    your sequences are DNA and want to ensure DNA encoding is used.

    Args:
        library: List of DNA sequences.
        config: Complete evaluator configuration.

    Returns:
        CodebookEvaluator instance ready for use.

    Raises:
        ValueError: If library is empty or configuration is invalid.

    Example:
        >>> config = EvaluatorConfig.default_position_varying(epsilon=[0.1, 0.1, 0.1])
        >>> evaluator = create_dna_evaluator(["ATG", "CGT", "TAC"], config)
    """
    return create_evaluator(library, config, alphabet_size=4)
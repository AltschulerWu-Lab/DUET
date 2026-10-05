"""Protective tests for the numpy-array branches of the library paths.

The 2-D synthetic benchmark generates pre-encoded integer sequences (np.int8
arrays) rather than character strings, so library functions grew numpy-aware
code paths:

- duet.providers.hash_library
- duet.evaluator_config.create_evaluator

These tests pin the new behavior so a future refactor doesn't silently
regress the integer-array path.
"""
from __future__ import annotations

import numpy as np
import pytest

from duet.evaluator_config import EvaluatorConfig, create_evaluator
from duet.providers import hash_library


class TestHashLibraryNumpy:

    def test_returns_16_char_hex(self):
        arr = np.array([[0, 1], [1, 0]], dtype=np.int8)
        h = hash_library(arr)
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_dtype_invariant(self):
        a8 = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int8)
        a16 = a8.astype(np.int16)
        a32 = a8.astype(np.int32)
        assert hash_library(a8) == hash_library(a16) == hash_library(a32)

    def test_distinct_from_string_hash(self):
        arr = np.array([[0, 1], [1, 0]], dtype=np.int8)
        strings = ["01", "10"]
        assert hash_library(arr) != hash_library(strings)


class TestCreateEvaluatorNumpyArray:

    def _config(self) -> EvaluatorConfig:
        return EvaluatorConfig.default_symmetric(
            epsilon=0.1, num_samples=50, num_cpus=1
        )

    def test_2d_int8_returns_evaluator(self):
        arr = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int8)
        evaluator = create_evaluator(arr, self._config(), alphabet_size=2)
        assert evaluator.codebook.shape == arr.shape

    def test_1d_array_raises_value_error(self):
        arr = np.array([0, 1, 0, 1], dtype=np.int8)
        with pytest.raises(ValueError, match="2-D"):
            create_evaluator(arr, self._config(), alphabet_size=2)

    def test_empty_array_raises_value_error(self):
        arr = np.zeros((0, 4), dtype=np.int8)
        with pytest.raises(ValueError, match="empty"):
            create_evaluator(arr, self._config(), alphabet_size=2)

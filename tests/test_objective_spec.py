"""Tests for ObjectiveSpec — key-set validation and picklability."""
import pickle
from typing import Dict

import pytest

from duet.runner.core import ObjectiveSpec, SwapCache


def _stub_cache_factory(candidates, *, codeword_to_group, candidate_to_sequence_idx=None) -> SwapCache:
    # Module-level so the enclosing ObjectiveSpec is picklable.
    return object()  # real cache built in other tests; this is just a shape stub


def _weights_ok(lambda_: float) -> Dict[str, float]:
    return {"decode": lambda_, "score": 1.0 - lambda_}


def _weights_bad_key(lambda_: float) -> Dict[str, float]:
    # Missing "score"; has "bogus" instead.
    return {"decode": lambda_, "bogus": 1.0 - lambda_}


def _weights_lambda_dependent_keys(lambda_: float) -> Dict[str, float]:
    # Drops "score" at lambda=1.0 (decode-only) — must be caught by endpoint probing.
    if lambda_ == 1.0:
        return {"decode": 1.0}
    return {"decode": lambda_, "score": 1.0 - lambda_}


class TestObjectiveSpecValidation:
    def test_valid_spec_constructs(self):
        spec = ObjectiveSpec(
            cache_factories={"score": _stub_cache_factory},
            weight_schedule=_weights_ok,
        )
        assert "score" in spec.cache_factories

    def test_mismatched_key_raises(self):
        with pytest.raises(ValueError, match="do not match"):
            ObjectiveSpec(
                cache_factories={"score": _stub_cache_factory},
                weight_schedule=_weights_bad_key,
            )

    def test_lambda_dependent_keys_caught_at_endpoint(self):
        with pytest.raises(ValueError, match="lambda=1.0"):
            ObjectiveSpec(
                cache_factories={"score": _stub_cache_factory},
                weight_schedule=_weights_lambda_dependent_keys,
            )

    def test_spec_is_picklable(self):
        spec = ObjectiveSpec(
            cache_factories={"score": _stub_cache_factory},
            weight_schedule=_weights_ok,
        )
        roundtrip = pickle.loads(pickle.dumps(spec))
        # Roundtrip preserves the function references.
        assert roundtrip.weight_schedule(0.5) == {"decode": 0.5, "score": 0.5}

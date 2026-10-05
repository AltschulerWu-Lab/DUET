"""Tests for GPU utility functions (parse_device, get_gpu_count).

These tests mock GPU availability so they run on any machine — no GPU required.
"""

import pytest
from unittest.mock import patch

from duet.gpu_utils import parse_device, get_gpu_count


class TestParseDevice:
    """Test device string parsing."""

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_gpu_default(self, _mock):
        assert parse_device("gpu") == [0]

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_gpu_single(self, _mock):
        assert parse_device("gpu:0") == [0]

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_gpu_single_nonzero(self, _mock):
        assert parse_device("gpu:2") == [2]

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_gpu_multi(self, _mock):
        assert parse_device("gpu:0,2") == [0, 2]

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_gpu_all(self, _mock):
        assert parse_device("gpu:all") == [0, 1, 2, 3]

    @patch("duet.gpu_utils.get_gpu_count", return_value=1)
    def test_gpu_all_single_device(self, _mock):
        assert parse_device("gpu:all") == [0]

    def test_invalid_prefix(self):
        with pytest.raises(ValueError, match="must start with 'gpu'"):
            parse_device("cpu")

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_invalid_format(self, _mock):
        with pytest.raises(ValueError, match="Invalid device string"):
            parse_device("gpufoo")

    @patch("duet.gpu_utils.get_gpu_count", return_value=2)
    def test_out_of_range(self, _mock):
        with pytest.raises(ValueError, match="out of range"):
            parse_device("gpu:3")

    @patch("duet.gpu_utils.get_gpu_count", return_value=0)
    def test_no_gpus_all(self, _mock):
        with pytest.raises(ValueError, match="no GPUs are available"):
            parse_device("gpu:all")

    @patch("duet.gpu_utils.get_gpu_count", return_value=0)
    def test_no_gpus_default(self, _mock):
        with pytest.raises(ValueError, match="no GPUs are available"):
            parse_device("gpu")

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_gpu_multi_with_spaces(self, _mock):
        assert parse_device("gpu:0, 2") == [0, 2]

    @patch("duet.gpu_utils.get_gpu_count", return_value=4)
    def test_non_integer_id(self, _mock):
        with pytest.raises(ValueError, match="Invalid GPU ID"):
            parse_device("gpu:abc")

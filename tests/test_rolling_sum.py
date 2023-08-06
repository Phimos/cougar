import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cougar import rolling_sum
from cougar.numpy import rolling_sum as rolling_sum_np
from cougar.pandas import rolling_sum as rolling_sum_pd


def assert_array_equal_with_nan(a, b):
    assert_array_equal(np.isnan(a), np.isnan(b))
    assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-5, atol=1e-6)


def test_rolling_sum_float64():
    array = np.random.randn(100)
    assert_array_equal_with_nan(rolling_sum(array, 10), rolling_sum_np(array, 10))


def test_rolling_sum_float32():
    array = np.random.randn(100).astype(np.float32)
    assert_array_equal_with_nan(rolling_sum(array, 10), rolling_sum_np(array, 10))


def test_rolling_sum_int64():
    array = np.random.randint(100, size=100)
    assert_array_equal_with_nan(rolling_sum(array, 10), rolling_sum_np(array, 10))


def test_rolling_sum_int32():
    array = np.random.randint(100, size=100).astype(np.int32)
    assert_array_equal_with_nan(rolling_sum(array, 10), rolling_sum_np(array, 10))


def test_rolling_sum_multi_dim():
    array = np.random.randn(100, 10)
    assert_array_equal_with_nan(
        rolling_sum(array, 10, axis=0),
        rolling_sum_np(array, 10, axis=0),
    )


@pytest.mark.benchmark(group="rolling_sum")
def test_rolling_sum_float64_cougar(benchmark):
    array = np.random.randn(1_000_000)
    benchmark(rolling_sum, array, 1024)


@pytest.mark.benchmark(group="rolling_sum")
def test_rolling_sum_float64_pandas(benchmark):
    array = np.random.randn(1_000_000)
    benchmark(rolling_sum_pd, array, 1024)

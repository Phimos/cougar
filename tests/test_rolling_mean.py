import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cougar import rolling_mean
from cougar.numpy import rolling_mean as rolling_mean_np
from cougar.pandas import rolling_mean as rolling_mean_pd


def assert_array_equal_with_nan(a, b):
    assert_array_equal(np.isnan(a), np.isnan(b))
    assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-6)


def test_rolling_mean_float64():
    array = np.random.randn(100)
    assert_array_equal_with_nan(rolling_mean(array, 10), rolling_mean_np(array, 10))


def test_rolling_mean_float32():
    array = np.random.randn(100).astype(np.float32)
    assert_array_equal_with_nan(rolling_mean(array, 10), rolling_mean_np(array, 10))


def test_rolling_mean_int64():
    array = np.random.randint(100, size=100)
    assert_array_equal_with_nan(rolling_mean(array, 10), rolling_mean_np(array, 10))


def test_rolling_mean_int32():
    array = np.random.randint(100, size=100).astype(np.int32)
    assert_array_equal_with_nan(rolling_mean(array, 10), rolling_mean_np(array, 10))


def test_rolling_mean_multi_dim():
    array = np.random.randn(100, 10)
    assert_array_equal_with_nan(
        rolling_mean(array, 10, axis=0),
        rolling_mean_np(array, 10, axis=0),
    )


@pytest.mark.benchmark(group="rolling_mean")
def test_rolling_mean_float64_cougar(benchmark):
    array = np.random.randn(1_000_000)
    benchmark(rolling_mean, array, 1024)


@pytest.mark.benchmark(group="rolling_mean")
def test_rolling_mean_float64_pandas(benchmark):
    array = np.random.randn(1_000_000)
    benchmark(rolling_mean_pd, array, 1024)

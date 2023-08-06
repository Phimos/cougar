import numpy as np
import pytest
from numpy.testing import assert_allclose

from cougar import rolling_sum
from cougar.numpy import rolling_sum as rolling_sum_np
from cougar.pandas import rolling_sum as rolling_sum_pd

array_float64_1M = np.random.randn(1_000_000)
array_float32_1M = np.random.randn(1_000_000).astype(np.float32)
array_int64_1M = np.random.randint(100, size=1_000_000)
array_int32_1M = np.random.randint(100, size=1_000_000).astype(np.int32)


class TestRollingSum:
    def test_rolling_sum_float64(self):
        for _ in range(10):
            array = np.random.randn(100)
            assert_allclose(
                rolling_sum(array, 10),
                rolling_sum_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    def test_rolling_sum_float32(self):
        for _ in range(10):
            array = np.random.randn(100).astype(np.float32)
            assert_allclose(
                rolling_sum(array, 10),
                rolling_sum_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    def test_rolling_sum_int64(self):
        for _ in range(10):
            array = np.random.randint(100, size=100)
            assert_allclose(
                rolling_sum(array, 10),
                rolling_sum_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    def test_rolling_sum_int32(self):
        for _ in range(10):
            array = np.random.randint(100, size=100).astype(np.int32)
            assert_allclose(
                rolling_sum(array, 10),
                rolling_sum_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    @pytest.mark.benchmark(group="rolling_sum_float64", disable_gc=True, warmup=True)
    def test_rolling_sum_float64_cougar(self, benchmark):
        benchmark(rolling_sum, array_float64_1M, 1024)

    @pytest.mark.benchmark(group="rolling_sum_float64", disable_gc=True, warmup=True)
    def test_rolling_sum_float64_pandas(self, benchmark):
        benchmark(rolling_sum_pd, array_float64_1M, 1024)

    @pytest.mark.benchmark(group="rolling_sum_int64", disable_gc=True, warmup=True)
    def test_rolling_sum_int64_cougar(self, benchmark):
        benchmark(rolling_sum, array_float64_1M, 1024)

    @pytest.mark.benchmark(group="rolling_sum_int64", disable_gc=True, warmup=True)
    def test_rolling_sum_int64_pandas(self, benchmark):
        benchmark(rolling_sum_pd, array_float64_1M, 1024)

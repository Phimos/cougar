import numpy as np
import pytest
from numpy.testing import assert_allclose

from cougar import rolling_min
from cougar.numpy import rolling_min as rolling_min_np
from cougar.pandas import rolling_min as rolling_min_pd

array_float64_1M = np.random.randn(1_000_000)
array_float32_1M = np.random.randn(1_000_000).astype(np.float32)
array_int64_1M = np.random.randint(100, size=1_000_000)
array_int32_1M = np.random.randint(100, size=1_000_000).astype(np.int32)


class TestRollingMin:
    def test_rolling_min_float64(self):
        for _ in range(10):
            array = np.random.randn(100)
            print(rolling_min(array, 10))
            print(rolling_min_np(array, 10))
            assert_allclose(
                rolling_min(array, 10),
                rolling_min_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    def test_rolling_min_float32(self):
        for _ in range(10):
            array = np.random.randn(100).astype(np.float32)
            assert_allclose(
                rolling_min(array, 10),
                rolling_min_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    def test_rolling_min_int64(self):
        for _ in range(10):
            array = np.random.randint(100, size=100)
            assert_allclose(
                rolling_min(array, 10),
                rolling_min_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    def test_rolling_min_int32(self):
        for _ in range(10):
            array = np.random.randint(100, size=100).astype(np.int32)
            assert_allclose(
                rolling_min(array, 10),
                rolling_min_np(array, 10),
                atol=1e-5,
                equal_nan=True,
            )

    @pytest.mark.benchmark(group="rolling_min_float64", disable_gc=True, warmup=True)
    def test_rolling_min_float64_cougar(self, benchmark):
        benchmark(rolling_min, array_float64_1M, 1024)

    @pytest.mark.benchmark(group="rolling_min_float64", disable_gc=True, warmup=True)
    def test_rolling_min_float64_pandas(self, benchmark):
        benchmark(rolling_min_pd, array_float64_1M, 1024)

    @pytest.mark.benchmark(group="rolling_min_int64", disable_gc=True, warmup=True)
    def test_rolling_min_int64_cougar(self, benchmark):
        benchmark(rolling_min, array_float64_1M, 1024)

    @pytest.mark.benchmark(group="rolling_min_int64", disable_gc=True, warmup=True)
    def test_rolling_min_int64_pandas(self, benchmark):
        benchmark(rolling_min_pd, array_float64_1M, 1024)

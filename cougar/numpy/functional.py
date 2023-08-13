import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pad(
    arr: np.ndarray, n: int, axis: int = -1, constant_values: float = np.nan
) -> np.ndarray:
    widths = [(0, 0) for _ in range(arr.ndim)]
    widths[axis] = (n, 0) if n > 0 else (0, -n)
    return np.pad(arr, widths, "constant", constant_values=constant_values)


def sliding_window(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return sliding_window_view(arr, window, axis=axis)


def rolling_sum(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).sum(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )


def rolling_mean(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).mean(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )


def rolling_std(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).std(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )


def rolling_skew(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).skew(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )


def rolling_kurt(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).kurt(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )


def rolling_max(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).max(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )


def rolling_min(arr: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    return pad(
        sliding_window(arr, window, axis=axis).min(axis=-1).astype(np.float64),
        window - 1,
        axis,
    )

from typing import Optional

import numpy as np
import pandas as pd


def rolling_mean(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling mean of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling mean.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling mean of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).mean().values


def rolling_sum(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling sum of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling sum.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling sum of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).sum().values


def rolling_min(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling min of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling min.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling min of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).min().values


def rolling_max(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling max of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling max.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling max of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).max().values


def rolling_std(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling standard deviation of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling standard deviation.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling standard deviation of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).std().values


def rolling_var(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling variance of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling variance.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling variance of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).var().values


def rolling_skew(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling skewness of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling skewness.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling skewness of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).skew().values


def rolling_kurt(
    arr: np.ndarray, window: int, min_count: Optional[int] = None
) -> np.ndarray:
    """Rolling kurtosis of an array.

    Args:
        arr (np.ndarray): Array to calculate rolling kurtosis.
        window (int): Window size.
        min_count (int, optional): Minimum number of non-NaN values in window. Defaults to None.

    Returns:
        np.ndarray: Rolling kurtosis of array.
    """
    return pd.Series(arr).rolling(window, min_periods=min_count).kurt().values

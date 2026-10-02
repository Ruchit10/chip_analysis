"""Histogram-space statistics for CHIP analysis.

This module keeps all calculations in binned-count space to avoid expanding
histograms into pseudo-sample arrays, which is not feasible at gnomAD v4 scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TwoSampleStats:
    """Container for two-sample test results on binned data."""

    statistic: float
    p_value: float
    n1: int
    n2: int


def _to_float_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array-like input.")
    return arr


def _nonnegative_int_counts(counts: Sequence[float]) -> np.ndarray:
    arr = _to_float_array(counts)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Counts contain non-finite values.")
    if np.any(arr < 0):
        raise ValueError("Counts must be non-negative.")
    # Counts are expected to be integers from Hail aggregations; round defensively.
    return np.rint(arr).astype(np.int64)


def expand_hist(repeat_numbers: Sequence[float], bin_numbers: Sequence[float]) -> np.ndarray:
    """Expand a histogram to repeated bin-floor observations (reference only)."""
    counts = _nonnegative_int_counts(repeat_numbers)
    bins = _to_float_array(bin_numbers)
    if len(counts) != len(bins):
        raise ValueError("repeat_numbers and bin_numbers must have the same length.")
    return np.repeat(bins, counts)


def _distinct_nonzero_bins(counts: np.ndarray) -> int:
    return int(np.count_nonzero(counts > 0))


def _value_at_position(counts: np.ndarray, bin_floors: np.ndarray, pos0: int) -> float:
    csum = np.cumsum(counts)
    idx = int(np.searchsorted(csum, pos0 + 1, side="left"))
    return float(bin_floors[idx])


def histogram_median(counts: Sequence[float], bin_floors: Sequence[float]) -> float:
    """Median of the expanded histogram without expansion."""
    c = _nonnegative_int_counts(counts)
    b = _to_float_array(bin_floors)
    if len(c) != len(b):
        raise ValueError("counts and bin_floors must have the same length.")
    n = int(c.sum())
    if n == 0:
        return np.nan
    if n % 2 == 1:
        return _value_at_position(c, b, n // 2)
    left = _value_at_position(c, b, (n // 2) - 1)
    right = _value_at_position(c, b, n // 2)
    return float((left + right) / 2.0)


def ks_2samp_from_hist(
    counts1: Sequence[float], counts2: Sequence[float]
) -> TwoSampleStats:
    """Kolmogorov-Smirnov two-sample test from aligned histogram counts."""
    c1 = _nonnegative_int_counts(counts1)
    c2 = _nonnegative_int_counts(counts2)
    if len(c1) != len(c2):
        raise ValueError("counts1 and counts2 must have the same length.")
    n1 = int(c1.sum())
    n2 = int(c2.sum())
    if n1 == 0 or n2 == 0:
        return TwoSampleStats(np.nan, np.nan, n1, n2)
    # Match legacy behavior: require >=2 distinct values in both groups.
    if _distinct_nonzero_bins(c1) <= 1 or _distinct_nonzero_bins(c2) <= 1:
        return TwoSampleStats(np.nan, np.nan, n1, n2)

    cdf1 = np.cumsum(c1) / n1
    cdf2 = np.cumsum(c2) / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))
    n_eff = int(np.rint((n1 * n2) / (n1 + n2)))
    if n_eff < 1:
        return TwoSampleStats(d, np.nan, n1, n2)
    p = float(stats.kstwo.sf(d, n_eff))
    return TwoSampleStats(d, p, n1, n2)


def _mood_counts_against_median(
    counts: np.ndarray, bin_floors: np.ndarray, median_value: float
) -> Tuple[int, int]:
    # scipy.stats.median_test default ties='below' behavior:
    # values equal to median are counted in the below group.
    above = int(counts[bin_floors > median_value].sum())
    below = int(counts.sum() - above)
    return above, below


def mood_median_test_from_hist(
    counts1: Sequence[float],
    counts2: Sequence[float],
    bin_floors: Sequence[float],
) -> TwoSampleStats:
    """Mood's median test p-value from aligned histogram counts."""
    c1 = _nonnegative_int_counts(counts1)
    c2 = _nonnegative_int_counts(counts2)
    bins = _to_float_array(bin_floors)
    if len(c1) != len(c2) or len(c1) != len(bins):
        raise ValueError("counts1, counts2, and bin_floors must share length.")
    n1 = int(c1.sum())
    n2 = int(c2.sum())
    if n1 == 0 or n2 == 0:
        return TwoSampleStats(np.nan, np.nan, n1, n2)
    if _distinct_nonzero_bins(c1) <= 1 or _distinct_nonzero_bins(c2) <= 1:
        return TwoSampleStats(np.nan, np.nan, n1, n2)

    pooled = c1 + c2
    grand_median = histogram_median(pooled, bins)
    above1, below1 = _mood_counts_against_median(c1, bins, grand_median)
    above2, below2 = _mood_counts_against_median(c2, bins, grand_median)
    table = np.array([[above1, above2], [below1, below2]], dtype=float)
    try:
        stat, p, _, _ = stats.chi2_contingency(table, correction=True)
    except ValueError:
        return TwoSampleStats(np.nan, np.nan, n1, n2)
    return TwoSampleStats(float(stat), float(p), n1, n2)


def histogram_total(counts: Sequence[float]) -> int:
    """Total number of observations represented by a histogram."""
    return int(_nonnegative_int_counts(counts).sum())


def density_from_hist(counts: Sequence[float], bin_width: float) -> np.ndarray:
    """Convert histogram counts to probability density values."""
    c = _nonnegative_int_counts(counts).astype(float)
    total = c.sum()
    if total == 0:
        return np.zeros_like(c, dtype=float)
    return c / (total * bin_width)

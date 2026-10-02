"""CHIP candidate criteria, shared by the pipeline and the report.

Kept free of Hail so `chip_report.py` can import it without a Spark context.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# Medians are differences of histogram bin floors, which are not exactly
# representable: 0.45 - 0.40 evaluates to 0.04999999999999998, so a genuine
# one-bin allele-balance shift would fail a bare `>= 0.05`.
_TOL = 1e-9


@dataclass(frozen=True)
class CandidateCriteria:
    """Thresholds defining a CHIP candidate.

    At v4 sample sizes every p-value in a well-powered transcript is
    effectively zero, so significance alone cannot rank genes: transcripts with
    identical pLoF and synonymous median allele balance clear a 7e-7 threshold
    on a KS statistic of 0.035. These criteria add the direction and
    effect-size requirements that the v2 write-up described but did not encode,
    and replace the v2 age *ratio* cutoff with an absolute shift.

    The ratio is not scale-free on a 5-year bin grid. A one-bin shift of 50->55
    gives exactly 1.100 and passes, while the identical one-bin shift of
    55->60 gives 1.0909 and fails. Because the modal synonymous median age in
    v4 is 55 rather than v2's 50, that cutoff excluded ASXL1 and DNMT3A.
    """

    min_age_diff: float = 5.0
    min_ks_stat_ab: float = 0.15
    min_ab_median_shift: float = 0.05
    require_ab_direction: bool = True
    bonferroni_scope: str = "global"
    # Set to 1.1 to reproduce the v2 criterion exactly; 0 disables.
    min_age_ratio: float = 0.0

    # Columns the mask reads, so callers can validate inputs up front.
    REQUIRED_COLUMNS = (
        "ks_p_ab",
        "mood_p_ab",
        "ks_p_age",
        "mood_p_age",
        "bonferroni_threshold",
        "ks_stat_ab",
        "med_ab_case",
        "med_ab_syn",
        "med_age_case",
        "med_age_syn",
    )


def _age_diff(df: pd.DataFrame) -> pd.Series:
    if "age_diff_case_vs_syn" in df.columns:
        return df["age_diff_case_vs_syn"]
    return df["med_age_case"] - df["med_age_syn"]


def _ab_shift(df: pd.DataFrame) -> pd.Series:
    if "ab_median_shift" in df.columns:
        return df["ab_median_shift"]
    return df["med_ab_syn"] - df["med_ab_case"]


def candidate_mask(df: pd.DataFrame, criteria: CandidateCriteria) -> pd.Series:
    """Boolean mask of rows meeting every CHIP criterion.

    NaN comparisons evaluate False, so untestable transcripts are excluded.
    """
    thr = df["bonferroni_threshold"]
    significant = (
        (df["ks_p_ab"] < thr)
        & (df["mood_p_ab"] < thr)
        & (df["ks_p_age"] < thr)
        & (df["mood_p_age"] < thr)
    )
    mask = (
        significant
        & (_age_diff(df) >= criteria.min_age_diff - _TOL)
        & (_ab_shift(df) >= criteria.min_ab_median_shift - _TOL)
        & (df["ks_stat_ab"] >= criteria.min_ks_stat_ab - _TOL)
    )
    if criteria.min_age_ratio > 0:
        ratio = (
            df["age_ratio_case_vs_syn"]
            if "age_ratio_case_vs_syn" in df.columns
            else df["med_age_case"] / df["med_age_syn"]
        )
        mask &= ratio >= criteria.min_age_ratio - _TOL
    if criteria.require_ab_direction:
        # Two-sided tests admit transcripts whose pLoF allele balance is
        # *higher* than synonymous, which is the opposite of the CHIP
        # signature. Redundant when min_ab_median_shift > 0, but the shift
        # threshold can be set to 0 to recover the v2 comparison.
        mask &= df["med_ab_case"] < df["med_ab_syn"]
    return mask.fillna(False).astype(bool)

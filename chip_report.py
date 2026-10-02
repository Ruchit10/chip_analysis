#!/usr/bin/env python3
"""Generate CHIP summary markdown and Supplementary-Figure-6-style plots."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chip_criteria import CandidateCriteria, candidate_mask
from chip_stats import density_from_hist


LOGGER = logging.getLogger("chip_report")
LITERATURE_CHIP_GENES = {
    "DNMT3A",
    "TET2",
    "ASXL1",
    "PPM1D",
    "TP53",
    "JAK2",
    "SF3B1",
    "SRSF2",
    "U2AF1",
    "IDH1",
    "IDH2",
    "CBL",
    "GNB1",
    "GNAS",
    "ZRSR2",
    "CREBBP",
    "SETD2",
    "KRAS",
    "NRAS",
    "RUNX1",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_array_column(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def candidate_filter(df: pd.DataFrame, criteria: CandidateCriteria) -> pd.Series:
    """Recompute the candidate mask so report thresholds are explicit."""
    return candidate_mask(df, criteria)


def add_closeness_metrics(df: pd.DataFrame, criteria: CandidateCriteria) -> pd.DataFrame:
    """Add helper metrics for ranking rows close to CHIP criteria."""
    out = df.copy()
    pmax = out[["ks_p_ab", "mood_p_ab", "ks_p_age", "mood_p_age"]].max(axis=1)
    out["p_over_threshold_max"] = pmax / out["bonferroni_threshold"]
    if "age_diff_case_vs_syn" not in out.columns:
        out["age_diff_case_vs_syn"] = out["med_age_case"] - out["med_age_syn"]
    if "ab_median_shift" not in out.columns:
        out["ab_median_shift"] = out["med_ab_syn"] - out["med_ab_case"]

    # Shortfall on each non-p criterion, normalized to its own threshold so the
    # components are comparable; 0 means the criterion is met.
    out["age_gap"] = (criteria.min_age_diff - out["age_diff_case_vs_syn"]).clip(lower=0)
    out["ab_shift_gap"] = (criteria.min_ab_median_shift - out["ab_median_shift"]).clip(lower=0)
    out["ks_stat_gap"] = (criteria.min_ks_stat_ab - out["ks_stat_ab"]).clip(lower=0)

    # Diagnostic for paralog/segdup artifacts: a gene whose signal comes from
    # one or two ultra-common "pLoF" calls rather than many independent ones.
    out["carriers_per_case_variant"] = (
        out["n_ab_case"] / out["variant_count_case"].replace(0, np.nan)
    ).round(1)

    out["p_component"] = out["p_over_threshold_max"]
    out["age_component"] = out["age_gap"] / max(criteria.min_age_diff, 1e-9)
    out["ab_component"] = out["ab_shift_gap"] / max(criteria.min_ab_median_shift, 1e-9)
    out["effect_component"] = out["ks_stat_gap"] / max(criteria.min_ks_stat_ab, 1e-9)
    # Lower is better; rows meeting every non-p criterion rank on p alone.
    out["distance_to_pass"] = (
        out["p_component"]
        + out["age_component"]
        + out["ab_component"]
        + out["effect_component"]
    )
    return out


def best_row_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    """Pick one representative row per gene for compact test summaries."""
    ranked = df.sort_values(
        by=["is_chip_candidate", "distance_to_pass", "ks_stat_ab"],
        ascending=[False, True, False],
    )
    return ranked.groupby("gene_symbol", as_index=False).head(1).reset_index(drop=True)


def failed_near_miss_rows(
    df: pd.DataFrame,
    *,
    p_threshold: float,
    age_diff_min: float,
    include_literature: bool = True,
) -> pd.DataFrame:
    """
    Select one near-miss failed row per gene for full-run summaries.

    Selection requires:
    - failed CHIP criteria
    - CHIP-like directionality in medians (AB lower in case, age higher in case)
    - p_over_threshold_max <= p_threshold
    - age_diff_case_vs_syn >= age_diff_min
    """
    per_gene = best_row_per_gene(df)
    failed = per_gene[~per_gene["is_chip_candidate"]].copy()
    directional = (failed["med_ab_case"] < failed["med_ab_syn"]) & (
        failed["med_age_case"] > failed["med_age_syn"]
    )
    failed = failed[
        directional
        & (failed["p_over_threshold_max"] <= p_threshold)
        & (failed["age_diff_case_vs_syn"] >= age_diff_min)
    ].copy()
    if include_literature:
        # Ensure literature genes are retained if close, even if thresholds are slightly missed.
        lit_failed = per_gene[
            (~per_gene["is_chip_candidate"]) & per_gene["is_literature_chip_gene"]
        ].copy()
        failed = pd.concat([failed, lit_failed], ignore_index=True).drop_duplicates(
            subset=["gene_symbol"], keep="first"
        )
    failed = failed.sort_values(
        by=["distance_to_pass", "p_component", "age_component"], ascending=[True, True, True]
    )
    return failed.reset_index(drop=True)


def _bin_width(floors: np.ndarray) -> float:
    if len(floors) < 2:
        return 1.0
    return float(floors[1] - floors[0])


def _sanitize_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)
    return out.strip("_")


def plot_grid(row: pd.Series, plots_dir: Path) -> Path:
    gene = row["gene_symbol"]
    tx = row["transcript_id"]
    sev = row["comparison_severity"]
    arm = row["ab_arm"]
    ab_floors = parse_array_column(row["ab_bin_floors"])
    age_floors = parse_array_column(row["age_bin_floors"])
    ab_case = parse_array_column(row["ab_hist_case"])
    ab_syn = parse_array_column(row["ab_hist_syn"])
    age_case = parse_array_column(row["age_hist_case"])
    age_syn = parse_array_column(row["age_hist_syn"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=False)
    pass_label = "PASS" if bool(row.get("is_chip_candidate", False)) else "FAIL"
    closeness = float(row.get("distance_to_pass", np.nan))
    p_component = float(row.get("p_component", np.nan))
    age_component = float(row.get("age_component", np.nan))
    fig.suptitle(f"{gene} ({tx}) [{arm}, {sev}]", y=0.98, fontsize=12)
    fig.text(
        0.5,
        0.94,
        (
            f"CHIP criteria: {pass_label} | Closeness: {closeness:.6g} "
            f"(p_component={p_component:.6g}, age_component={age_component:.6g})"
        ),
        ha="center",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f2f2f2", "alpha": 0.8, "edgecolor": "none"},
    )

    colors = "#555555"
    ab_w = _bin_width(ab_floors)
    age_w = _bin_width(age_floors)

    axes[0, 0].bar(ab_floors, ab_case, width=ab_w, align="edge", color=colors)
    axes[0, 1].bar(age_floors, age_case, width=age_w, align="edge", color=colors)
    axes[1, 0].bar(ab_floors, ab_syn, width=ab_w, align="edge", color=colors)
    axes[1, 1].bar(age_floors, age_syn, width=age_w, align="edge", color=colors)

    axes[0, 0].set_ylabel(f"{sev} Count")
    axes[1, 0].set_ylabel("synonymous Count")
    axes[1, 0].set_xlabel("Bin floor (AB)")
    axes[1, 1].set_xlabel("Bin floor (age)")

    axes[0, 0].set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes[1, 0].set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axes[0, 1].set_xticks([20, 40, 60, 80])
    axes[1, 1].set_xticks([20, 40, 60, 80])
    axes[0, 0].set_xlim(ab_floors.min(), ab_floors.max() + ab_w)
    axes[1, 0].set_xlim(ab_floors.min(), ab_floors.max() + ab_w)
    axes[0, 1].set_xlim(age_floors.min(), age_floors.max() + age_w)
    axes[1, 1].set_xlim(age_floors.min(), age_floors.max() + age_w)

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    name = _sanitize_name(f"{gene}_{tx}_{arm}_{sev}_grid.png")
    out_path = plots_dir / name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_overlay(row: pd.Series, plots_dir: Path) -> Path:
    gene = row["gene_symbol"]
    tx = row["transcript_id"]
    sev = row["comparison_severity"]
    arm = row["ab_arm"]
    ab_floors = parse_array_column(row["ab_bin_floors"])
    age_floors = parse_array_column(row["age_bin_floors"])
    ab_case = parse_array_column(row["ab_hist_case"])
    ab_syn = parse_array_column(row["ab_hist_syn"])
    age_case = parse_array_column(row["age_hist_case"])
    age_syn = parse_array_column(row["age_hist_syn"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=False)
    pass_label = "PASS" if bool(row.get("is_chip_candidate", False)) else "FAIL"
    closeness = float(row.get("distance_to_pass", np.nan))
    p_component = float(row.get("p_component", np.nan))
    age_component = float(row.get("age_component", np.nan))
    fig.suptitle(f"{gene} ({tx}) density overlay [{arm}, {sev}]", y=0.98, fontsize=12)
    fig.text(
        0.5,
        0.93,
        (
            f"CHIP criteria: {pass_label} | Closeness: {closeness:.6g} "
            f"(p_component={p_component:.6g}, age_component={age_component:.6g})"
        ),
        ha="center",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f2f2f2", "alpha": 0.8, "edgecolor": "none"},
    )

    ab_w = _bin_width(ab_floors)
    age_w = _bin_width(age_floors)
    d_case_ab = density_from_hist(ab_case, ab_w)
    d_syn_ab = density_from_hist(ab_syn, ab_w)
    d_case_age = density_from_hist(age_case, age_w)
    d_syn_age = density_from_hist(age_syn, age_w)

    axes[0].plot(ab_floors, d_case_ab, drawstyle="steps-post", label=sev)
    axes[0].plot(ab_floors, d_syn_ab, drawstyle="steps-post", label="synonymous")
    axes[0].set_xlabel("Bin floor (AB)")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(ab_floors.min(), ab_floors.max() + ab_w)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(age_floors, d_case_age, drawstyle="steps-post", label=sev)
    axes[1].plot(age_floors, d_syn_age, drawstyle="steps-post", label="synonymous")
    axes[1].set_xlabel("Bin floor (age)")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(age_floors.min(), age_floors.max() + age_w)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))

    name = _sanitize_name(f"{gene}_{tx}_{arm}_{sev}_overlay.png")
    out_path = plots_dir / name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_markdown(
    out_md: Path,
    summary_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    failed_near_miss_df: pd.DataFrame,
    plot_rows: List[Tuple[pd.Series, Path, Path]],
    *,
    criteria: CandidateCriteria,
    test_summary: bool = False,
    test_top_n: int = 50,
) -> None:
    lines: List[str] = []
    lines.append("# CHIP analysis summary\n")
    lines.append(f"- Total comparisons: **{len(summary_df)}**")
    lines.append(
        f"- CHIP candidate rows: **{len(candidate_df)}** "
        f"across **{candidate_df['gene_symbol'].nunique()}** genes"
    )
    lines.append(f"- Failed near-miss genes (full-run filter): **{len(failed_near_miss_df)}**")
    lines.append(
        f"- Literature CHIP genes among candidates: "
        f"**{candidate_df['gene_symbol'][candidate_df['is_literature_chip_gene']].nunique()}"
        f" / {len(LITERATURE_CHIP_GENES)}**"
    )
    lines.append("")
    lines.append("## Criteria\n")
    lines.append(
        f"- Bonferroni threshold (`{criteria.bonferroni_scope}` scope): "
        f"{summary_df['bonferroni_threshold'].iloc[0]:.3g}, required for all four "
        f"p-values (allele-balance and age, KS and Mood's)"
    )
    lines.append(f"- Median age excess >= {criteria.min_age_diff:g} years")
    lines.append(f"- Median allele-balance decrease >= {criteria.min_ab_median_shift:g}")
    lines.append(f"- Allele-balance KS statistic >= {criteria.min_ks_stat_ab:g}")
    lines.append(
        f"- Median allele balance lower in case variants: "
        f"{'required' if criteria.require_ab_direction else 'not required'}"
    )
    lines.append("")

    if test_summary:
        lines.append("## Test Run Overview (All Tested Genes)\n")
        gene_view = best_row_per_gene(summary_df)
        gene_view = gene_view.sort_values(
            by=["is_chip_candidate", "distance_to_pass", "age_ratio_case_vs_syn"],
            ascending=[False, True, False],
        )
        if test_top_n > 0:
            gene_view = gene_view.head(test_top_n)
        cols = [
            "gene_symbol",
            "transcript_id",
            "comparison_severity",
            "ab_arm",
            "is_chip_candidate",
            "ks_p_ab",
            "mood_p_ab",
            "ks_p_age",
            "mood_p_age",
            "bonferroni_threshold",
            "ks_stat_ab",
            "ab_median_shift",
            "age_diff_case_vs_syn",
            "p_over_threshold_max",
            "p_component",
            "age_component",
            "ab_component",
            "effect_component",
            "distance_to_pass",
        ]
        lines.append(
            "One best row per gene ranked by closeness to passing; "
            "`is_chip_candidate=True` means it passes all CHIP criteria."
        )
        lines.append("")
        lines.append(gene_view[cols].to_markdown(index=False))
        lines.append("")

    if len(candidate_df) > 0:
        lines.append("## PASS Results\n")
        top = candidate_df.sort_values(
            by=["comparison_severity", "ab_arm", "ks_stat_ab", "age_diff_case_vs_syn"],
            ascending=[True, True, False, False],
        )
        lines.append("### Candidate Table\n")
        cols = [
            "gene_symbol",
            "comparison_severity",
            "ab_arm",
            "ks_stat_ab",
            "ks_p_ab",
            "mood_p_ab",
            "ks_p_age",
            "mood_p_age",
            "med_ab_case",
            "med_ab_syn",
            "ab_median_shift",
            "med_age_case",
            "med_age_syn",
            "age_diff_case_vs_syn",
            "carriers_per_case_variant",
            "bonferroni_threshold",
            "is_literature_chip_gene",
        ]
        lines.append(top[cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("## PASS Results\n")
        lines.append("### Candidate Table\n")
        lines.append("No genes passed the CHIP criteria with the provided inputs.\n")

    if len(failed_near_miss_df) > 0:
        lines.append("## FAIL Results\n")
        lines.append("### Failed Near-Miss Genes\n")
        lines.append(
            "One row per gene (best row), selected using CHIP-like directionality and "
            "near-miss thresholds for full-run triage."
        )
        lines.append("")
        nm_cols = [
            "gene_symbol",
            "transcript_id",
            "comparison_severity",
            "ab_arm",
            "ks_stat_ab",
            "ks_p_ab",
            "mood_p_ab",
            "ks_p_age",
            "mood_p_age",
            "bonferroni_threshold",
            "med_ab_case",
            "med_ab_syn",
            "ab_median_shift",
            "med_age_case",
            "med_age_syn",
            "age_diff_case_vs_syn",
            "p_over_threshold_max",
            "p_component",
            "age_component",
            "ab_component",
            "effect_component",
            "distance_to_pass",
            "is_literature_chip_gene",
        ]
        lines.append(failed_near_miss_df[nm_cols].to_markdown(index=False))
        lines.append("")

    if len(plot_rows) > 0:
        pass_plots = [x for x in plot_rows if bool(x[0]["is_chip_candidate"])]
        fail_plots = [x for x in plot_rows if not bool(x[0]["is_chip_candidate"])]

        if len(pass_plots) > 0:
            lines.append("## PASS Plots\n")
        for row, grid_path, overlay_path in pass_plots:
            lines.append(
                f"### {row['gene_symbol']} ({row['transcript_id']}) - {row['comparison_severity']} [{row['ab_arm']}]"
            )
            lines.append(f"- AB median case/syn: {row['med_ab_case']:.3f} / {row['med_ab_syn']:.3f}")
            lines.append(f"- Age median case/syn: {row['med_age_case']:.1f} / {row['med_age_syn']:.1f}")
            lines.append(
                f"- CHIP criteria pass: {bool(row['is_chip_candidate'])} | "
                f"Closeness score: {row['distance_to_pass']:.6g} "
                f"(p_component={row['p_component']:.6g}, age_component={row['age_component']:.6g})"
            )
            lines.append(f"- KS p (AB/Age): {row['ks_p_ab']:.3e} / {row['ks_p_age']:.3e}")
            lines.append(f"- Mood p (AB/Age): {row['mood_p_ab']:.3e} / {row['mood_p_age']:.3e}")
            lines.append("")
            lines.append(f"![supp-grid]({grid_path.as_posix()})")
            lines.append("")
            lines.append(f"![overlay]({overlay_path.as_posix()})")
            lines.append("")

        if len(fail_plots) > 0:
            lines.append("## FAIL Plots\n")
        for row, grid_path, overlay_path in fail_plots:
            lines.append(
                f"### {row['gene_symbol']} ({row['transcript_id']}) - {row['comparison_severity']} [{row['ab_arm']}]"
            )
            lines.append(f"- AB median case/syn: {row['med_ab_case']:.3f} / {row['med_ab_syn']:.3f}")
            lines.append(f"- Age median case/syn: {row['med_age_case']:.1f} / {row['med_age_syn']:.1f}")
            lines.append(
                f"- CHIP criteria pass: {bool(row['is_chip_candidate'])} | "
                f"Closeness score: {row['distance_to_pass']:.6g} "
                f"(p_component={row['p_component']:.6g}, age_component={row['age_component']:.6g})"
            )
            lines.append(f"- KS p (AB/Age): {row['ks_p_ab']:.3e} / {row['ks_p_age']:.3e}")
            lines.append(f"- Mood p (AB/Age): {row['mood_p_ab']:.3e} / {row['mood_p_age']:.3e}")
            lines.append("")
            lines.append(f"![supp-grid]({grid_path.as_posix()})")
            lines.append("")
            lines.append(f"![overlay]({overlay_path.as_posix()})")
            lines.append("")

    out_md.write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    compression = "gzip" if args.input.endswith(".gz") else "infer"
    df = pd.read_csv(args.input, sep="\t", compression=compression)
    if len(df) == 0:
        raise ValueError("Input TSV is empty.")

    if args.primary_arm_only:
        df = df[(df["ab_arm"] == "adj") & (df["comparison_severity"] == "lof")].copy()
    if args.arm != "all":
        df = df[df["ab_arm"] == args.arm].copy()
    if args.severity != "all":
        df = df[df["comparison_severity"] == args.severity].copy()
    if len(df) == 0:
        raise ValueError("No rows left after arm/severity filtering.")

    if args.bonferroni_scope == "arm" and "bonferroni_threshold_arm" in df.columns:
        df["bonferroni_threshold"] = df["bonferroni_threshold_arm"]
    elif args.bonferroni_scope == "global" and "bonferroni_threshold_global" in df.columns:
        df["bonferroni_threshold"] = df["bonferroni_threshold_global"]

    criteria = CandidateCriteria(
        min_age_diff=args.min_age_diff,
        min_ks_stat_ab=args.min_ks_stat_ab,
        min_ab_median_shift=args.min_ab_median_shift,
        require_ab_direction=not args.no_ab_direction,
        bonferroni_scope=args.bonferroni_scope,
        min_age_ratio=args.min_age_ratio,
    )
    LOGGER.info("Applying criteria: %s", criteria)

    df = add_closeness_metrics(df, criteria)
    df["is_chip_candidate"] = candidate_filter(df, criteria)
    df["is_literature_chip_gene"] = df["gene_symbol"].isin(LITERATURE_CHIP_GENES)
    candidate_df = df[df["is_chip_candidate"]].copy()
    if args.require_literature:
        candidate_df = candidate_df[candidate_df["is_literature_chip_gene"]]
    failed_nm_df = pd.DataFrame(columns=df.columns)
    if args.failed_near_miss_summary:
        failed_nm_df = failed_near_miss_rows(
            df,
            p_threshold=args.failed_p_threshold,
            age_diff_min=args.failed_age_diff_min,
            include_literature=args.failed_include_literature,
        )
        if args.failed_max_plots > 0:
            failed_nm_df = failed_nm_df.head(args.failed_max_plots)

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    pass_dir = plots_dir / "pass"
    fail_dir = plots_dir / "fail"
    pass_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    plot_rows: List[Tuple[pd.Series, Path, Path]] = []
    if args.test_summary:
        # In test-summary mode, plot one best row per tested gene so near-miss genes are visible.
        to_plot = best_row_per_gene(df).sort_values(
            by=["is_chip_candidate", "distance_to_pass", "age_ratio_case_vs_syn"],
            ascending=[False, True, False],
        )
        if args.test_summary_top_n > 0:
            to_plot = to_plot.head(args.test_summary_top_n)
    else:
        # Default (full-run-safe): only plot candidates.
        to_plot = candidate_df
        if args.plot_top_n:
            to_plot = to_plot.head(args.plot_top_n)
        if args.failed_near_miss_summary and len(failed_nm_df) > 0:
            to_plot = pd.concat([to_plot, failed_nm_df], ignore_index=True).drop_duplicates(
                subset=["gene_symbol", "transcript_id", "comparison_severity", "ab_arm"],
                keep="first",
            )
    for _, row in to_plot.iterrows():
        target_dir = pass_dir if bool(row["is_chip_candidate"]) else fail_dir
        grid = plot_grid(row, target_dir)
        overlay = plot_overlay(row, target_dir)
        plot_rows.append((row, grid, overlay))

    out_md = Path(args.output_markdown)
    make_markdown(
        out_md,
        df,
        candidate_df,
        failed_nm_df,
        plot_rows,
        criteria=criteria,
        test_summary=args.test_summary,
        test_top_n=args.test_summary_top_n,
    )
    LOGGER.info(
        "Wrote report %s with %d plots.",
        out_md,
        len(plot_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CHIP markdown report and plots.")
    parser.add_argument("--input", required=True, help="Input TSV from chip_analysis.py")
    parser.add_argument(
        "--output-markdown",
        default="CHIP_v4_summary.md",
        help="Output markdown path.",
    )
    parser.add_argument("--plots-dir", default="plots", help="Output directory for plot PNGs.")
    parser.add_argument(
        "--arm",
        default="all",
        choices=["all", "adj", "raw"],
        help="Filter to one AB arm.",
    )
    parser.add_argument(
        "--severity",
        default="all",
        choices=["all", "lof", "missense_high", "missense_sens"],
        help="Filter to one comparison severity.",
    )
    parser.add_argument(
        "--plot-top-n",
        type=int,
        default=0,
        help="If >0, only plot first N candidate rows after filtering.",
    )
    parser.add_argument(
        "--require-literature",
        action="store_true",
        help="Only include candidates in the literature CHIP panel.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--test-summary",
        action="store_true",
        help=(
            "Test-run mode: include a compact all-genes overview table with pass/fail "
            "and closeness metrics. Default behavior remains candidate-only."
        ),
    )
    parser.add_argument(
        "--test-summary-top-n",
        type=int,
        default=50,
        help="Maximum number of genes in the test-summary table (0 = no limit).",
    )
    parser.add_argument(
        "--failed-near-miss-summary",
        action="store_true",
        help=(
            "Full-run mode helper: add a failed near-miss gene table and optional failed-gene "
            "plots using thresholds so output stays bounded."
        ),
    )
    parser.add_argument(
        "--failed-p-threshold",
        type=float,
        default=5.0,
        help="Near-miss filter: keep failed rows with p_over_threshold_max <= this value.",
    )
    parser.add_argument(
        "--failed-age-diff-min",
        type=float,
        default=2.5,
        help=(
            "Near-miss filter: keep failed rows with age_diff_case_vs_syn >= this "
            "many years."
        ),
    )
    parser.add_argument(
        "--primary-arm-only",
        action="store_true",
        help="Restrict to the primary arm (adj allele balance, pLoF vs synonymous).",
    )
    parser.add_argument(
        "--min-age-diff",
        type=float,
        default=5.0,
        help="Minimum median-age excess in years for case vs synonymous carriers.",
    )
    parser.add_argument(
        "--min-age-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional v2-style median-age ratio floor (0 disables). Set to 1.1 "
            "to reproduce the v2 criterion exactly; prefer --min-age-diff, "
            "since the ratio is not scale-free on a 5-year bin grid."
        ),
    )
    parser.add_argument(
        "--min-ks-stat-ab",
        type=float,
        default=0.15,
        help="Minimum allele-balance KS statistic (effect size) for a candidate.",
    )
    parser.add_argument(
        "--min-ab-median-shift",
        type=float,
        default=0.05,
        help="Minimum decrease in median allele balance (case vs synonymous).",
    )
    parser.add_argument(
        "--no-ab-direction",
        action="store_true",
        help="Do not require median allele balance to be lower in case variants.",
    )
    parser.add_argument(
        "--bonferroni-scope",
        default="global",
        choices=["global", "arm"],
        help="Use the across-arm (global) or within-arm threshold column.",
    )
    parser.add_argument(
        "--failed-max-plots",
        type=int,
        default=20,
        help="Maximum number of failed near-miss rows to plot when enabled.",
    )
    parser.add_argument(
        "--failed-include-literature",
        action="store_true",
        help="Always include literature CHIP failed genes in near-miss summary selection.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    run(args)

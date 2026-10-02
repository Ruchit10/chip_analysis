#!/usr/bin/env python3
"""Port of v2 CHIP analysis to gnomAD v4 release tables."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from io import TextIOWrapper
from typing import List, Optional, Sequence, Tuple

import hail as hl
import numpy as np
import pandas as pd

from chip_criteria import CandidateCriteria, candidate_mask
from chip_stats import (
    TwoSampleStats,
    histogram_median,
    histogram_total,
    ks_2samp_from_hist,
    mood_median_test_from_hist,
)
from gnomad.resources.grch38.gnomad import public_release
from gnomad.resources.grch38.reference_data import ensembl_interval


LOGGER = logging.getLogger("chip_v4")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _is_gcs_path(path: str) -> bool:
    return path.startswith("gs://")


def _serialize_array(values: Sequence[float]) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def parse_gene_list(test_genes: Optional[str]) -> List[str]:
    if not test_genes:
        return ["ASXL1", "DNMT3A", "TET2", "JAK2", "PPM1D", "TP53", "SF3B1", "SRSF2"]
    return [x.strip() for x in test_genes.split(",") if x.strip()]


def get_release_ht_path(release: str) -> str:
    return f"gs://gcp-public-data--gnomad/release/{release}/ht/exomes/gnomad.exomes.v{release}.sites.ht"


def load_exomes_ht(release: str) -> hl.Table:
    resource = public_release("exomes")
    if release in resource.versions:
        LOGGER.info("Loading release from gnomad resource map: %s", release)
        return resource.versions[release].ht()
    release_path = get_release_ht_path(release)
    LOGGER.info(
        "Release %s not in installed gnomad package map; reading explicit path: %s",
        release,
        release_path,
    )
    return hl.read_table(release_path)


def resolve_test_intervals(genes: Sequence[str], ensembl_version: str = "105") -> List[hl.Interval]:
    interval_ht = ensembl_interval.versions[ensembl_version].ht()
    gene_set = hl.literal(set(genes))
    filtered = interval_ht.filter(gene_set.contains(interval_ht.gene_name))
    intervals = filtered.aggregate(hl.agg.collect_as_set(filtered.interval))
    return sorted(intervals, key=lambda iv: (iv.start.contig, iv.start.position, iv.end.position))


def get_adj_freq_index(ht: hl.Table) -> int:
    """Index of the callset-wide adj frequency entry in `ht.freq`."""
    try:
        idx = hl.eval(ht.index_globals().freq_index_dict.get("adj"))
    except Exception:  # resource schema varies across releases
        idx = None
    if idx is None:
        LOGGER.warning("freq_index_dict['adj'] unavailable; defaulting to freq[0].")
        return 0
    return int(idx)


def prune_release_fields(ht: hl.Table, adj_freq_index: int = 0) -> hl.Table:
    tc = ht.vep.transcript_consequences.map(
        lambda x: hl.struct(
            transcript_id=x.transcript_id,
            gene_id=x.gene_id,
            gene_symbol=x.gene_symbol,
            biotype=x.biotype,
            canonical=x.canonical,
            mane_select=x.mane_select,
            lof=x.lof,
            consequence_terms=x.consequence_terms,
        )
    )
    return ht.select(
        filters=ht.filters,
        AF=ht.freq[adj_freq_index].AF,
        histograms=ht.histograms,
        in_silico_predictors=ht.in_silico_predictors,
        transcript_consequences=tc,
    )


def downscale_hist(
    bin_freq: hl.expr.ArrayNumericExpression, cap: int
) -> hl.expr.ArrayNumericExpression:
    """Proportionally shrink a histogram so its total is at most `cap`.

    The tests treat every (variant, carrier) pair as an independent
    observation, so a single ultra-common call can contribute hundreds of
    thousands of pseudo-observations and drive the p-value on its own.
    Downscaling preserves each variant's distribution shape while bounding its
    contribution to the pooled sample size.
    """
    total = hl.sum(bin_freq)
    scale = hl.if_else(total > cap, hl.float64(cap) / hl.float64(total), 1.0)
    # Hail has no round(); floor(x + 0.5) is equivalent for non-negative counts.
    return bin_freq.map(lambda c: hl.int64(hl.floor(hl.float64(c) * scale + 0.5)))


def get_hist_bin_floors(ht: hl.Table) -> Tuple[List[float], List[float]]:
    ab_edges = ht.aggregate(hl.agg.take(ht.histograms.qual_hists.ab_hist_alt.bin_edges, 1))[0]
    age_edges = ht.aggregate(hl.agg.take(ht.histograms.age_hists.age_hist_het.bin_edges, 1))[0]
    ab_floors = [float(x) for x in ab_edges[:-1]]
    age_floors = [25.0] + [float(x) for x in age_edges]
    return ab_floors, age_floors


def filter_and_prepare_variants(
    ht: hl.Table,
    revel_threshold: float,
    revel_sensitivity_threshold: float,
    include_missense: bool,
    max_af: float = 1.0,
    max_carriers_per_variant: int = 0,
    ensembl_only: bool = True,
) -> hl.Table:
    ht = ht.filter(hl.len(ht.filters) == 0)

    if max_af < 1.0:
        # Common germline pLoF calls (often paralog/segdup mismapping) have
        # skewed allele balance for reasons unrelated to clonal expansion, and
        # at v4 sample sizes each one is individually well powered.
        LOGGER.info("Dropping variants with adj AF > %s.", max_af)
        ht = ht.filter(hl.or_else(ht.AF, 0.0) <= max_af)

    def _has_target_csq(tc: hl.StructExpression) -> hl.BooleanExpression:
        is_lof = hl.is_defined(tc.lof) & (tc.lof == "HC")
        is_syn = tc.consequence_terms.contains("synonymous_variant")
        is_missense = tc.consequence_terms.contains("missense_variant")
        is_coding = tc.biotype == "protein_coding"
        has_transcript = (
            (hl.is_defined(tc.mane_select) & (tc.mane_select != ""))
            | hl.or_else(tc.canonical == 1, False)
        )
        keep = is_coding & has_transcript & (is_lof | is_syn | is_missense)
        if ensembl_only:
            # gnomAD v4 is annotated with a merged VEP cache, so every gene
            # appears twice: once as (ENSG..., ENST...) and once as
            # (Entrez id, NM_...). Keeping both tests each gene twice with
            # identical statistics and doubles the reported candidate count.
            keep &= tc.gene_id.startswith("ENSG")
        return keep

    ht = ht.annotate(
        transcript_consequences=ht.transcript_consequences.filter(_has_target_csq)
    )
    ht = ht.filter(hl.len(ht.transcript_consequences) > 0)

    # Drop rows with no observed carriers in histograms.
    has_any_ab = (
        (hl.sum(ht.histograms.qual_hists.ab_hist_alt.bin_freq) > 0)
        | (hl.sum(ht.histograms.raw_qual_hists.ab_hist_alt.bin_freq) > 0)
    )
    age_het = ht.histograms.age_hists.age_hist_het
    age_hom = ht.histograms.age_hists.age_hist_hom
    age_total = (
        hl.sum(age_het.bin_freq)
        + age_het.n_smaller
        + age_het.n_larger
        + hl.sum(age_hom.bin_freq)
        + age_hom.n_smaller
        + age_hom.n_larger
    )
    ht = ht.filter(has_any_ab & (age_total > 0))

    ht = ht.explode(ht.transcript_consequences)
    tc = ht.transcript_consequences
    is_lof = hl.is_defined(tc.lof) & (tc.lof == "HC")
    is_syn = tc.consequence_terms.contains("synonymous_variant")
    is_missense = tc.consequence_terms.contains("missense_variant")
    revel = hl.or_else(ht.in_silico_predictors.revel_max, 0.0)

    severity = (
        hl.case()
        .when(is_lof, "lof")
        .when(is_syn, "synonymous")
        .when(include_missense & is_missense & (revel >= revel_threshold), "missense_high")
        .when(
            include_missense
            & is_missense
            & (revel >= revel_sensitivity_threshold),
            "missense_sens",
        )
        .default("other")
    )

    age_het = ht.histograms.age_hists.age_hist_het
    age_hom = ht.histograms.age_hists.age_hist_hom
    age_middle = hl.zip(age_het.bin_freq, age_hom.bin_freq).map(lambda x: x[0] + x[1])
    age_hist = hl.array([age_het.n_smaller + age_hom.n_smaller]).extend(age_middle).append(
        age_het.n_larger + age_hom.n_larger
    )

    ab_adj_hist = ht.histograms.qual_hists.ab_hist_alt.bin_freq
    ab_raw_hist = ht.histograms.raw_qual_hists.ab_hist_alt.bin_freq
    if max_carriers_per_variant > 0:
        LOGGER.info(
            "Capping per-variant histogram totals at %d carriers.", max_carriers_per_variant
        )
        ab_adj_hist = downscale_hist(ab_adj_hist, max_carriers_per_variant)
        ab_raw_hist = downscale_hist(ab_raw_hist, max_carriers_per_variant)
        age_hist = downscale_hist(age_hist, max_carriers_per_variant)

    ht = ht.annotate(
        severity=severity,
        is_mane=hl.is_defined(tc.mane_select) & (tc.mane_select != ""),
        is_canonical=hl.or_else(tc.canonical == 1, False),
        ab_adj_hist=ab_adj_hist,
        ab_raw_hist=ab_raw_hist,
        age_hist=age_hist,
        transcript_id=tc.transcript_id,
        gene_id=tc.gene_id,
        gene_symbol=tc.gene_symbol,
    )
    ht = ht.filter(ht.severity != "other")
    return ht


def aggregate_histograms(ht: hl.Table, checkpoint_path: Optional[str], overwrite: bool) -> hl.Table:
    agg_ht = ht.group_by(
        ht.gene_id,
        ht.gene_symbol,
        ht.transcript_id,
        ht.is_mane,
        ht.is_canonical,
        ht.severity,
    ).aggregate(
        ab_adj_hist=hl.agg.array_sum(ht.ab_adj_hist),
        ab_raw_hist=hl.agg.array_sum(ht.ab_raw_hist),
        age_hist=hl.agg.array_sum(ht.age_hist),
        variant_count=hl.agg.count(),
    )
    if checkpoint_path:
        LOGGER.info("Checkpointing aggregated table to %s", checkpoint_path)
        return agg_ht.checkpoint(checkpoint_path, overwrite=overwrite)
    return agg_ht


def choose_transcript_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    tx = df[["gene_id", "transcript_id", "is_mane", "is_canonical"]].drop_duplicates().copy()
    carrier = (
        df.groupby(["gene_id", "transcript_id"])["ab_raw_hist"]
        .apply(lambda s: float(np.max([np.sum(np.asarray(x, dtype=float)) for x in s])))
        .rename("carrier_score")
        .reset_index()
    )
    tx = tx.merge(carrier, on=["gene_id", "transcript_id"], how="left")
    selected_rows = []
    for _, group in tx.groupby("gene_id"):
        mane = group[group["is_mane"]]
        if len(mane) > 0:
            candidates = mane
            source = "mane_select"
        else:
            canonical = group[group["is_canonical"]]
            candidates = canonical if len(canonical) > 0 else group
            source = "canonical" if len(canonical) > 0 else "fallback"
        selected = candidates.sort_values(
            by=["carrier_score", "transcript_id"], ascending=[False, True]
        ).iloc[0]
        selected_rows.append(
            {
                "gene_id": selected["gene_id"],
                "transcript_id": selected["transcript_id"],
                "transcript_source": source,
            }
        )
    selected_df = pd.DataFrame(selected_rows)
    out = df.merge(selected_df, on=["gene_id", "transcript_id"], how="inner")
    return out


def _safe_ratio(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return float(a / b)


def _global_syn_hist(df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    syn = df[df["severity"] == "synonymous"][col]
    if len(syn) == 0:
        return None
    return np.median(np.stack(syn.apply(lambda x: np.asarray(x, dtype=float))), axis=0)


def compute_statistics(
    df: pd.DataFrame,
    ab_bin_floors: Sequence[float],
    age_bin_floors: Sequence[float],
    criteria: Optional[CandidateCriteria] = None,
) -> pd.DataFrame:
    criteria = criteria or CandidateCriteria()
    syn_df = df[df["severity"] == "synonymous"].set_index(["gene_id", "transcript_id"])
    if len(syn_df) == 0:
        raise ValueError("No synonymous rows found after filtering; cannot compute comparisons.")

    global_ab_adj = _global_syn_hist(df, "ab_adj_hist")
    global_ab_raw = _global_syn_hist(df, "ab_raw_hist")
    global_age = _global_syn_hist(df, "age_hist")

    rows = []
    for _, row in df.iterrows():
        if row["severity"] == "synonymous":
            continue
        key = (row["gene_id"], row["transcript_id"])
        if key not in syn_df.index:
            continue
        syn_row = syn_df.loc[key]
        if isinstance(syn_row, pd.DataFrame):
            syn_row = syn_row.iloc[0]

        for arm in ("adj", "raw"):
            case_ab = row[f"ab_{arm}_hist"]
            syn_ab = syn_row[f"ab_{arm}_hist"]
            ks_ab = ks_2samp_from_hist(case_ab, syn_ab)
            mood_ab = mood_median_test_from_hist(case_ab, syn_ab, ab_bin_floors)
            med_ab_case = histogram_median(case_ab, ab_bin_floors)
            med_ab_syn = histogram_median(syn_ab, ab_bin_floors)

            case_age = row["age_hist"]
            syn_age = syn_row["age_hist"]
            ks_age = ks_2samp_from_hist(case_age, syn_age)
            mood_age = mood_median_test_from_hist(case_age, syn_age, age_bin_floors)
            med_age_case = histogram_median(case_age, age_bin_floors)
            med_age_syn = histogram_median(syn_age, age_bin_floors)

            global_ab = global_ab_adj if arm == "adj" else global_ab_raw
            if global_ab is not None:
                ks_ab_global = ks_2samp_from_hist(case_ab, global_ab)
            else:
                ks_ab_global = TwoSampleStats(np.nan, np.nan, 0, 0)
            if global_age is not None:
                ks_age_global = ks_2samp_from_hist(case_age, global_age)
            else:
                ks_age_global = TwoSampleStats(np.nan, np.nan, 0, 0)

            rows.append(
                {
                    "gene_symbol": row["gene_symbol"],
                    "gene_id": row["gene_id"],
                    "transcript_id": row["transcript_id"],
                    "transcript_source": row["transcript_source"],
                    "comparison_severity": row["severity"],
                    "ab_arm": arm,
                    "variant_count_case": int(row["variant_count"]),
                    "variant_count_synonymous": int(syn_row["variant_count"]),
                    "ks_stat_ab": ks_ab.statistic,
                    "ks_p_ab": ks_ab.p_value,
                    "mood_stat_ab": mood_ab.statistic,
                    "mood_p_ab": mood_ab.p_value,
                    "ks_stat_age": ks_age.statistic,
                    "ks_p_age": ks_age.p_value,
                    "mood_stat_age": mood_age.statistic,
                    "mood_p_age": mood_age.p_value,
                    "ks_stat_ab_global_syn": ks_ab_global.statistic,
                    "ks_p_ab_global_syn": ks_ab_global.p_value,
                    "ks_stat_age_global_syn": ks_age_global.statistic,
                    "ks_p_age_global_syn": ks_age_global.p_value,
                    "n_ab_case": histogram_total(case_ab),
                    "n_ab_syn": histogram_total(syn_ab),
                    "n_age_case": histogram_total(case_age),
                    "n_age_syn": histogram_total(syn_age),
                    "med_ab_case": med_ab_case,
                    "med_ab_syn": med_ab_syn,
                    "med_age_case": med_age_case,
                    "med_age_syn": med_age_syn,
                    "age_ratio_case_vs_syn": _safe_ratio(med_age_case, med_age_syn),
                    "age_diff_case_vs_syn": med_age_case - med_age_syn,
                    "ab_median_shift": med_ab_syn - med_ab_case,
                    "ab_hist_case": _serialize_array(case_ab),
                    "ab_hist_syn": _serialize_array(syn_ab),
                    "age_hist_case": _serialize_array(case_age),
                    "age_hist_syn": _serialize_array(syn_age),
                }
            )

    result_df = pd.DataFrame(rows)
    if len(result_df) == 0:
        return result_df

    tested_pairs = result_df[
        ["ab_arm", "comparison_severity", "gene_id", "transcript_id"]
    ].drop_duplicates()
    n_tested_map = tested_pairs.groupby(["ab_arm", "comparison_severity"]).size().to_dict()
    n_tests_total = len(tested_pairs)

    result_df["n_transcripts_tested"] = [
        int(n_tested_map.get(k, 0))
        for k in zip(result_df["ab_arm"], result_df["comparison_severity"])
    ]
    result_df["n_tests_total"] = n_tests_total
    result_df["bonferroni_threshold_arm"] = result_df["n_transcripts_tested"].apply(
        lambda n: (0.05 / (2 * n)) if n > 0 else np.nan
    )
    # The pipeline runs six correlated test families ({adj, raw} x
    # {lof, missense_high, missense_sens}); an arm-local Bonferroni does not
    # account for that, so the global scope is the default.
    result_df["bonferroni_threshold_global"] = (
        (0.05 / (2 * n_tests_total)) if n_tests_total > 0 else np.nan
    )
    result_df["bonferroni_threshold"] = (
        result_df["bonferroni_threshold_arm"]
        if criteria.bonferroni_scope == "arm"
        else result_df["bonferroni_threshold_global"]
    )
    result_df["is_primary_arm"] = (result_df["ab_arm"] == "adj") & (
        result_df["comparison_severity"] == "lof"
    )
    result_df["is_chip_candidate"] = candidate_mask(result_df, criteria)
    result_df["ab_bin_floors"] = _serialize_array(ab_bin_floors)
    result_df["age_bin_floors"] = _serialize_array(age_bin_floors)
    return result_df


def write_dataframe(df: pd.DataFrame, output: str) -> None:
    LOGGER.info("Writing output: %s", output)
    is_gzip = output.endswith(".gz")
    if _is_gcs_path(output):
        if is_gzip:
            with hl.hadoop_open(output, "wb") as raw_out:
                with gzip.GzipFile(fileobj=raw_out, mode="wb") as gz_out:
                    with TextIOWrapper(gz_out, encoding="utf-8") as out:
                        df.to_csv(out, sep="\t", index=False)
        else:
            with hl.hadoop_open(output, "w") as out:
                df.to_csv(out, sep="\t", index=False)
    else:
        if is_gzip:
            with gzip.open(output, "wt", encoding="utf-8") as out:
                df.to_csv(out, sep="\t", index=False)
        else:
            df.to_csv(output, sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    if args.spark_local_ip:
        # Useful on macOS where Spark can fail to bind hostname.
        import os

        os.environ["SPARK_LOCAL_IP"] = args.spark_local_ip

    hl.init(log=args.hail_log, default_reference="GRCh38")
    LOGGER.info("Hail initialized.")

    criteria = CandidateCriteria(
        min_age_diff=args.min_age_diff,
        min_ks_stat_ab=args.min_ks_stat_ab,
        min_ab_median_shift=args.min_ab_median_shift,
        require_ab_direction=not args.no_ab_direction,
        bonferroni_scope=args.bonferroni_scope,
        min_age_ratio=args.min_age_ratio,
    )
    LOGGER.info("Candidate criteria: %s", criteria)

    ht = load_exomes_ht(args.release)
    if args.test:
        genes = parse_gene_list(args.test_genes)
        intervals = resolve_test_intervals(genes, args.ensembl_version)
        if len(intervals) == 0:
            raise ValueError(f"No intervals found for test genes: {genes}")
        LOGGER.info("Applying test interval filter for %d genes / %d intervals.", len(genes), len(intervals))
        ht = hl.filter_intervals(ht, intervals)

    ht = prune_release_fields(ht, get_adj_freq_index(ht))
    ab_floors, age_floors = get_hist_bin_floors(ht)
    ht = filter_and_prepare_variants(
        ht,
        revel_threshold=args.revel_threshold,
        revel_sensitivity_threshold=args.revel_sensitivity_threshold,
        include_missense=not args.no_missense,
        max_af=args.max_af,
        max_carriers_per_variant=args.max_carriers_per_variant,
        ensembl_only=not args.keep_refseq,
    )
    agg_ht = aggregate_histograms(ht, args.checkpoint_path, args.overwrite)

    LOGGER.info("Converting aggregated table to pandas.")
    agg_df = agg_ht.key_by().to_pandas()
    if len(agg_df) == 0:
        raise ValueError("No rows after aggregation.")
    agg_df = choose_transcript_per_gene(agg_df)
    result_df = compute_statistics(agg_df, ab_floors, age_floors, criteria)
    if len(result_df) == 0:
        raise ValueError("No result rows produced after comparisons.")
    write_dataframe(result_df, args.output)
    n_pass = int(result_df["is_chip_candidate"].sum())
    n_genes = result_df.loc[result_df["is_chip_candidate"], "gene_symbol"].nunique()
    LOGGER.info(
        "Done. Wrote %d rows; %d candidate rows across %d genes.",
        len(result_df),
        n_pass,
        n_genes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CHIP analysis for gnomAD v4 from release histograms."
    )
    parser.add_argument("--output", required=True, help="Output TSV path (local or gs://).")
    parser.add_argument(
        "--release",
        default="4.1.1",
        help="gnomAD exome release to read (default: 4.1.1).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path for aggregated HT (local or gs://).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite checkpoint path if it exists.",
    )
    parser.add_argument(
        "--revel-threshold",
        type=float,
        default=0.7,
        help="REVEL threshold for missense_high category.",
    )
    parser.add_argument(
        "--revel-sensitivity-threshold",
        type=float,
        default=0.5,
        help="Sensitivity REVEL threshold for missense_sens category.",
    )
    parser.add_argument(
        "--no-missense",
        action="store_true",
        help="Disable missense arms and run only LoF vs synonymous.",
    )
    parser.add_argument(
        "--keep-refseq",
        action="store_true",
        help=(
            "Keep RefSeq transcript consequences from the merged VEP cache. "
            "Off by default: keeping them tests every gene twice (once under "
            "its ENSG/ENST ids, once under its Entrez/NM_ ids)."
        ),
    )
    parser.add_argument(
        "--max-af",
        type=float,
        default=0.01,
        help=(
            "Drop variants with adj AF above this value (1.0 disables). Common "
            "germline pLoF calls dominate the pooled histograms and have skewed "
            "allele balance for reasons unrelated to clonal expansion."
        ),
    )
    parser.add_argument(
        "--max-carriers-per-variant",
        type=int,
        default=1000,
        help=(
            "Proportionally downscale any single variant's histograms so they "
            "contribute at most this many pseudo-observations (0 disables)."
        ),
    )
    parser.add_argument(
        "--min-age-diff",
        type=float,
        default=5.0,
        help=(
            "Minimum median-age excess in years for case vs synonymous carriers. "
            "Replaces the v2 age-ratio cutoff, which was degenerate on the "
            "5-year bin grid."
        ),
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
        help=(
            "Minimum decrease in median allele balance (case vs synonymous), in "
            "AB units. Default is one histogram bin."
        ),
    )
    parser.add_argument(
        "--no-ab-direction",
        action="store_true",
        help=(
            "Do not require median allele balance to be lower in case variants. "
            "Only meaningful with --min-ab-median-shift 0."
        ),
    )
    parser.add_argument(
        "--bonferroni-scope",
        default="global",
        choices=["global", "arm"],
        help=(
            "Correct across all arms/severities (global, default) or within each "
            "arm/severity independently (arm, the v2-style behavior)."
        ),
    )
    parser.add_argument("--test", action="store_true", help="Run only selected gene intervals.")
    parser.add_argument(
        "--test-genes",
        default=None,
        help="Comma-separated symbols for --test mode.",
    )
    parser.add_argument(
        "--ensembl-version",
        default="105",
        help="ensembl_interval resource version for test interval resolution.",
    )
    parser.add_argument(
        "--hail-log",
        default="/tmp/chip_analysis_hail.log",
        help="Hail log path.",
    )
    parser.add_argument(
        "--spark-local-ip",
        default=None,
        help="Set SPARK_LOCAL_IP before hl.init (e.g. 127.0.0.1).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    run(args)

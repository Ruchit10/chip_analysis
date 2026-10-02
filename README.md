# CHIP v4 port

This directory contains a gnomAD v4 port of the v2 `chip_analysis.py` workflow
for clonal hematopoiesis (CHIP) signal detection from release histograms.

## Files

- `chip_analysis.py`: Hail pipeline on gnomAD release HT data.
- `chip_stats.py`: closed-form histogram-space statistics (no expansion).
- `chip_criteria.py`: candidate thresholds, shared by the pipeline and report.
- `chip_report.py`: Supplementary-Figure-6-style plots + markdown summary.
- `test_chip_stats.py`: unit tests proving closed-form/scipy equivalence.
- `test_chip_criteria.py`: unit tests for the candidate criteria.

## Criteria: differences from v2

A direct port of the v2 criteria flags ~65 genes per arm at v4 scale, versus
v2's 3, and drops two of v2's three genes. The defaults below correct that.
Each has a flag to restore the v2 behavior for comparison.

| Change | Flag to revert |
|---|---|
| Drop RefSeq consequences from the merged VEP cache. v4 annotates both Ensembl and RefSeq, so every gene is otherwise tested twice — once as `(ENSG…, ENST…)`, once as `(Entrez id, NM_…)` — with identical statistics. | `--keep-refseq` |
| Age criterion is an absolute shift (>= 5 years), not a ratio >= 1.1. The ratio is not scale-free on a 5-year grid: 50->55 is exactly 1.100 and passes, the identical 55->60 shift is 1.0909 and fails. v4's modal synonymous median age is 55 (v2's was 50), so the ratio cutoff excluded ASXL1 and DNMT3A. | `--min-age-diff -1000 --min-age-ratio 1.1` |
| Require median allele balance to be *lower* in case variants, and by at least one histogram bin. Two-sided tests otherwise admit transcripts whose pLoF allele balance is higher — the opposite of the CHIP signature, and over half of the uncorrected pass list. | `--min-ab-median-shift -1000 --no-ab-direction` |
| Require an allele-balance KS statistic >= 0.15. At v4 sample sizes, transcripts with *identical* pLoF and synonymous median allele balance clear a 7e-7 threshold on a KS statistic of 0.035, so significance alone cannot rank genes. | `--min-ks-stat-ab 0` |
| Drop variants with adj AF > 0.01 and cap each variant's histogram contribution at 1000 carriers. The tests treat every (variant, carrier) pair as independent, so one ultra-common call in a paralog/segdup region can supply >500k pseudo-observations and drive the p-value alone. | `--max-af 1.0 --max-carriers-per-variant 0` |
| Bonferroni across all six arms (`{adj, raw}` x `{lof, missense_high, missense_sens}`) rather than within each. | `--bonferroni-scope arm` |

The primary arm is `adj` allele balance, pLoF vs synonymous — the only one v2
ran. `chip_report.py --primary-arm-only` restricts to it, and the output
carries an `is_primary_arm` column.

`chip_report.py` re-applies the criteria from its own flags, so thresholds can
be re-cut from an existing TSV without re-running the pipeline. Keep them in
sync with the pipeline flags if you change either.

## Environment

Target conda env: `hail_137`.

Install gnomAD methods from GitHub (release map in PyPI `gnomad` may lag):

```bash
conda run -n hail_137 pip install --upgrade \
  "gnomad @ git+https://github.com/broadinstitute/gnomad_methods.git"
```

Quick sanity check:

```bash
conda run -n hail_137 python -c "import hail, gnomad; print(hail.version())"
```

On macOS, local Spark may need:

```bash
export SPARK_LOCAL_IP=127.0.0.1
```

## Local tests

```bash
cd /Users/rpanchal/Desktop/Constraint/chip_analysis
conda run -n hail_137 pytest -q test_chip_stats.py test_chip_criteria.py
```

## Test-mode run (known CHIP genes only)

This run uses `hl.filter_intervals` on `ASXL1,DNMT3A,TET2,JAK2,PPM1D,TP53,SF3B1,SRSF2`.

```bash
cd /Users/rpanchal/Desktop/Constraint/chip_analysis
SPARK_LOCAL_IP=127.0.0.1 conda run -n hail_137 python chip_analysis.py \
  --test \
  --release 4.1.1 \
  --output chip_test_results.tsv \
  --checkpoint-path /tmp/chip_test_agg.ht \
  --overwrite
```

Generate report and plots:

```bash
conda run -n hail_137 python chip_report.py \
  --input chip_test_results.tsv \
  --output-markdown CHIP_v4_summary.md \
  --plots-dir plots
```

## Dataproc run (full dataset)

Start cluster (example sizing; tune for your budget/runtime target):

```bash
hailctl dataproc start chip-v4 --region us-central1 \
  --num-workers 10 \
  --num-secondary-workers 100 \
  --worker-machine-type n1-highmem-8 \
  --max-idle 30m \
  --packages "gnomad@git+https://github.com/broadinstitute/gnomad_methods.git"
```

Submit CHIP analysis:

```bash
hailctl dataproc submit chip-v4 \
  /Users/rpanchal/Desktop/Constraint/chip_analysis/chip_analysis.py \
  --pyfiles /Users/rpanchal/Desktop/Constraint/chip_analysis/chip_stats.py,/Users/rpanchal/Desktop/Constraint/chip_analysis/chip_criteria.py \
  -- \
  --release 4.1.1 \
  --output gs://<bucket>/chip/v4/chip_results.tsv \
  --checkpoint-path gs://<bucket>/chip/v4/chip_agg.ht \
  --overwrite
```

Generate plots/report locally after downloading the TSV:

```bash
conda run -n hail_137 python chip_report.py \
  --input chip_results.tsv \
  --output-markdown CHIP_v4_summary.md \
  --plots-dir plots \
  --failed-near-miss-summary --failed-include-literature
```

To reproduce the v2 criteria from the same TSV for comparison:

```bash
conda run -n hail_137 python chip_report.py \
  --input chip_results.tsv \
  --output-markdown CHIP_v2_criteria.md \
  --plots-dir plots_v2_criteria \
  --primary-arm-only --bonferroni-scope arm \
  --min-age-diff -1000 --min-age-ratio 1.1 --min-ks-stat-ab 0 \
  --min-ab-median-shift -1000 --no-ab-direction
```

Note that this reverts only the post-hoc criteria. The AF filter, carrier cap,
and RefSeq exclusion happen in the Hail step and need a re-run to change.
The thresholds use `-1000` rather than `0` because `0` still requires the
shift to be non-negative, which is itself a constraint v2 did not apply.

## Expected runtime/cost

These are starting estimates (highly dependent on preemption rates, disk I/O,
and other workloads in region):

- `--test` mode: minutes, typically <$5.
- Full exomes run with the sizing above: ~1-3 hours.
- Approximate compute spend with heavy preemptible use: ~$75-$300.

Treat these numbers as planning guidance, not guarantees.

#!/usr/bin/env python3
"""Recompute the paired Table 6 statistics from published per-image AP50."""

import csv
from pathlib import Path

import numpy as np
from scipy import stats


def main():
    path = Path(__file__).resolve().parents[1] / "results/statistics/per_image_ap50.tsv"
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="	"))
    deltas = np.array([float(row["paired_delta"]) for row in rows if row["valid_pair"] == "YES"])
    test = stats.ttest_1samp(deltas, 0.0)
    sem = stats.sem(deltas)
    ci = stats.t.interval(0.95, len(deltas) - 1, loc=deltas.mean(), scale=sem)
    wilcoxon = stats.wilcoxon(deltas)
    print(f"n={len(deltas)}")
    print(f"mean_delta={deltas.mean():.15f}")
    print(f"t={test.statistic:.15f}; p={test.pvalue:.15f}")
    print(f"ci95=[{ci[0]:.15f}, {ci[1]:.15f}]")
    print(f"cohen_dz={deltas.mean() / deltas.std(ddof=1):.15f}")
    print(f"wilcoxon_w={wilcoxon.statistic:.15f}; p={wilcoxon.pvalue:.15f}")


if __name__ == "__main__":
    main()

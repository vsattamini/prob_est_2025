"""
Statistical tests and effect size calculations for the stylometric dataset.

This module provides functions to perform non‑parametric hypothesis tests
and compute effect sizes on features extracted from the corpus.  The
primary test used is the Mann–Whitney U test (equivalent to the
Wilcoxon rank‑sum test for independent samples).  Effect sizes are
reported using Cliff’s δ, which measures the degree of overlap between
two distributions.  P‑values can be adjusted for multiple comparisons
using the Benjamini–Hochberg false discovery rate (FDR).

The command line interface reads a CSV containing features and a label
column, performs the tests for all numeric columns and writes a summary
table to disk.  The summary includes medians by group, p‑values,
adjusted p‑values and Cliff’s δ.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu  # type: ignore
except ImportError:
    mannwhitneyu = None  # type: ignore

__all__ = [
    "mann_whitney_u",
    "cliffs_delta",
    "fdr_bh",
    "run_tests",
]


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> float:
    """Compute two‑sided Mann–Whitney U test p‑value.

    Falls back to a simple permutation approach if SciPy is not available.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays of observations for the two groups.

    Returns
    -------
    float
        Two‑sided p‑value.
    """
    if mannwhitneyu is not None:
        # use SciPy implementation
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(p)
    # fallback: compute U statistic manually and approximate p by permutation
    # Note: this is slower but avoids external dependencies.
    n1 = len(x)
    n2 = len(y)
    all_values = np.concatenate([x, y])
    ranks = pd.Series(all_values).rank(method="average").values
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    # compute two‑sided p by random permutation (1000 permutations)
    obs = u1
    combined = np.concatenate([x, y])
    greater = 0
    n_perms = 1000
    for _ in range(n_perms):
        np.random.shuffle(combined)
        r = pd.Series(combined).rank(method="average").values
        r1_perm = r[:n1].sum()
        u1_perm = r1_perm - n1 * (n1 + 1) / 2
        if u1_perm >= obs:
            greater += 1
    p = (greater + 1) / (n_perms + 1)
    # two‑sided: double p but cap at 1
    return min(1.0, 2 * p)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff’s δ effect size.

    δ is the probability that a randomly selected element from x is
    greater than a randomly selected element from y, minus the reverse.
    Values lie in [‑1, 1], where 0 indicates complete overlap, positive
    values favour x and negative values favour y.
    """
    n1 = len(x)
    n2 = len(y)
    # broadcast comparisons: x[:, None] > y[None, :] yields a matrix of
    # booleans.  Similarly for <.  Count the number of True values.
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / float(n1 * n2)


def fdr_bh(p_values: List[float]) -> List[float]:
    """Benjamini–Hochberg procedure to control the false discovery rate.

    Given a list of p‑values, returns a list of adjusted q‑values in the
    same order.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    q = np.empty(m, dtype=float)
    min_coeff = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        coeff = m / rank * sorted_p[i]
        min_coeff = min(min_coeff, coeff)
        q[i] = min_coeff
    # reorder to original order and cap at 1
    q_values = np.minimum(1.0, q[np.argsort(sorted_indices)])
    return q_values.tolist()


def run_tests(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Perform Mann–Whitney tests and compute effect sizes for numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numeric feature columns and a label column with two
        unique values (e.g. 'human' and 'llm').
    label_col : str, default 'label'
        Name of the column indicating group membership.

    Returns
    -------
    pandas.DataFrame
        Summary table with medians by group, p‑value, q‑value and Cliff’s δ
        for each feature.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame")
    labels = df[label_col].unique()
    if len(labels) != 2:
        raise ValueError("Exactly two groups are required for Mann–Whitney test")
    group1, group2 = labels
    results = []
    numeric_cols = [c for c in df.columns if c not in [label_col, "topic"] and pd.api.types.is_numeric_dtype(df[c])]
    p_values: List[float] = []
    for col in numeric_cols:
        x = df[df[label_col] == group1][col].dropna().values.astype(float)
        y = df[df[label_col] == group2][col].dropna().values.astype(float)
        p = mann_whitney_u(x, y)
        delta = cliffs_delta(x, y)
        median1 = float(np.median(x)) if len(x) > 0 else float("nan")
        median2 = float(np.median(y)) if len(y) > 0 else float("nan")
        results.append({
            "feature": col,
            f"median_{group1}": median1,
            f"median_{group2}": median2,
            "p_value": p,
            "delta": delta,
        })
        p_values.append(p)
    # adjust p‑values using FDR
    q_values = fdr_bh(p_values)
    for res, q in zip(results, q_values):
        res["q_value"] = q
    return pd.DataFrame(results)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Mann–Whitney tests for all numeric features.")
    parser.add_argument("--features", required=True, help="Path to CSV with extracted features and labels.")
    parser.add_argument("--label-col", default="label", help="Name of the label column.")
    parser.add_argument("--out", default="results_tests.csv", help="Path to save the summary table.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    df = pd.read_csv(args.features)
    summary = run_tests(df, args.label_col)
    summary.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
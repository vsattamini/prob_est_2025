"""
Evaluation utilities for the fuzzy classifier.

This module provides functions to assess the performance of the
``FuzzyClassifier`` under cross‑validation.  It mirrors the
``evaluate_classifiers`` function from ``models.py`` but uses the
fuzzy inference system instead of LDA or logistic regression.  The
evaluation returns ROC and precision–recall curves for each fold so
that users can plot or summarise them later.

The command‑line interface allows invocation from the shell:

```
python -m src.evaluate_fuzzy --features data/processed/features.csv --label-col label --topic-col topic --pos-label human --neg-label llm --n-splits 5 --roc-out fuzzy_roc.pkl --pr-out fuzzy_pr.pkl
```

"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold

from .fuzzy import FuzzyClassifier

__all__ = ["evaluate_fuzzy"]


def evaluate_fuzzy(
    df: pd.DataFrame,
    label_col: str = "label",
    topic_col: Optional[str] = "topic",
    pos_label: Optional[str] = None,
    neg_label: Optional[str] = None,
    n_splits: int = 5,
) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
    """Cross‑validate the ``FuzzyClassifier`` and collect performance curves.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature dataset with numeric features, a label column, and optionally
        a topic column.
    label_col : str, default 'label'
        Name of the column containing class labels.
    topic_col : str or None, default 'topic'
        If provided and present in ``df``, group splits by this column using
        ``GroupKFold`` to avoid topic leakage.  If None or missing, use
        ``StratifiedKFold``.
    pos_label : str or None
        The value in ``label_col`` that should be considered the positive class.
        If None, the second unique label alphabetically is taken as positive.
    neg_label : str or None
        The value in ``label_col`` that should be considered the negative class.
        If None, the first unique label alphabetically is taken as negative.
    n_splits : int, default 5
        Number of cross‑validation folds.

    Returns
    -------
    (roc_results, pr_results) : Tuple[dict, dict]
        Dictionaries containing lists of fold metrics for ROC and PR curves.
        Each entry has keys ``'fpr'``, ``'tpr'``, ``'auc'`` or
        ``'precision'``, ``'recall'``, ``'ap'`` respectively.
    """
    # Determine labels and group information
    labels = sorted(df[label_col].astype(str).unique())
    if len(labels) != 2:
        raise ValueError("Binary classification expected for evaluate_fuzzy")
    # Assign positive and negative labels if not provided
    if pos_label is None or neg_label is None:
        # assign alphabetically: second label is positive
        neg_label_default, pos_label_default = labels[0], labels[1]
        pos_label = pos_label or pos_label_default
        neg_label = neg_label or neg_label_default
    # Cross‑validation splits
    if topic_col and topic_col in df.columns:
        groups = df[topic_col].values
        cv = GroupKFold(n_splits=n_splits)
        splits = cv.split(df, df[label_col], groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = cv.split(df, df[label_col])
    # Prepare results containers
    roc_results: Dict[str, List[dict]] = {"Fuzzy": []}
    pr_results: Dict[str, List[dict]] = {"Fuzzy": []}
    # For each fold, fit and evaluate
    for train_idx, test_idx in splits:
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        # Fit fuzzy classifier
        clf = FuzzyClassifier(pos_label=pos_label, neg_label=neg_label)
        clf.fit(train_df, label_col=label_col, topic_col=topic_col or "topic")
        # Compute probabilities for test set: probability of positive class is at index 1
        probs = clf.predict_proba(test_df)[:, 1]
        # Binarise true labels: 1 if positive
        y_true = (test_df[label_col] == pos_label).astype(int).values
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        roc_results["Fuzzy"].append({"fpr": fpr, "tpr": tpr, "auc": roc_auc})
        pr_results["Fuzzy"].append({"precision": precision, "recall": recall, "ap": ap})
    return roc_results, pr_results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate fuzzy classifier via cross‑validation.")
    parser.add_argument("--features", required=True, help="Path to CSV file with features.")
    parser.add_argument("--label-col", default="label", help="Name of the label column.")
    parser.add_argument(
        "--topic-col",
        default="topic",
        help="Name of the topic column used for group splits (optional).",
    )
    parser.add_argument(
        "--pos-label",
        default=None,
        help="Value to treat as positive class (default: second unique label).",
    )
    parser.add_argument(
        "--neg-label",
        default=None,
        help="Value to treat as negative class (default: first unique label).",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument("--roc-out", default="fuzzy_roc.pkl", help="Output path for ROC curves (pickle).")
    parser.add_argument("--pr-out", default="fuzzy_pr.pkl", help="Output path for PR curves (pickle).")
    args = parser.parse_args(argv)
    # Load data
    df = pd.read_csv(args.features)
    # Evaluate fuzzy classifier
    roc_results, pr_results = evaluate_fuzzy(
        df,
        label_col=args.label_col,
        topic_col=args.topic_col or None,
        pos_label=args.pos_label,
        neg_label=args.neg_label,
        n_splits=args.n_splits,
    )
    # Persist results
    import pickle
    with open(args.roc_out, "wb") as f:
        pickle.dump(roc_results, f)
    with open(args.pr_out, "wb") as f:
        pickle.dump(pr_results, f)


if __name__ == "__main__":
    main()
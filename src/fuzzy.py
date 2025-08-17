"""
Fuzzy membership functions and inference for stylometric classification.

This module defines a simple fuzzy classifier that assigns degrees of
membership to the categories 'human' and 'llm' based on stylometric
features.  Membership functions are constructed from quantiles of the
training data, and rules are derived automatically from the direction
of group medians.  The inference engine aggregates degrees using
averaging and selects the class with the highest aggregated score.

The design is intentionally transparent: instead of a complex fuzzy
system with dozens of hand‑crafted rules, this implementation learns
orientation and thresholds directly from the data.  It is intended as
a pedagogical example rather than a production system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

__all__ = ["FuzzyClassifier", "triangular_membership"]


def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function.

    Parameters
    ----------
    x : float
        Input value.
    a : float
        Left foot of the triangle (membership rises from 0 to 1).
    b : float
        Peak of the triangle (membership is 1).
    c : float
        Right foot of the triangle (membership falls back to 0).

    Returns
    -------
    float
        Membership degree in [0, 1].
    """
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    if b <= x < c:
        return (c - x) / (c - b) if c != b else 0.0
    return 0.0


@dataclass
class MembershipFunction:
    """A triangular membership function with optional orientation.

    The orientation indicates whether larger values correspond to
    increased membership of the positive class ('direct') or
    decreased membership ('inverse').  See FuzzyClassifier.fit().
    """
    a: float
    b: float
    c: float
    orientation: str = "direct"

    def compute(self, x: float) -> float:
        """Compute membership degree for a value."""
        return triangular_membership(x, self.a, self.b, self.c)


@dataclass
class FuzzyClassifier:
    """A simple fuzzy inference classifier for binary problems.

    This classifier learns triangular membership functions for each
    feature based on the 33rd and 66th percentiles of the training
    data.  It uses the direction of group medians to decide whether
    high values favour the positive class ('direct') or low values
    ('inverse').

    Parameters
    ----------
    pos_label : str
        The label considered the positive class (e.g. 'human').
    neg_label : str
        The label considered the negative class (e.g. 'llm').
    feature_columns : Optional[List[str]]
        Names of features to use.  If None, all numeric columns other
        than the label and topic columns will be used.
    """
    pos_label: str
    neg_label: str
    feature_columns: Optional[List[str]] = None
    memberships_: Dict[str, Dict[str, MembershipFunction]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, label_col: str = "label", topic_col: str = "topic") -> "FuzzyClassifier":
        """Fit membership functions using training data.

        The method computes quantiles (33% and 66%) for each feature
        and defines triangular membership functions with vertices at
        (min, q33, q66) for the 'low' set, (q33, median, q66) for
        'medium' and (q33, q66, max) for 'high'.  It then determines
        orientation: if the median for the positive class is greater
        than that for the negative class, high membership indicates
        belonging to the positive class (direct orientation).  Otherwise
        the orientation is inverse.
        """
        # Determine which columns to use
        if self.feature_columns is None:
            numeric_cols = [c for c in df.columns if c not in [label_col, topic_col] and pd.api.types.is_numeric_dtype(df[c])]
        else:
            numeric_cols = list(self.feature_columns)
        # Compute per‑feature quantiles and membership functions
        self.memberships_.clear()
        for col in numeric_cols:
            series = df[col].astype(float)
            q0 = series.min()
            q33 = series.quantile(0.33)
            q50 = series.quantile(0.50)
            q66 = series.quantile(0.66)
            q100 = series.max()
            # Median by group for orientation
            med_pos = df[df[label_col] == self.pos_label][col].median()
            med_neg = df[df[label_col] == self.neg_label][col].median()
            orientation = "direct" if med_pos >= med_neg else "inverse"
            self.memberships_[col] = {
                "low": MembershipFunction(q0, q0, q33, orientation),
                "medium": MembershipFunction(q33, q50, q66, orientation),
                "high": MembershipFunction(q33, q66, q100, orientation),
            }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Compute class probabilities for new samples.

        Returns an array of shape (n_samples, 2) with probabilities
        for (neg_label, pos_label).
        """
        if not self.memberships_:
            raise RuntimeError("FuzzyClassifier has not been fitted yet")
        # Determine features in order
        features = list(self.memberships_.keys())
        scores_pos: List[float] = []
        scores_neg: List[float] = []
        for _, row in X[features].iterrows():
            pos_vals: List[float] = []
            neg_vals: List[float] = []
            for col in features:
                # For each membership set (low, medium, high), compute degree
                mems = self.memberships_[col]
                val = row[col]
                low = mems["low"].compute(val)
                med = mems["medium"].compute(val)
                high = mems["high"].compute(val)
                # Determine orientation: high values favour pos if orientation='direct'
                if mems["low"].orientation == "direct":
                    # High value membership supports positive class; low supports negative
                    pos_vals.append(high)
                    neg_vals.append(low)
                else:
                    # High supports negative; low supports positive
                    pos_vals.append(low)
                    neg_vals.append(high)
                # Medium membership contributes equally to both
                # Add half of medium to both classes
                pos_vals[-1] += 0.5 * med
                neg_vals[-1] += 0.5 * med
            # Aggregate degrees by averaging
            pos_score = float(np.mean(pos_vals)) if pos_vals else 0.0
            neg_score = float(np.mean(neg_vals)) if neg_vals else 0.0
            # Normalise to sum to 1
            total = pos_score + neg_score
            if total == 0:
                scores_pos.append(0.5)
                scores_neg.append(0.5)
            else:
                scores_pos.append(pos_score / total)
                scores_neg.append(neg_score / total)
        return np.vstack([scores_neg, scores_pos]).T

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predict class labels for new samples."""
        probs = self.predict_proba(X)
        # Choose the class with the higher probability
        preds = [self.neg_label if p[0] >= p[1] else self.pos_label for p in probs]
        return preds

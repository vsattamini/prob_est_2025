# Guide 6: Fuzzy Classification

**Module**: [src/fuzzy.py](../src/fuzzy.py)
**Previous**: [Guide 5: Multivariate Models](05_multivariate_models.md)
**Next**: Guide 7: Results Interpretation

---

## Overview

Classical classifiers (LDA, logistic regression) use **crisp boundaries**:
- Input: feature vector → Output: "human" or "llm" (binary decision)

**Fuzzy logic** allows **gradual transitions**:
- Input: feature vector → Output: 0.7 "human", 0.3 "llm" (degrees of membership)

This guide explains our **data-driven fuzzy classifier** that:
1. Learns fuzzy membership functions from training data
2. Automatically derives rules based on feature medians
3. Provides interpretable classification with transparency

**Pedagogical focus**: This is a teaching example, not a production system. Real-world fuzzy systems use expert knowledge and complex rule bases. Ours learns everything from data.

---

## Why Fuzzy Logic for Text Classification?

### The Problem with Hard Boundaries

Logistic regression might learn:
```
If func_word_ratio > 0.38, predict "human"
```

**But what if**:
- `func_word_ratio = 0.379` → "llm" (just barely)
- `func_word_ratio = 0.381` → "human" (just barely)

**Fuzzy approach**:
```
func_word_ratio = 0.37 → 0.4 "human", 0.6 "llm"
func_word_ratio = 0.38 → 0.5 "human", 0.5 "llm"
func_word_ratio = 0.39 → 0.6 "human", 0.4 "llm"
```

Gradual transition captures uncertainty!

### Interpretability

Fuzzy rules like:
```
IF sent_burst is HIGH AND func_word_ratio is HIGH
THEN text is HUMAN
```

are more **human-readable** than:
```
logit = 0.82×func_word_ratio + 0.65×sent_burst - 3.21
P(human) = 1 / (1 + e^(-logit))
```

---

## Fuzzy Set Theory Basics

### Membership Functions

A **membership function** μ(x) maps a value x to [0, 1]:

```
μ_HIGH(sent_burst) = {
  0.0  if sent_burst ≤ 0.33
  linear from 0.33 to 0.66
  1.0  if sent_burst ≥ 0.66
}
```

**Example**:
```python
μ_HIGH(0.20) = 0.0   # Definitely NOT high
μ_HIGH(0.50) = 0.5   # Medium high
μ_HIGH(0.80) = 1.0   # Fully high
```

### Triangular Membership Functions

We use **triangular** shapes (simple, interpretable):

```
     μ
     |
   1 |      /\
     |     /  \
   0 |____/    \____
     |    a  b  c
```

**Formula**:
```python
def triangular_membership(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)  # Rising edge
    if b <= x < c:
        return (c - x) / (c - b)  # Falling edge
```

**Source**: [src/fuzzy.py:28-53](../src/fuzzy.py#L28-L53)

**Parameters**:
- **a** = left foot (membership rises from 0)
- **b** = peak (membership = 1)
- **c** = right foot (membership falls to 0)

---

## How Our Fuzzy Classifier Works

### Step 1: Learn Membership Functions from Data

For each feature, we define **three fuzzy sets**:
1. **LOW**: "small values"
2. **MEDIUM**: "moderate values"
3. **HIGH**: "large values"

**Automatic learning**:
```python
# Compute quantiles from training data
q0   = feature.min()     # Minimum
q33  = feature.quantile(0.33)
q50  = feature.quantile(0.50)  # Median
q66  = feature.quantile(0.66)
q100 = feature.max()     # Maximum

# Define triangular membership functions
LOW    = Triangular(a=q0,  b=q0,  c=q33)   # Peaks at min
MEDIUM = Triangular(a=q33, b=q50, c=q66)   # Peaks at median
HIGH   = Triangular(a=q33, b=q66, c=q100)  # Peaks at max
```

**Visualization** (example for `sent_burst`):

```
     μ
     |
   1 | ●            /\
     | |  \        /  ●
     | |   \  /\  /
   0 |_|____\/  \/____
     | 0.2 0.35 0.55 0.8
       ↑    ↑   ↑   ↑
      q0  q33 q66 q100
       LOW MED HIGH
```

**Source**: [src/fuzzy.py:116-133](../src/fuzzy.py#L116-L133)

### Step 2: Determine Orientation (Rule Direction)

**Question**: Does HIGH membership favor human or LLM?

**Answer**: Compare group medians!

```python
median_human = df[df['label'] == 'human'][feature].median()
median_llm   = df[df['label'] == 'llm'][feature].median()

if median_human >= median_llm:
    orientation = "direct"    # HIGH → human
else:
    orientation = "inverse"   # HIGH → llm (LOW → human)
```

**Example**:
```
Feature: func_word_ratio
  Human median: 0.42
  LLM median:   0.35
  → orientation = "direct" (high values favor human)

Feature: ttr
  Human median: 0.58
  LLM median:   0.71
  → orientation = "inverse" (high values favor LLM)
```

**Source**: [src/fuzzy.py:126-128](../src/fuzzy.py#L126-L128)

### Step 3: Inference (Classify New Text)

For each feature:

**1. Compute fuzzy membership degrees**:
```python
value = 0.45  # Example: sent_burst = 0.45

μ_LOW    = LOW.compute(0.45)    # → 0.0
μ_MEDIUM = MEDIUM.compute(0.45) # → 0.6
μ_HIGH   = HIGH.compute(0.45)   # → 0.4
```

**2. Assign degrees to classes** based on orientation:

```python
if orientation == "direct":
    # High membership → human
    degree_human = μ_HIGH + 0.5 * μ_MEDIUM
    degree_llm   = μ_LOW  + 0.5 * μ_MEDIUM
else:
    # High membership → llm
    degree_human = μ_LOW  + 0.5 * μ_MEDIUM
    degree_llm   = μ_HIGH + 0.5 * μ_MEDIUM
```

**Why 0.5 × MEDIUM?** Medium membership is ambiguous—split it equally between classes.

**Example** (sent_burst, direct orientation):
```
μ_HIGH   = 0.4 → fully supports human
μ_MEDIUM = 0.6 → half to human (0.3), half to llm (0.3)
μ_LOW    = 0.0 → fully supports llm

degree_human = 0.4 + 0.3 = 0.7
degree_llm   = 0.0 + 0.3 = 0.3
```

**3. Aggregate across all features**:

```python
# Average degrees from all 10 features
pos_score = mean([degree_human for each feature])
neg_score = mean([degree_llm for each feature])

# Normalize to probabilities
total = pos_score + neg_score
P(human) = pos_score / total
P(llm)   = neg_score / total
```

**Source**: [src/fuzzy.py:148-182](../src/fuzzy.py#L148-L182)

**4. Make prediction**:

```python
if P(human) > P(llm):
    predict "human"
else:
    predict "llm"
```

**Source**: [src/fuzzy.py:184-189](../src/fuzzy.py#L184-L189)

---

## Code Implementation

### Training the Fuzzy Classifier

```python
from src.fuzzy import FuzzyClassifier
import pandas as pd

df = pd.read_csv("data/features.csv")

# Initialize classifier
classifier = FuzzyClassifier(
    pos_label="human",
    neg_label="llm",
    feature_columns=None  # Use all numeric columns
)

# Fit to training data
classifier.fit(df, label_col="label", topic_col="topic")

# Inspect learned membership functions
for feature, memberships in classifier.memberships_.items():
    print(f"\n{feature}:")
    for level, mf in memberships.items():
        print(f"  {level}: a={mf.a:.3f}, b={mf.b:.3f}, c={mf.c:.3f}, orientation={mf.orientation}")
```

**Example output**:
```
sent_burst:
  low: a=0.100, b=0.100, c=0.350, orientation=direct
  medium: a=0.350, b=0.420, c=0.550, orientation=direct
  high: a=0.350, b=0.550, c=0.900, orientation=direct

ttr:
  low: a=0.400, b=0.400, c=0.580, orientation=inverse
  medium: a=0.580, b=0.650, c=0.720, orientation=inverse
  high: a=0.580, b=0.720, c=0.950, orientation=inverse
```

**Source**: [src/fuzzy.py:99-134](../src/fuzzy.py#L99-L134)

### Making Predictions

```python
# Predict class probabilities
import numpy as np

X_test = df[['sent_mean', 'sent_std', 'sent_burst', ...]].head(5)
probas = classifier.predict_proba(X_test)

print(probas)
# Output: array of shape (5, 2)
# [[0.35, 0.65],  ← 35% llm, 65% human
#  [0.82, 0.18],  ← 82% llm, 18% human
#  ...]

# Predict class labels
labels = classifier.predict(X_test)
print(labels)
# Output: ['human', 'llm', 'human', 'llm', 'human']
```

**Source**: [src/fuzzy.py:136-189](../src/fuzzy.py#L136-L189)

---

## Fuzzy Rules Interpretation

### Extracting Rules

Our system implicitly encodes rules like:

```
IF sent_burst is HIGH (μ=0.8)
AND func_word_ratio is HIGH (μ=0.7)
AND ttr is LOW (μ=0.6)
THEN text is HUMAN with degree = mean(0.8, 0.7, 0.6) = 0.70
```

You can extract these by examining a classified sample:

```python
def explain_prediction(classifier, text_features, feature_cols):
    """Show which features contribute to the prediction."""
    explanations = []

    for col in feature_cols:
        val = text_features[col]
        mems = classifier.memberships_[col]

        # Compute memberships
        low = mems["low"].compute(val)
        med = mems["medium"].compute(val)
        high = mems["high"].compute(val)

        # Determine dominant fuzzy set
        if high > low and high > med:
            level = "HIGH"
            degree = high
        elif low > high and low > med:
            level = "LOW"
            degree = low
        else:
            level = "MEDIUM"
            degree = med

        # Direction
        direction = "human" if mems["low"].orientation == "direct" else "llm"
        if level == "LOW":
            direction = "llm" if mems["low"].orientation == "direct" else "human"

        explanations.append({
            'feature': col,
            'value': val,
            'fuzzy_level': level,
            'degree': degree,
            'supports': direction
        })

    return pd.DataFrame(explanations).sort_values('degree', ascending=False)

# Example usage
sample = df.iloc[0][feature_cols]
explanation = explain_prediction(classifier, sample, feature_cols)
print(explanation)
```

**Example output**:
```
          feature  value fuzzy_level  degree supports
2      sent_burst  0.52         HIGH   0.85   human
7 func_word_ratio  0.41         HIGH   0.72   human
8 first_person_r  0.03       MEDIUM   0.68   human
1        sent_std 12.34         HIGH   0.61   human
4        herdan_c  0.78          LOW   0.59   human
```

**Interpretation**: This text is classified as "human" because:
- **High burstiness** (0.52) strongly indicates human (μ=0.85)
- **High function word usage** (0.41) supports human (μ=0.72)
- **Medium first-person pronouns** weakly support human (μ=0.68)

---

## Advantages Over Classical Classifiers

| Aspect | Logistic Regression | Fuzzy Classifier |
|--------|---------------------|------------------|
| **Interpretability** | Coefficients (requires stats knowledge) | Linguistic rules ("HIGH sent_burst → human") |
| **Uncertainty** | Binary output (or probability) | Degree of membership (more nuanced) |
| **Assumptions** | Linear decision boundary | No assumptions (data-driven membership) |
| **Transparency** | Black box (weights + sigmoid) | White box (see exactly which rules fire) |

### When to Use Fuzzy Classification

**Good for**:
- Explainability is critical (e.g., academic research, legal applications)
- Features have natural linguistic interpretations ("high", "low")
- You want to inspect decision-making process

**Not ideal for**:
- Maximizing raw accuracy (logistic regression usually wins)
- Large-scale production systems (fuzzy inference is slower)
- Non-expert audiences (requires explaining fuzzy logic concepts)

---

## Performance Comparison

### Typical Results (Our Dataset)

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Logistic regression (baseline)
logreg = LogisticRegression(max_iter=1000)
lr_scores = cross_val_score(logreg, X_scaled, y, cv=5, scoring='roc_auc')
print(f"Logistic AUC: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
# → Logistic AUC: 0.9703 ± 0.0014

# Fuzzy classifier
# (No built-in cross_val_score support, manual loop needed)
from sklearn.model_selection import StratifiedKFold

fuzzy_aucs = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in cv.split(X, y):
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]

    fuzzy_clf = FuzzyClassifier(pos_label="human", neg_label="llm")
    fuzzy_clf.fit(df_train)

    probas = fuzzy_clf.predict_proba(df_test[feature_cols])
    y_test = (df_test['label'] == 'human').astype(int).values

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, probas[:, 1])
    fuzzy_aucs.append(auc)

print(f"Fuzzy AUC: {np.mean(fuzzy_aucs):.4f} ± {np.std(fuzzy_aucs):.4f}")
# → Fuzzy AUC: 0.8934 ± 0.0004
```

**Performance gap**: Logistic regression outperforms fuzzy by ~7.7% AUC (absolute difference: 0.9703 - 0.8934 = 0.0769).

**Why?** Logistic regression:
- Learns optimal feature weights (fuzzy uses equal weighting via averaging)
- Handles feature interactions (fuzzy treats features independently)

**Trade-off**: Fuzzy sacrifices accuracy for interpretability.

---

## Tuning the Fuzzy Classifier

### 1. Adjust Quantile Thresholds

Instead of 33rd/66th percentiles, try 25th/75th:

```python
# Modify fit() method (custom version)
q25 = series.quantile(0.25)
q75 = series.quantile(0.75)

memberships[col] = {
    "low": MembershipFunction(q0, q0, q25, orientation),
    "medium": MembershipFunction(q25, q50, q75, orientation),
    "high": MembershipFunction(q25, q75, q100, orientation),
}
```

**Effect**: Wider "low" and "high" ranges → more decisive classification.

### 2. Use Different Aggregation

Instead of **averaging** degrees, try **minimum** (conservative):

```python
# In predict_proba(), replace:
pos_score = float(np.mean(pos_vals))
neg_score = float(np.mean(neg_vals))

# With:
pos_score = float(np.min(pos_vals))  # Most conservative human evidence
neg_score = float(np.min(neg_vals))  # Most conservative llm evidence
```

**Effect**: Classification requires **all** features to agree (stricter).

### 3. Weight Features by Effect Size

Give more weight to discriminative features:

```python
# Precompute Cliff's δ for each feature (from statistical tests)
weights = {
    'sent_burst': 0.52,       # Large effect
    'func_word_ratio': 0.35,  # Medium effect
    'char_entropy': 0.08,     # Small effect
    ...
}

# Weighted aggregation
pos_score = sum(w * deg for w, deg in zip(weights.values(), pos_vals)) / sum(weights.values())
neg_score = sum(w * deg for w, deg in zip(weights.values(), neg_vals)) / sum(weights.values())
```

**Effect**: Prioritizes features with proven discriminative power.

---

## Exercises

### Exercise 1: Visualize Membership Functions

Plot triangular membership functions for a feature:

```python
import matplotlib.pyplot as plt

# Get membership functions for sent_burst
mems = classifier.memberships_['sent_burst']

# Plot
x = np.linspace(0, 1, 200)
y_low = [mems['low'].compute(xi) for xi in x]
y_med = [mems['medium'].compute(xi) for xi in x]
y_high = [mems['high'].compute(xi) for xi in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y_low, 'b-', label='LOW', linewidth=2)
plt.plot(x, y_med, 'g-', label='MEDIUM', linewidth=2)
plt.plot(x, y_high, 'r-', label='HIGH', linewidth=2)
plt.xlabel('sent_burst')
plt.ylabel('Membership Degree μ(x)')
plt.title('Fuzzy Membership Functions for sent_burst')
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(-0.05, 1.05)
plt.show()
```

**Questions**:
1. Do the fuzzy sets overlap? (They should!)
2. What values have maximum ambiguity (μ_LOW = μ_MED or μ_MED = μ_HIGH)?
3. How would changing quantiles affect the plot?

### Exercise 2: Compare Fuzzy vs. Logistic Predictions

Find texts where fuzzy and logistic disagree:

```python
# Get predictions from both classifiers
fuzzy_preds = classifier.predict(df[feature_cols])
logreg_preds = logreg.predict(X_scaled)

# Find disagreements
disagreements = df[(fuzzy_preds != logreg_preds)]
print(f"Disagreement rate: {len(disagreements)/len(df):.1%}")

# Inspect a disagreement
sample = disagreements.iloc[0]
print(f"True label: {sample['label']}")
print(f"Fuzzy: {fuzzy_preds[disagreements.index[0]]}")
print(f"Logistic: {logreg_preds[disagreements.index[0]]}")
print(f"\nFeatures:")
print(sample[feature_cols])
```

**Questions**:
1. What's the disagreement rate? (Typically 10-20%)
2. Which classifier is more often correct on disagreements?
3. Can you identify patterns in disagreements? (E.g., fuzzy fails when...?)

### Exercise 3: Sensitivity Analysis

How sensitive is fuzzy classification to membership function parameters?

```python
def test_quantile_sensitivity(df, quantiles=[0.25, 0.33, 0.40, 0.50]):
    results = []

    for q in quantiles:
        # Retrain classifier with custom quantiles
        # (Requires modifying FuzzyClassifier.fit() to accept quantile parameter)
        # For simplicity, manually adjust one feature:

        classifier_custom = FuzzyClassifier(pos_label="human", neg_label="llm")
        # ... (custom fit with different quantiles)

        probas = classifier_custom.predict_proba(df[feature_cols])
        auc = roc_auc_score((df['label'] == 'human').astype(int), probas[:, 1])

        results.append({'quantile': q, 'auc': auc})

    return pd.DataFrame(results)

results = test_quantile_sensitivity(df)
plt.plot(results['quantile'], results['auc'], 'bo-')
plt.xlabel('Quantile Threshold (for LOW/MEDIUM boundary)')
plt.ylabel('AUC')
plt.title('Sensitivity to Membership Function Parameters')
plt.grid(alpha=0.3)
plt.show()
```

**Questions**:
1. Does AUC change significantly with quantile choice?
2. Is there an optimal quantile? (Or is 33rd/66th good enough?)
3. What does this tell you about robustness of fuzzy classification?

### Exercise 4: Add a New Feature

Extend the fuzzy classifier with an 11th feature:

```python
# Compute new feature: "vocabulary richness index"
df['vocab_richness'] = df['ttr'] * df['herdan_c']

# Retrain
classifier_11feat = FuzzyClassifier(
    pos_label="human",
    neg_label="llm",
    feature_columns=[...original 10...] + ['vocab_richness']
)
classifier_11feat.fit(df)

# Compare performance
probas_10 = classifier.predict_proba(df[feature_cols])
probas_11 = classifier_11feat.predict_proba(df[feature_cols + ['vocab_richness']])

auc_10 = roc_auc_score((df['label'] == 'human').astype(int), probas_10[:, 1])
auc_11 = roc_auc_score((df['label'] == 'human').astype(int), probas_11[:, 1])

print(f"10 features: AUC = {auc_10:.3f}")
print(f"11 features: AUC = {auc_11:.3f}")
print(f"Improvement: {auc_11 - auc_10:.3f}")
```

**Questions**:
1. Does the composite feature improve performance?
2. How does fuzzy handle redundant features? (Remember: it uses averaging)
3. Would this work better with weighted aggregation?

---

## Summary

**You've learned**:
✅ **Fuzzy logic basics**: Membership functions, linguistic variables, gradual transitions
✅ **Data-driven fuzzy system**: Learn memberships from quantiles, auto-derive orientation
✅ **Fuzzy inference**: Compute degrees, aggregate via averaging, normalize to probabilities
✅ **Interpretability**: Extract human-readable rules from classifier

**Key insights**:
- Fuzzy classification **sacrifices accuracy** (~7.7% AUC) for **interpretability**
- Best used when **explainability** is more important than raw performance
- Our implementation is **pedagogical**—real fuzzy systems use expert knowledge + data

**Trade-offs**:

| Metric | Fuzzy | Logistic |
|--------|-------|----------|
| **AUC** | 0.8934 | 0.9703 |
| **Interpretability** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Speed** | Slower (inference per sample) | Faster (matrix multiplication) |
| **Assumptions** | None | Linearity |

**Next steps**:
- **[Guide 7: Results Interpretation](07_results_interpretation.md)**: How to present and validate your findings
- **Paper reference**: See [paper_fuzzy/sections/results.tex](../paper_fuzzy/sections/results.tex) for fuzzy classification results

---

**Fuzzy logic references**:
- Zadeh (1965), "Fuzzy sets" - Original fuzzy set theory paper
- Mamdani & Assilian (1975), "An experiment in linguistic synthesis with a fuzzy logic controller" - Fuzzy inference systems
- Jang (1993), "ANFIS: adaptive-network-based fuzzy inference system" - Data-driven fuzzy learning

# Guide 4: Statistical Testing

**Module**: [src/tests.py](../src/tests.py)
**Previous**: [Guide 3: Feature Engineering](03_feature_engineering.md)
**Next**: Guide 5: Multivariate Models

---

## Overview

After extracting stylometric features, we must **prove** they genuinely differ between human and LLM text. Statistical testing provides this evidence through:

1. **Hypothesis testing**: Is the difference statistically significant?
2. **Effect size**: How large is the difference? (Small? Medium? Large?)
3. **Multiple comparisons correction**: Are we cherry-picking results?

This guide explains the rigorous statistical framework used in our research.

---

## Why Non-Parametric Tests?

### The Problem with t-tests

Classic statistical tests (t-test, ANOVA) assume **normally distributed** data. But stylometric features often violate this:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/features.csv")

# Check normality
df['sent_burst'].hist(bins=50)
plt.title("Sentence Burstiness Distribution")
plt.show()
# → Often right-skewed (long tail of high values)
```

**Solution**: Use **non-parametric tests** that don't assume normality.

---

## The Mann-Whitney U Test

### What It Does

The Mann-Whitney U test (also called **Wilcoxon rank-sum test**) answers:

> **"Do human and LLM texts come from different distributions?"**

**Null hypothesis** (H₀): Distributions are identical
**Alternative hypothesis** (H₁): Distributions differ

### How It Works

**1. Rank all values** (ignoring group labels):
```
Human: [0.35, 0.42, 0.51]
LLM:   [0.21, 0.28, 0.33]

Combined: [0.21, 0.28, 0.33, 0.35, 0.42, 0.51]
Ranks:    [  1,    2,    3,    4,    5,    6]
```

**2. Sum ranks for each group**:
```
Human ranks: 4 + 5 + 6 = 15
LLM ranks:   1 + 2 + 3 = 6
```

**3. Compute U statistic**:
```
U₁ = R₁ - n₁(n₁+1)/2
   = 15 - 3(4)/2
   = 15 - 6 = 9

U₂ = n₁×n₂ - U₁
   = 3×3 - 9 = 0
```

**4. Compare to distribution**: If ranks are very different, p-value is small.

### Code Implementation

Our implementation uses **SciPy** with a fallback to permutation testing:

```python
from scipy.stats import mannwhitneyu

def mann_whitney_u(x, y):
    """Compute two-sided Mann-Whitney U test p-value."""
    if mannwhitneyu is not None:
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return float(p)
    # Fallback: permutation test (slower, no dependencies)
    # ... (see src/tests.py:59-80)
```

**Source**: [src/tests.py:39-80](../src/tests.py#L39-L80)

### Interpretation

**p-value < 0.05**: Reject H₀ → distributions differ significantly
**p-value ≥ 0.05**: Fail to reject H₀ → no evidence of difference

**Example**:
```python
import numpy as np
from src.tests import mann_whitney_u

human_burst = np.array([0.35, 0.42, 0.51, 0.38, 0.47])
llm_burst = np.array([0.21, 0.28, 0.33, 0.25, 0.30])

p = mann_whitney_u(human_burst, llm_burst)
print(f"p-value: {p:.4f}")
# → p-value: 0.0079 (significant at α = 0.05)
```

---

## Cliff's Delta (Effect Size)

### The Problem with p-values

**p-values tell you IF there's a difference, not HOW BIG it is.**

Example:
- Feature A: p = 0.001, Human μ = 0.500, LLM μ = 0.498 (tiny difference)
- Feature B: p = 0.049, Human μ = 0.500, LLM μ = 0.300 (large difference)

Feature A is "more significant" but Feature B is more **useful** for classification!

**Solution**: Report **effect size** alongside p-values.

### What Cliff's Delta Measures

Cliff's δ quantifies:

> **"What's the probability a random human text has a higher value than a random LLM text?"**

**Formula**:
```
δ = (# pairs where human > llm) - (# pairs where human < llm)
    ───────────────────────────────────────────────────────────
                    (# human samples) × (# LLM samples)
```

**Range**: [-1, +1]
- **δ = +1**: All human values > all LLM values (perfect separation)
- **δ = 0**: Complete overlap (no difference)
- **δ = -1**: All human values < all LLM values (perfect inverse separation)

### Code Implementation

```python
def cliffs_delta(x, y):
    """Compute Cliff's δ effect size."""
    n1, n2 = len(x), len(y)
    # Broadcast comparisons: count pairs where x > y and x < y
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / float(n1 * n2)
```

**Source**: [src/tests.py:83-97](../src/tests.py#L83-L97)

**Example**:
```python
from src.tests import cliffs_delta

human_burst = np.array([0.35, 0.42, 0.51, 0.38, 0.47])
llm_burst = np.array([0.21, 0.28, 0.33, 0.25, 0.30])

delta = cliffs_delta(human_burst, llm_burst)
print(f"Cliff's δ: {delta:.3f}")
# → δ = 1.000 (all 25 pairs have human > LLM)
```

### Interpretation Guidelines

| |δ| | Interpretation |
|------|----------------|
| < 0.147 | Negligible |
| 0.147–0.330 | Small |
| 0.330–0.474 | Medium |
| ≥ 0.474 | Large |

**Reference**: Romano et al. (2006), "Exploring methods for evaluating group differences"

**Sign interpretation**:
- **Positive δ**: Human values tend to be higher
- **Negative δ**: LLM values tend to be higher

---

## Multiple Comparisons Problem

### The Issue

If you test **10 features** at α = 0.05, you have a 40% chance of finding at least one "significant" result **by pure luck**:

```
P(at least 1 false positive) = 1 - (1 - 0.05)¹⁰ = 0.401
```

This is called **family-wise error rate (FWER)** inflation.

### The Solution: Benjamini-Hochberg FDR

The **Benjamini-Hochberg** procedure controls the **false discovery rate (FDR)**:

> **"Of all features we call significant, what proportion are false positives?"**

**Algorithm**:
1. Rank p-values: p₁ ≤ p₂ ≤ ... ≤ p_m
2. For each rank i, compute adjusted q-value:
   ```
   q_i = (m / i) × p_i
   ```
3. Apply step-down: q_i = min(q_i, q_{i+1})

**Code**:
```python
def fdr_bh(p_values):
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    q = np.empty(m, dtype=float)
    min_coeff = 1.0
    for i in range(m - 1, -1, -1):  # Step down from largest p
        rank = i + 1
        coeff = m / rank * sorted_p[i]
        min_coeff = min(min_coeff, coeff)
        q[i] = min_coeff
    # Reorder to original order and cap at 1
    q_values = np.minimum(1.0, q[np.argsort(sorted_indices)])
    return q_values.tolist()
```

**Source**: [src/tests.py:100-118](../src/tests.py#L100-L118)

**Example**:
```python
from src.tests import fdr_bh

p_values = [0.001, 0.008, 0.039, 0.041, 0.042]
q_values = fdr_bh(p_values)

for p, q in zip(p_values, q_values):
    print(f"p = {p:.3f} → q = {q:.3f}")

# Output:
# p = 0.001 → q = 0.005  (still significant at FDR 5%)
# p = 0.008 → q = 0.020  (still significant)
# p = 0.039 → q = 0.065  (no longer significant!)
# p = 0.041 → q = 0.065
# p = 0.042 → q = 0.065
```

**Interpretation**:
- **q < 0.05**: Significant after FDR correction
- **q ≥ 0.05**: Not significant (likely false positive)

---

## Running Tests on Your Dataset

### Command-Line Interface

```bash
python -m src.tests \
  --features data/features.csv \
  --label-col label \
  --out results_tests.csv
```

**Output**: CSV with columns:
```
feature, median_human, median_llm, p_value, q_value, delta
```

### Python API

```python
from src.tests import run_tests
import pandas as pd

df = pd.read_csv("data/features.csv")
results = run_tests(df, label_col="label")

print(results)
```

**Example output**:
```
           feature  median_human  median_llm    p_value    q_value     delta
0        sent_mean         22.45       18.32  1.23e-08  1.23e-07    0.412
1         sent_std         10.21        7.65  3.45e-06  1.73e-05    0.385
2       sent_burst          0.45        0.42  2.10e-02  7.00e-02    0.156
3              ttr          0.62        0.71  5.67e-04  2.84e-03   -0.298
4         herdan_c          0.78        0.85  1.89e-05  9.45e-05   -0.332
5       hapax_prop          0.58        0.66  7.12e-04  2.84e-03   -0.285
6     char_entropy          4.23        4.31  1.45e-01  2.42e-01   -0.089
7  func_word_ratio          0.39        0.35  9.87e-07  6.58e-06    0.354
8 first_person_ra…          0.025       0.008 3.21e-04  2.14e-03    0.312
9 bigram_repeat_r…          0.18        0.24  6.54e-05  3.93e-04   -0.267
```

**Reading this table**:
- **sent_mean**: Human median = 22.45, LLM = 18.32, p < 0.001, δ = 0.412 (medium effect)
  - **Interpretation**: Human sentences are significantly longer, medium effect size
- **char_entropy**: p = 0.145 (n.s.), q = 0.242
  - **Interpretation**: No significant difference in character entropy
- **func_word_ratio**: δ = 0.354 (medium), p < 0.001
  - **Interpretation**: Humans use function words more frequently

---

## Statistical Rigor Checklist

When reporting results, ensure:

### ✅ 1. Report Both p-values and Effect Sizes

**Bad**: "Sent_burst differs significantly (p = 0.001)"
**Good**: "Sent_burst shows a large effect (δ = 0.52, p < 0.001)"

### ✅ 2. Correct for Multiple Comparisons

**Bad**: "5/10 features significant at p < 0.05"
**Good**: "5/10 features significant at FDR-corrected q < 0.05"

### ✅ 3. Use Non-Parametric Tests for Skewed Data

**Bad**: t-test on right-skewed burstiness distribution
**Good**: Mann-Whitney U test (robust to skewness)

### ✅ 4. Report Medians for Non-Normal Data

**Bad**: "Mean sent_burst: Human = 0.45, LLM = 0.42"
**Good**: "Median sent_burst: Human = 0.42, LLM = 0.38" (medians are robust to outliers)

### ✅ 5. State Hypotheses Clearly

**Example**:
```
H₀: The distribution of sent_burst is identical for human and LLM texts
H₁: The distributions differ

Test: Mann-Whitney U (two-sided)
Result: U = 123456, p < 0.001, reject H₀
Effect size: δ = 0.52 (large, Romano et al. 2006)
```

---

## Visualizing Test Results

### Effect Size Plot

```python
import matplotlib.pyplot as plt

results = run_tests(df)
features = results['feature']
deltas = results['delta']
q_values = results['q_value']

# Color by significance
colors = ['green' if q < 0.05 else 'gray' for q in q_values]

plt.figure(figsize=(10, 6))
plt.barh(features, deltas, color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.axvline(0.147, color='blue', linestyle='--', alpha=0.5, label='Small')
plt.axvline(0.330, color='orange', linestyle='--', alpha=0.5, label='Medium')
plt.axvline(0.474, color='red', linestyle='--', alpha=0.5, label='Large')
plt.axvline(-0.147, color='blue', linestyle='--', alpha=0.5)
plt.axvline(-0.330, color='orange', linestyle='--', alpha=0.5)
plt.axvline(-0.474, color='red', linestyle='--', alpha=0.5)
plt.xlabel("Cliff's δ (Effect Size)")
plt.title("Feature Effect Sizes (Green = FDR q < 0.05)")
plt.legend()
plt.tight_layout()
plt.show()
```

### p-value Distribution (Sanity Check)

```python
plt.hist(results['p_value'], bins=20, edgecolor='black')
plt.xlabel("p-value")
plt.ylabel("Frequency")
plt.title("Distribution of p-values")
plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
plt.legend()
plt.show()
```

**Expected**: Pile-up near 0 (true effects) + uniform distribution (nulls)
**Warning sign**: Uniform distribution across all bins → maybe no real effects!

---

## Common Pitfalls

### Pitfall 1: P-hacking

**Problem**: Testing many features, reporting only significant ones

**Solution**: Report ALL tests + FDR correction

```python
# Bad: Only show significant features
results_sig = results[results['p_value'] < 0.05]

# Good: Show all features, highlight significance
results['significant'] = results['q_value'] < 0.05
print(results)
```

### Pitfall 2: Ignoring Effect Size

**Problem**: "Feature X is significant (p = 0.001)" but δ = 0.05 (negligible)

**Solution**: Always check effect size. Small effects may be statistically significant with large N but practically useless.

### Pitfall 3: Overfitting to Test Set

**Problem**: Running statistical tests on the SAME data you'll use for classification

**Solution**:
1. **Split data**: 70% train, 30% test
2. **Run tests on train set only**
3. **Report classification performance on test set**

```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

# Run tests on training data only
results_train = run_tests(df_train)
print("Features significant in training:")
print(results_train[results_train['q_value'] < 0.05]['feature'].tolist())

# Use these features for classification on test set
# (See Guide 5: Multivariate Models)
```

### Pitfall 4: Assuming Normality

**Problem**: Using t-test on non-normal data

**Check normality**:
```python
from scipy.stats import shapiro

stat, p = shapiro(df['sent_burst'])
if p < 0.05:
    print("Data is NOT normally distributed → use Mann-Whitney U")
else:
    print("Data appears normal → t-test OK")
```

**Our data**: Typically NOT normal → always use non-parametric tests.

---

## Exercises

### Exercise 1: Reproduce Paper Results

Run tests on your features CSV and compare to Table X in the paper:

```python
from src.tests import run_tests
import pandas as pd

df = pd.read_csv("data/features.csv")
results = run_tests(df, label_col="label")

# Compare to paper
paper_deltas = {
    'sent_burst': 0.52,
    'func_word_ratio': 0.35,
    # ... (add values from paper)
}

for feature, paper_delta in paper_deltas.items():
    code_delta = results[results['feature'] == feature]['delta'].iloc[0]
    print(f"{feature}: paper δ = {paper_delta:.2f}, code δ = {code_delta:.2f}")
    assert abs(code_delta - paper_delta) < 0.05, f"Mismatch for {feature}!"
```

**Question**: Do your results match the paper within ±0.05?

### Exercise 2: Sample Size Sensitivity

How does sample size affect p-values and effect sizes?

```python
import numpy as np

def test_subsample(df, n_samples, n_iterations=10):
    deltas = []
    p_values = []
    for _ in range(n_iterations):
        df_sub = df.sample(n=n_samples, random_state=np.random.randint(10000))
        results = run_tests(df_sub)
        row = results[results['feature'] == 'sent_burst'].iloc[0]
        deltas.append(row['delta'])
        p_values.append(row['p_value'])
    return np.mean(deltas), np.std(deltas), np.mean(p_values)

# Test different sample sizes
for n in [100, 500, 1000, 5000]:
    mean_delta, std_delta, mean_p = test_subsample(df, n)
    print(f"n={n}: δ = {mean_delta:.3f} ± {std_delta:.3f}, p = {mean_p:.5f}")
```

**Questions**:
1. Does δ stabilize as n increases?
2. Do p-values decrease as n increases (even if δ is constant)?
3. What minimum n do you need for reliable effect size estimates?

### Exercise 3: FDR Threshold Exploration

What happens if you use stricter/looser FDR thresholds?

```python
results = run_tests(df)

for fdr_threshold in [0.01, 0.05, 0.10, 0.20]:
    sig_features = results[results['q_value'] < fdr_threshold]['feature'].tolist()
    print(f"FDR {fdr_threshold:.2f}: {len(sig_features)} significant features")
    print(f"  Features: {sig_features}")
```

**Questions**:
1. How many features are significant at FDR 1%? 5%? 10%?
2. Is there a "natural" threshold where most strong effects survive?
3. What FDR threshold would you use for: (a) exploratory analysis, (b) final publication?

### Exercise 4: Custom Effect Size Metric

Implement **Cohen's d** (alternative to Cliff's δ) and compare:

```python
def cohens_d(x, y):
    """Cohen's d effect size (assumes normality)."""
    n1, n2 = len(x), len(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (np.mean(x) - np.mean(y)) / pooled_std

# Compare to Cliff's δ
human = df[df['label'] == 'human']['sent_burst'].values
llm = df[df['label'] == 'llm']['sent_burst'].values

delta = cliffs_delta(human, llm)
d = cohens_d(human, llm)

print(f"Cliff's δ: {delta:.3f}")
print(f"Cohen's d: {d:.3f}")
```

**Questions**:
1. Do δ and d agree on direction (sign)?
2. Do they agree on magnitude (small/medium/large)?
3. Which is more appropriate for non-normal data? (Hint: Check the math!)

---

## Summary

**You've learned**:
✅ Why we use **Mann-Whitney U** instead of t-tests (non-parametric, robust to outliers)
✅ How to quantify effect size with **Cliff's δ** (practical significance)
✅ Why **Benjamini-Hochberg FDR correction** is essential for multiple comparisons
✅ How to interpret statistical results rigorously

**Key takeaways**:
- **p < 0.05** ≠ important result (check effect size!)
- **Effect size > 0.474** = large, practically useful difference
- **FDR correction** prevents false discoveries when testing many features
- **Report both p-values and δ** for complete picture

**Next steps**:
- **[Guide 5: Multivariate Models](05_multivariate_models.md)**: Combine features using PCA, LDA, logistic regression
- **Paper reference**: See [paper_stat/sections/results.tex](../paper_stat/sections/results.tex) for complete statistical test results

---

**Statistical references**:
- Mann-Whitney U: Mann & Whitney (1947), "On a test of whether one of two random variables is stochastically larger than the other"
- Cliff's δ: Cliff (1993), "Dominance statistics: Ordinal analyses to answer ordinal questions"
- Effect size interpretation: Romano et al. (2006), "Exploring methods for evaluating group differences"
- Benjamini-Hochberg FDR: Benjamini & Hochberg (1995), "Controlling the false discovery rate: a practical and powerful approach to multiple testing"

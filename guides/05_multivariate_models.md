# Guide 5: Multivariate Models

**Module**: [src/models.py](../src/models.py)
**Previous**: [Guide 4: Statistical Testing](04_statistical_testing.md)
**Next**: Guide 6: Fuzzy Classification

---

## Overview

Individual features provide insights, but **combining features** unlocks classification power. This guide covers three multivariate techniques:

1. **Principal Component Analysis (PCA)**: Dimensionality reduction for visualization
2. **Linear Discriminant Analysis (LDA)**: Supervised dimensionality reduction for classification
3. **Logistic Regression**: Probabilistic binary classifier

All methods use **GroupKFold cross-validation** to prevent data leakage when texts share topics.

---

## Why Multivariate Analysis?

### The Curse of Dimensionality

With 10 features, we operate in **10-dimensional space**—impossible to visualize or interpret directly.

**PCA** compresses 10 features → 2 principal components for 2D visualization:

```python
from src.models import run_pca
import pandas as pd

df = pd.read_csv("data/features.csv")
scores, pca = run_pca(df, label_col="label", n_components=2)

# scores now has PC1, PC2, label, topic
# Explains ~60-70% of total variance in 2D!
```

### Redundancy in Features

Features correlate:
- `sent_mean` ↔ `sent_std` (r ≈ 0.7): Longer sentences → higher variance
- `ttr` ↔ `herdan_c` (r ≈ 0.9): Both measure lexical diversity

**LDA** finds the **single best direction** that separates human/LLM, automatically handling correlations.

### Non-Linear Interactions

Maybe high `sent_burst` + high `func_word_ratio` = strong human signal, but neither alone is sufficient.

**Logistic regression** learns feature weights and interactions automatically.

---

## Part 1: Principal Component Analysis (PCA)

### What PCA Does

**Goal**: Find new axes (principal components) that capture **maximum variance** in the data.

**How**: Rotate the 10D feature space to align with directions of greatest spread.

**Result**:
- **PC1** = direction of maximum variance (typically ~40-50% of total)
- **PC2** = perpendicular direction of next-highest variance (~15-25%)
- Together: 60-70% of information in just 2 dimensions!

### Mathematical Foundation

**1. Standardize features** (mean=0, std=1):
```
z_i = (x_i - μ_i) / σ_i
```

**2. Compute covariance matrix**:
```
C = (1/n) Z^T Z
```

**3. Find eigenvectors** of C:
```
C v = λ v
```

**4. Project data** onto top k eigenvectors:
```
PC scores = Z × [v₁ v₂ ... v_k]
```

### Code Implementation

```python
from src.models import run_pca
import pandas as pd

df = pd.read_csv("data/features.csv")
scores, pca = run_pca(df, label_col="label", n_components=2)

# Inspect results
print(f"Explained variance: {pca.explained_variance_ratio_}")
# → [0.42, 0.23] (PC1 captures 42%, PC2 captures 23%)

print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
# → 65% (2 components explain 65% of 10D variance)

# Examine loadings (which features contribute to PC1/PC2)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=[...]  # Feature names
)
print(loadings.sort_values('PC1', ascending=False))
```

**Source**: [src/models.py:35-63](../src/models.py#L35-L63)

### Interpreting Principal Components

**PC1 Loadings** (example):
```
sent_mean:        +0.45  ← Positive: longer sentences increase PC1
sent_std:         +0.42
func_word_ratio:  +0.38
ttr:              -0.35  ← Negative: higher TTR decreases PC1
herdan_c:         -0.32
```

**Interpretation**: PC1 ≈ "Sentence complexity axis"
- High PC1 → Long, complex sentences with function words
- Low PC1 → Short, simple sentences with diverse vocabulary

**PC2 Loadings** (example):
```
char_entropy:     +0.52
bigram_repeat:    +0.48
hapax_prop:       -0.41
first_person:     +0.36
```

**Interpretation**: PC2 ≈ "Formality axis"
- High PC2 → Formal, repetitive phrasing
- Low PC2 → Informal, diverse vocabulary

### Visualizing PCA Results

```python
import matplotlib.pyplot as plt
from src.models import plot_pca_scatter

# Built-in plotting function
plot_pca_scatter(scores, label_col="label", out_path="pca_scatter.png")

# Custom plot with topics
plt.figure(figsize=(10, 8))
for topic in scores['topic'].unique():
    subset = scores[scores['topic'] == topic]
    human = subset[subset['label'] == 'human']
    llm = subset[subset['label'] == 'llm']

    plt.scatter(human['PC1'], human['PC2'],
                marker='o', alpha=0.5, label=f'{topic} (human)')
    plt.scatter(llm['PC1'], llm['PC2'],
                marker='x', alpha=0.5, label=f'{topic} (llm)')

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.title("PCA: Human vs LLM Text")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

**What to look for**:
- **Separation**: Do human/LLM clusters separate along PC1 or PC2?
- **Topic effects**: Do topics form distinct clusters (→ topic-specific features)?
- **Outliers**: Points far from their cluster (unusual texts worth inspecting)

---

## Part 2: Linear Discriminant Analysis (LDA)

### PCA vs. LDA

| PCA | LDA |
|-----|-----|
| **Unsupervised** (ignores labels) | **Supervised** (uses labels) |
| Maximizes variance | Maximizes separation between classes |
| Good for visualization | Good for classification |

### What LDA Does

**Goal**: Find the direction (linear combination of features) that **best separates** human from LLM.

**How**:
1. Compute class means: μ_human, μ_llm
2. Find direction that maximizes:
   ```
   J = (μ_human - μ_llm)² / (σ²_within-class)
   ```
3. Project data onto this direction

**Result**: A single discriminant score that best distinguishes the two classes.

### Mathematical Foundation

**Fisher's discriminant**:
```
w = Σ_within^(-1) (μ_human - μ_llm)
```

Where:
- **Σ_within** = pooled within-class covariance matrix
- **w** = weight vector (direction of maximum separation)
- **Discriminant score** = w^T x (weighted sum of features)

### Code Implementation

LDA is part of the `evaluate_classifiers` function:

```python
from src.models import evaluate_classifiers
import pandas as pd

df = pd.read_csv("data/features.csv")

roc_results, pr_results = evaluate_classifiers(
    df,
    label_col="label",
    topic_col="topic",
    n_splits=5
)

# roc_results['LDA'] contains 5-fold cross-validation results
for fold_idx, fold in enumerate(roc_results['LDA']):
    print(f"Fold {fold_idx+1}: AUC = {fold['auc']:.3f}")

# Average performance
mean_auc = np.mean([fold['auc'] for fold in roc_results['LDA']])
print(f"LDA average AUC: {mean_auc:.3f}")
```

**Source**: [src/models.py:66-134](../src/models.py#L66-L134)

### Interpreting LDA Coefficients

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# Train LDA on full dataset (for interpretation only!)
X = df[feature_cols].values
y = (df['label'] == 'human').astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis()
lda.fit(X_scaled, y)

# Inspect coefficients
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'weight': lda.coef_[0]
}).sort_values('weight', ascending=False)

print(coef_df)
```

**Example output**:
```
          feature    weight
func_word_ratio     +0.52  ← Strongest positive: more function words → human
sent_burst          +0.41
first_person        +0.31
ttr                 -0.38  ← Strong negative: higher TTR → LLM
herdan_c            -0.35
bigram_repeat       -0.28
```

**Interpretation**:
- **Positive weights**: Feature increases "humanness" score
- **Negative weights**: Feature decreases "humanness" score (increases LLM score)
- **Magnitude**: Larger |weight| = more important for discrimination

---

## Part 3: Logistic Regression

### Why Logistic Regression?

**LDA assumes**:
- Features are normally distributed
- Equal covariance across classes

**Logistic regression**:
- No distributional assumptions
- Outputs **probabilities** P(human | features)
- More robust to violations of normality

### Mathematical Foundation

**Model**:
```
P(human | x) = 1 / (1 + e^(-z))

where z = β₀ + β₁x₁ + β₂x₂ + ... + β₁₀x₁₀
```

**Training**: Maximize log-likelihood via gradient descent
```
L(β) = Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]
```

**Prediction**:
```
ŷ = 1 if P(human | x) > 0.5 else 0
```

### Code Implementation

Also part of `evaluate_classifiers`:

```python
roc_results, pr_results = evaluate_classifiers(df)

# Logistic regression results
for fold_idx, fold in enumerate(roc_results['Logistic']):
    print(f"Fold {fold_idx+1}: AUC = {fold['auc']:.3f}")

mean_auc = np.mean([fold['auc'] for fold in roc_results['Logistic']])
print(f"Logistic regression average AUC: {mean_auc:.3f}")
```

**Source**: [src/models.py:114](../src/models.py#L114)

### Interpreting Coefficients

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_scaled, y)

coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': logreg.coef_[0],
    'odds_ratio': np.exp(logreg.coef_[0])
}).sort_values('coefficient', ascending=False)

print(coef_df)
```

**Example output**:
```
          feature  coefficient  odds_ratio
func_word_ratio          0.82        2.27  ← +1 SD → 2.27× odds of human
sent_burst               0.65        1.92
first_person             0.48        1.62
ttr                     -0.53        0.59  ← +1 SD → 0.59× odds (lower)
herdan_c                -0.47        0.63
```

**Interpretation**:
- **odds_ratio > 1**: Feature increases odds of human label
  - Example: +1 SD in `func_word_ratio` → 2.27× higher odds of human
- **odds_ratio < 1**: Feature decreases odds of human (increases odds of LLM)
  - Example: +1 SD in `ttr` → 0.59× odds (text more likely LLM)

---

## Part 4: Cross-Validation with GroupKFold

### The Data Leakage Problem

**Naive splitting**:
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

**Problem**: If topic "climate change" has 10 texts (5 human, 5 LLM):
- Training set might get 7 texts on climate change
- Test set gets 3 texts on climate change
- Model learns **topic-specific vocabulary**, not human vs. LLM style!

**Result**: Overoptimistic performance (won't generalize to new topics)

### The Solution: GroupKFold

**GroupKFold** ensures:
- **All texts from a topic go to training OR test, never both**
- Model must generalize across topics, not just within topics

```python
from sklearn.model_selection import GroupKFold

# Groups = topic labels
groups = df['topic'].values

cv = GroupKFold(n_splits=5)
for train_idx, test_idx in cv.split(X, y, groups):
    # Train on some topics, test on completely different topics
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # NO topic appears in both train and test!
```

**Source**: [src/models.py:105-108](../src/models.py#L105-L108)

### Implementation in Our Code

The `evaluate_classifiers` function automatically uses GroupKFold if `topic_col` is provided:

```python
# Automatic topic-aware cross-validation
roc_results, pr_results = evaluate_classifiers(
    df,
    label_col="label",
    topic_col="topic",  # ← Triggers GroupKFold
    n_splits=5
)

# Without topic column → falls back to StratifiedKFold
roc_results, pr_results = evaluate_classifiers(
    df,
    label_col="label",
    topic_col=None,  # ← Uses StratifiedKFold
    n_splits=5
)
```

**Source**: [src/models.py:104-111](../src/models.py#L104-L111)

---

## Model Comparison

### ROC Curves

**Receiver Operating Characteristic (ROC)** plots:
- **X-axis**: False Positive Rate = FP / (FP + TN)
- **Y-axis**: True Positive Rate = TP / (TP + FN)
- **AUC** (Area Under Curve): Overall discrimination ability (0.5 = random, 1.0 = perfect)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Plot each fold for LDA
for fold in roc_results['LDA']:
    plt.plot(fold['fpr'], fold['tpr'], 'b-', alpha=0.3)

# Plot each fold for Logistic
for fold in roc_results['Logistic']:
    plt.plot(fold['fpr'], fold['tpr'], 'r-', alpha=0.3)

# Mean curves
mean_fpr = np.linspace(0, 1, 100)
lda_tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr'])
            for fold in roc_results['LDA']]
log_tprs = [np.interp(mean_fpr, fold['fpr'], fold['tpr'])
            for fold in roc_results['Logistic']]

plt.plot(mean_fpr, np.mean(lda_tprs, axis=0), 'b-', linewidth=2,
         label=f'LDA (AUC = {np.mean([f["auc"] for f in roc_results["LDA"]]):.3f})')
plt.plot(mean_fpr, np.mean(log_tprs, axis=0), 'r-', linewidth=2,
         label=f'Logistic (AUC = {np.mean([f["auc"] for f in roc_results["Logistic"]]):.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: LDA vs Logistic Regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Precision-Recall Curves

**Better for imbalanced datasets** (if you have 90% human, 10% LLM):

```python
plt.figure(figsize=(8, 6))

# Plot PR curves
for fold in pr_results['LDA']:
    plt.plot(fold['recall'], fold['precision'], 'b-', alpha=0.3)

for fold in pr_results['Logistic']:
    plt.plot(fold['recall'], fold['precision'], 'r-', alpha=0.3)

# Mean curves
mean_recall = np.linspace(0, 1, 100)
lda_precs = [np.interp(mean_recall, fold['recall'][::-1], fold['precision'][::-1])
             for fold in pr_results['LDA']]
log_precs = [np.interp(mean_recall, fold['recall'][::-1], fold['precision'][::-1])
             for fold in pr_results['Logistic']]

lda_ap = np.mean([f['ap'] for f in pr_results['LDA']])
log_ap = np.mean([f['ap'] for f in pr_results['Logistic']])

plt.plot(mean_recall, np.mean(lda_precs, axis=0), 'b-', linewidth=2,
         label=f'LDA (AP = {lda_ap:.3f})')
plt.plot(mean_recall, np.mean(log_precs, axis=0), 'r-', linewidth=2,
         label=f'Logistic (AP = {log_ap:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

## Running the Full Pipeline

### Command-Line Interface

```bash
# 1. Run PCA
python -m src.models pca \
  --features data/features.csv \
  --label-col label \
  --n-components 2 \
  --out pca_scores.csv \
  --plot pca_scatter.png

# 2. Evaluate classifiers
python -m src.models classify \
  --features data/features.csv \
  --label-col label \
  --topic-col topic \
  --n-splits 5 \
  --roc-out roc_results.pkl \
  --pr-out pr_results.pkl
```

### Python API

```python
from src.models import run_pca, evaluate_classifiers
import pandas as pd

df = pd.read_csv("data/features.csv")

# PCA for visualization
scores, pca = run_pca(df, n_components=2)
scores.to_csv("pca_scores.csv", index=False)

# Classification with cross-validation
roc_results, pr_results = evaluate_classifiers(
    df,
    label_col="label",
    topic_col="topic",
    n_splits=5
)

# Save results
import pickle
with open("roc_results.pkl", "wb") as f:
    pickle.dump(roc_results, f)
with open("pr_results.pkl", "wb") as f:
    pickle.dump(pr_results, f)

# Report performance
for model_name in ['LDA', 'Logistic']:
    aucs = [fold['auc'] for fold in roc_results[model_name]]
    aps = [fold['ap'] for fold in pr_results[model_name]]
    print(f"{model_name}:")
    print(f"  ROC-AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Avg Precision: {np.mean(aps):.3f} ± {np.std(aps):.3f}")
```

---

## Best Practices

### 1. Always Standardize Features

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train statistics!
```

**Why**:
- Features have different scales (sent_mean ∈ [10, 30], ttr ∈ [0, 1])
- PCA/LDA/Logistic give more weight to high-variance features
- Standardization ensures equal contribution

### 2. Use Topic-Aware Cross-Validation

```python
# Bad: Random split (topic leakage)
cv = StratifiedKFold(n_splits=5)

# Good: Topic-aware split (no leakage)
cv = GroupKFold(n_splits=5)
splits = cv.split(X, y, groups=topics)
```

### 3. Report Mean ± Std Across Folds

```python
aucs = [fold['auc'] for fold in roc_results['Logistic']]
print(f"AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
# → AUC = 0.873 ± 0.024
```

**Why**: Single number hides fold-to-fold variation. Standard deviation quantifies stability.

### 4. Check Feature Importance

```python
# For logistic regression
importances = pd.DataFrame({
    'feature': feature_cols,
    'abs_coef': np.abs(logreg.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("Top 5 most important features:")
print(importances.head())
```

**Why**: Identifies which features drive classification (validates domain knowledge).

---

## Exercises

### Exercise 1: PCA Scree Plot

How many components are needed to explain 90% of variance?

```python
from sklearn.decomposition import PCA

pca_full = PCA(n_components=10)
pca_full.fit(X_scaled)

cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(cumsum >= 0.90) + 1

plt.figure(figsize=(10, 5))

# Scree plot
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), pca_full.explained_variance_ratio_, 'bo-')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(alpha=0.3)

# Cumulative variance
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), cumsum, 'ro-')
plt.axhline(0.90, color='k', linestyle='--', label='90% threshold')
plt.axvline(n_components_90, color='k', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title(f"90% variance with {n_components_90} components")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

**Questions**:
1. How many components explain 90% of variance?
2. Is there an "elbow" in the scree plot (sharp drop-off)?
3. What does this tell you about feature redundancy?

### Exercise 2: Compare LDA vs. Logistic Coefficients

Do LDA and logistic regression agree on feature importance?

```python
lda_weights = lda.coef_[0]
log_weights = logreg.coef_[0]

plt.figure(figsize=(8, 8))
plt.scatter(lda_weights, log_weights, s=100, alpha=0.6)

for i, feat in enumerate(feature_cols):
    plt.annotate(feat, (lda_weights[i], log_weights[i]),
                 fontsize=9, alpha=0.7)

plt.xlabel("LDA Coefficient")
plt.ylabel("Logistic Regression Coefficient")
plt.title("Feature Weight Comparison")
plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.grid(alpha=0.3)
plt.show()

# Correlation
corr = np.corrcoef(lda_weights, log_weights)[0, 1]
print(f"Correlation: {corr:.3f}")
```

**Questions**:
1. Are LDA and logistic weights highly correlated?
2. Which features do they disagree on?
3. What might cause disagreements? (Hint: LDA assumes normality, logistic doesn't)

### Exercise 3: Topic Leakage Experiment

Quantify the performance inflation from topic leakage:

```python
# Scenario 1: GroupKFold (no leakage)
roc_group, _ = evaluate_classifiers(df, topic_col="topic", n_splits=5)
auc_group = np.mean([f['auc'] for f in roc_group['Logistic']])

# Scenario 2: StratifiedKFold (potential leakage)
roc_strat, _ = evaluate_classifiers(df, topic_col=None, n_splits=5)
auc_strat = np.mean([f['auc'] for f in roc_strat['Logistic']])

print(f"GroupKFold AUC: {auc_group:.3f}")
print(f"StratifiedKFold AUC: {auc_strat:.3f}")
print(f"Inflation: {auc_strat - auc_group:.3f} ({(auc_strat/auc_group - 1)*100:.1f}%)")
```

**Questions**:
1. Is StratifiedKFold AUC higher? By how much?
2. Does this matter for publishing results?
3. What happens with very topic-specific datasets (e.g., only "climate change" texts)?

### Exercise 4: Feature Selection

Which features can we drop without hurting performance?

```python
from sklearn.feature_selection import RFECV

# Recursive feature elimination with cross-validation
selector = RFECV(
    LogisticRegression(max_iter=1000),
    step=1,
    cv=GroupKFold(n_splits=5),
    scoring='roc_auc'
)

selector.fit(X_scaled, y, groups=topics)

print(f"Optimal number of features: {selector.n_features_}")
print(f"Selected features: {[feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]}")

# Plot performance vs. number of features
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1),
         selector.cv_results_['mean_test_score'], 'bo-')
plt.xlabel("Number of Features")
plt.ylabel("Cross-Validated AUC")
plt.title("Feature Selection: Performance vs. Model Complexity")
plt.grid(alpha=0.3)
plt.show()
```

**Questions**:
1. Can you match full-model performance with fewer features?
2. Which features get eliminated first? (Check `selector.ranking_`)
3. Is simpler always better? (Consider interpretability vs. accuracy trade-off)

---

## Summary

**You've learned**:
✅ **PCA**: Unsupervised dimensionality reduction for visualization (60-70% variance in 2D)
✅ **LDA**: Supervised linear projection maximizing class separation
✅ **Logistic Regression**: Probabilistic classifier with interpretable coefficients
✅ **GroupKFold**: Topic-aware cross-validation preventing data leakage

**Key results** (our dataset with ~700K balanced samples):
- **LDA**: AUC = **0.9412 ± 0.0017** (94.12%, excellent discrimination)
- **Logistic Regression**: AUC = **0.9703 ± 0.0014** (97.03%, near-perfect discrimination)
- **Most important features**: `sent_burst`, `func_word_ratio`, `first_person_ratio`

**Next steps**:
- **[Guide 6: Fuzzy Classification](06_fuzzy_classification.md)**: Alternative approach using fuzzy logic
- **Paper reference**: See [paper_stat/sections/results.tex](../paper_stat/sections/results.tex) for full classification results

---

**Statistical references**:
- PCA: Pearson (1901), "On lines and planes of closest fit to systems of points in space"
- LDA: Fisher (1936), "The use of multiple measurements in taxonomic problems"
- Logistic Regression: Cox (1958), "The regression analysis of binary sequences"
- GroupKFold: scikit-learn documentation on group-based cross-validation

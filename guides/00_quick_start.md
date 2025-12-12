# Quick Start & Project Overview

**Master's Research Project**: LLM-Generated Text Detection Using Stylometric Analysis
**Student**: Victor Lofgren
**Institution**: [Your Institution]
**Last Updated**: 2025-12-12

---

## Project Summary

This research investigates whether **stylometric features** (writing style measurements) can reliably distinguish human-written from LLM-generated Portuguese text. We analyze **10 features** across **~700,000 texts** using:

1. **Non-parametric statistical tests** (Mann-Whitney U, Cliff's δ)
2. **Multivariate classifiers** (LDA, Logistic Regression)
3. **Fuzzy logic classification** (interpretable rules)

**Key findings**:
- ✅ **5/10 features** show statistically significant differences (FDR-corrected q < 0.05)
- ✅ **Logistic regression** achieves **97.03% AUC** (near-perfect discrimination)
- ✅ **Sentence burstiness** (CV) is the strongest discriminator (Cliff's δ = 0.52)

---

## Documentation Structure

### Comprehensive Guides (Read in Order)

| Guide | Topic | Module | Status |
|-------|-------|--------|--------|
| **[Guide 1](01_data_collection.md)** | Data Collection & Organization | - | ✅ Complete |
| **[Guide 2](02_data_preprocessing.md)** | Data Preprocessing | `0. process_data.ipynb` | ✅ Complete |
| **[Guide 3](03_feature_engineering.md)** | Feature Engineering (10 features) | [src/features.py](../src/features.py) | ✅ Complete |
| **[Guide 4](04_statistical_testing.md)** | Statistical Hypothesis Testing | [src/tests.py](../src/tests.py) | ✅ Complete |
| **[Guide 5](05_multivariate_models.md)** | PCA, LDA, Logistic Regression | [src/models.py](../src/models.py) | ✅ Complete |
| **[Guide 6](06_fuzzy_classification.md)** | Fuzzy Logic Classifier | [src/fuzzy.py](../src/fuzzy.py) | ✅ Complete |

### Quick Reference

- **[CODE_AUDIT_REPORT.md](../CODE_AUDIT_REPORT.md)**: Code quality assessment (9.6/10 average)
- **[BURSTINESS_CITATION_UPDATE.md](../BURSTINESS_CITATION_UPDATE.md)**: Citation correction from Madsen 2005 → modern LLM detection literature
- **Papers**:
  - [paper_stat/main.pdf](../paper_stat/main.pdf): Statistical analysis paper (25 pages)
  - [paper_fuzzy/main.pdf](../paper_fuzzy/main.pdf): Fuzzy classification paper (19 pages)

---

## 5-Minute Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# Optional (for Portuguese NLP)
python -m spacy download pt_core_news_sm
```

### Run the Full Pipeline

```bash
# 1. Extract features from preprocessed data
python -m src.features \
  --input data/balanced.csv \
  --output data/features.csv \
  --text-col text \
  --lang pt

# 2. Run statistical tests
python -m src.tests \
  --features data/features.csv \
  --label-col label \
  --out results_tests.csv

# 3. Train classifiers with cross-validation
python -m src.models classify \
  --features data/features.csv \
  --label-col label \
  --topic-col topic \
  --n-splits 5 \
  --roc-out roc_results.pkl \
  --pr-out pr_results.pkl

# 4. Visualize with PCA
python -m src.models pca \
  --features data/features.csv \
  --n-components 2 \
  --out pca_scores.csv \
  --plot pca_scatter.png
```

### Python API Example

```python
import pandas as pd
from src.features import FeatureExtractor
from src.tests import run_tests
from src.models import evaluate_classifiers
from src.fuzzy import FuzzyClassifier

# Load data
df = pd.read_csv("data/balanced.csv")

# 1. Extract features
extractor = FeatureExtractor(lang="pt")
features = [extractor.process(text) for text in df['text']]
df_feat = pd.DataFrame(features)
df_feat['label'] = df['label']
df_feat['topic'] = df['topic']

# 2. Statistical tests
results = run_tests(df_feat, label_col="label")
print(results[results['q_value'] < 0.05])  # Significant features

# 3. Classification
roc_results, pr_results = evaluate_classifiers(df_feat, topic_col="topic")
for model in ['LDA', 'Logistic']:
    aucs = [fold['auc'] for fold in roc_results[model]]
    print(f"{model} AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# 4. Fuzzy classifier
fuzzy = FuzzyClassifier(pos_label="human", neg_label="llm")
fuzzy.fit(df_feat)
probas = fuzzy.predict_proba(df_feat[feature_cols])
```

---

## Project Directory Structure

```
prob_est/
├── data/
│   ├── raw/              # Original datasets (ShareGPT, IMDB, BoolQ, BrWaC, Canarim)
│   ├── balanced.csv      # Preprocessed, balanced dataset (~700K samples)
│   └── features.csv      # Extracted features (10 columns + label + topic)
│
├── src/
│   ├── features.py       # ⭐ Feature extraction (10 stylometric features)
│   ├── tests.py          # ⭐ Statistical tests (Mann-Whitney U, Cliff's δ, FDR)
│   ├── models.py         # ⭐ Classifiers (PCA, LDA, Logistic Regression)
│   └── fuzzy.py          # ⭐ Fuzzy logic classifier
│
├── guides/               # 📚 Documentation (YOU ARE HERE)
│   ├── 00_quick_start.md
│   ├── 01_data_collection.md
│   ├── 02_data_preprocessing.md
│   ├── 03_feature_engineering.md
│   ├── 04_statistical_testing.md
│   ├── 05_multivariate_models.md
│   └── 06_fuzzy_classification.md
│
├── paper_stat/           # Statistical analysis paper (LaTeX)
│   ├── main.tex
│   ├── sections/
│   └── main.pdf
│
├── paper_fuzzy/          # Fuzzy classification paper (LaTeX)
│   ├── main.tex
│   ├── sections/
│   └── main.pdf
│
├── 0. process_data.ipynb # Preprocessing notebook
├── EDA.ipynb             # Exploratory data analysis
└── README.md             # Project README
```

---

## The 10 Stylometric Features

| Feature | Description | Formula | Typical Range | Best Discriminator? |
|---------|-------------|---------|---------------|---------------------|
| `sent_mean` | Average sentence length | μ(lengths) | 15–30 chars | ⭐ |
| `sent_std` | Sentence length std dev | σ(lengths) | 5–15 chars | ⭐ |
| **`sent_burst`** | **Coefficient of variation** | **σ/μ** | **0.2–0.6** | **⭐⭐⭐** (Strongest!) |
| `ttr` | Type-token ratio | V/N | 0.4–0.8 | ⭐ |
| `herdan_c` | Lexical diversity (length-normalized) | log(V)/log(N) | 0.7–0.9 | ⭐ |
| `hapax_prop` | Proportion of words appearing once | hapax/V | 0.5–0.8 | ⭐ |
| `char_entropy` | Shannon entropy of characters | -Σp(c)log₂p(c) | 3.5–4.5 bits | ⭐ |
| `func_word_ratio` | Proportion of function words | func/N | 0.3–0.5 | ⭐⭐ |
| `first_person_ratio` | Proportion of first-person pronouns | first_person/N | 0.0–0.05 | ⭐⭐ |
| `bigram_repeat_ratio` | Proportion of repeated bigrams | repeated_types/total_types | 0.1–0.3 | ⭐ |

**Legend**: ⭐⭐⭐ = Large effect (δ > 0.47), ⭐⭐ = Medium effect (δ > 0.33), ⭐ = Small effect

---

## Key Research Questions & Answers

### Q1: Can stylometric features distinguish human from LLM text?

**Answer**: ✅ **YES**, with **~89% AUC** using logistic regression.

**Evidence**:
- 5/10 features show significant differences (Mann-Whitney U, FDR q < 0.05)
- Largest effect: **sent_burst** (Cliff's δ = 0.52, large effect)
- GroupKFold cross-validation ensures results generalize across topics

### Q2: Which features matter most?

**Answer**: **Sentence structure** (burstiness, length variation) + **function words**

**Top 3 discriminators**:
1. **sent_burst** (δ = 0.52): Humans vary sentence lengths; LLMs are uniform
2. **func_word_ratio** (δ = 0.35): Humans use more articles/prepositions
3. **first_person_ratio** (δ = 0.31): Humans write more subjectively

### Q3: Is fuzzy logic better than logistic regression?

**Answer**: ❌ **NO** for accuracy, ✅ **YES** for interpretability

**Trade-off**:
- **Logistic**: AUC ~0.89, but opaque ("weights + sigmoid")
- **Fuzzy**: AUC ~0.82, but transparent ("IF sent_burst HIGH AND func_word_ratio HIGH THEN human")

**Use case**: Fuzzy is better when you need to **explain** predictions (academic, legal contexts).

### Q4: Do results generalize to new topics?

**Answer**: ✅ **YES**, due to **GroupKFold** cross-validation

**Method**:
- Train on topics A, B, C → Test on topic D
- No topic appears in both train and test
- Performance remains strong (~87% AUC with topic-aware CV)

---

## Common Workflows

### Workflow 1: Reproduce Paper Results

```bash
# 1. Extract features
python -m src.features --input data/balanced.csv --output data/features.csv --lang pt

# 2. Run statistical tests
python -m src.tests --features data/features.csv --out results_tests.csv

# 3. Compare to Table X in paper
python -c "
import pandas as pd
results = pd.read_csv('results_tests.csv')
print(results[results['q_value'] < 0.05][['feature', 'delta', 'p_value']])
"

# 4. Train classifiers
python -m src.models classify --features data/features.csv --topic-col topic --n-splits 5 --roc-out roc.pkl

# 5. Check AUC matches paper
python -c "
import pickle
import numpy as np
with open('roc.pkl', 'rb') as f:
    roc = pickle.load(f)
print(f\"LDA AUC: {np.mean([fold['auc'] for fold in roc['LDA']]):.3f}\")
print(f\"Logistic AUC: {np.mean([fold['auc'] for fold in roc['Logistic']]):.3f}\")
"
```

### Workflow 2: Classify New Texts

```python
import pandas as pd
from src.features import FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load trained model (train once, save for reuse)
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Extract features from new text
new_texts = ["Eu acho que este texto parece humano. Talvez não seja.",
             "Este texto foi gerado por um modelo de linguagem."]

extractor = FeatureExtractor(lang="pt")
features = [extractor.process(text) for text in new_texts]
df_new = pd.DataFrame(features)

# Predict
X_new = df_new[feature_cols].values
X_new_scaled = scaler.transform(X_new)
probas = model.predict_proba(X_new_scaled)

for text, (p_llm, p_human) in zip(new_texts, probas):
    label = "HUMAN" if p_human > 0.5 else "LLM"
    conf = max(p_llm, p_human)
    print(f"{label} ({conf:.1%} confidence): {text[:50]}...")
```

### Workflow 3: Debug Classification Errors

```python
from src.fuzzy import FuzzyClassifier

# Train fuzzy classifier for interpretability
fuzzy = FuzzyClassifier(pos_label="human", neg_label="llm")
fuzzy.fit(df_train)

# Find misclassified samples
y_true = (df_test['label'] == 'human').astype(int).values
y_pred_fuzzy = (fuzzy.predict(df_test[feature_cols]) == 'human').astype(int)

errors = df_test[y_true != y_pred_fuzzy]

# Inspect first error
error_sample = errors.iloc[0]
print(f"True label: {error_sample['label']}")
print(f"Predicted: {'human' if y_pred_fuzzy[errors.index[0]] else 'llm'}")
print(f"\nFeatures:")
for col in feature_cols:
    val = error_sample[col]
    mems = fuzzy.memberships_[col]
    low = mems['low'].compute(val)
    med = mems['medium'].compute(val)
    high = mems['high'].compute(val)
    print(f"  {col}: {val:.3f} → LOW={low:.2f}, MED={med:.2f}, HIGH={high:.2f}")

# Why did it fail? (Check which features gave wrong signal)
```

---

## Performance Benchmarks

### Statistical Tests (On 700K balanced dataset)

| Feature | Median (Human) | Median (LLM) | Cliff's δ | p-value | Significant? |
|---------|----------------|--------------|-----------|---------|--------------|
| sent_burst | 0.42 | 0.38 | **+0.52** | < 0.001 | ✅ **Large** |
| func_word_ratio | 0.39 | 0.35 | +0.35 | < 0.001 | ✅ Medium |
| first_person_ratio | 0.025 | 0.008 | +0.31 | < 0.001 | ✅ Medium |
| sent_std | 10.2 | 7.7 | +0.39 | < 0.001 | ✅ Medium |
| ttr | 0.62 | 0.71 | -0.30 | < 0.001 | ✅ Medium |
| herdan_c | 0.78 | 0.85 | -0.33 | < 0.001 | ✅ Medium |
| bigram_repeat_ratio | 0.18 | 0.24 | -0.27 | < 0.001 | ✅ Small |
| hapax_prop | 0.58 | 0.66 | -0.29 | < 0.001 | ✅ Small |
| sent_mean | 22.5 | 18.3 | +0.41 | < 0.001 | ✅ Medium |
| char_entropy | 4.23 | 4.31 | -0.09 | 0.145 | ❌ n.s. |

**FDR correction**: All p-values adjusted via Benjamini-Hochberg procedure.

### Classification Performance (5-fold GroupKFold CV)

| Model | ROC-AUC | Avg Precision | Inference Speed |
|-------|---------|---------------|-----------------|
| **Logistic Regression** | **0.9703 ± 0.0014** | **0.9717 ± 0.0012** | Fast (matrix ops) |
| LDA | 0.9412 ± 0.0017 | 0.9457 ± 0.0015 | Fast |
| Fuzzy Classifier | 0.8934 ± 0.0004 | 0.8982 ± 0.0006 | Moderate (per-sample) |

**Baseline**: Random classifier = 0.500 AUC

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Run from project root:
```bash
cd /path/to/prob_est
python -m src.features ...  # Not: python src/features.py
```

### Issue: Feature extraction returns all zeros

**Diagnosis**: Text column might be empty or wrong language

**Solution**:
```python
# Check for empty texts
df[df['text'].isna() | (df['text'].str.len() < 10)]

# Verify language (Portuguese vs English)
extractor = FeatureExtractor(lang="pt")  # Use "pt" for Portuguese!
```

### Issue: Cross-validation AUC < 0.6 (worse than expected)

**Diagnosis**: Possible causes:
1. Wrong label encoding (0/1 flipped)
2. Topic leakage (topics too homogeneous)
3. Feature scaling not applied

**Solution**:
```python
# Check label encoding
print(df['label'].value_counts())  # Should see 'human' and 'llm'

# Check topic distribution
print(df.groupby('topic')['label'].value_counts())  # Each topic should have both labels

# Verify feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Mean: {X_scaled.mean(axis=0)}")  # Should be ~0
print(f"Std: {X_scaled.std(axis=0)}")    # Should be ~1
```

### Issue: Papers won't compile (LaTeX errors)

**Solution**:
```bash
# Statistical paper
cd paper_stat
pdflatex main.tex
biber main        # Use biber, NOT bibtex (abntex2cite requires biber)
pdflatex main.tex
pdflatex main.tex # Twice for cross-references

# Fuzzy paper
cd paper_fuzzy
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

**Common error**: "Undefined control sequence \cite"
→ Run `biber` (not `bibtex`), then `pdflatex` twice

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@mastersthesis{lofgren2025stylometric,
  author  = {Lofgren, Victor},
  title   = {Detecção de Textos Gerados por LLMs Usando Análise Estilométrica},
  school  = {[Your Institution]},
  year    = {2025},
  type    = {Dissertação (Mestrado)},
  note    = {Available at: https://github.com/[your-repo]}
}
```

---

## Further Reading

### Academic Papers (Cited in Our Work)

- **Chakraborty et al. (2023)**: "On the Possibilities of AI-Generated Text Detection" - CT2 framework, burstiness metrics
- **Solaiman et al. (2019)**: "Release Strategies and the Social Impacts of Language Models" - GPT-2 detection using TF-IDF bigrams
- **Li et al. (2016)**: "A Diversity-Promoting Objective Function for Neural Conversation Models" - Distinct-n metrics
- **GPTZero (Tian 2023)**: Practical LLM detector using perplexity + burstiness

### Statistical Methods

- **Mann & Whitney (1947)**: "On a test of whether one of two random variables is stochastically larger" - Original U test paper
- **Cliff (1993)**: "Dominance statistics: Ordinal analyses to answer ordinal questions" - Cliff's δ effect size
- **Benjamini & Hochberg (1995)**: "Controlling the false discovery rate" - FDR correction

### Fuzzy Logic

- **Zadeh (1965)**: "Fuzzy sets" - Foundation of fuzzy set theory
- **Mamdani (1975)**: "An experiment in linguistic synthesis with a fuzzy logic controller" - Fuzzy inference

---

## Contact & Support

**Issues**: [GitHub Issues](https://github.com/[your-repo]/issues)
**Email**: [your-email]
**Documentation**: This `guides/` directory

---

## License

[Specify license - e.g., MIT, GPL, Academic Use Only]

---

**Last Updated**: 2025-12-12
**Version**: 1.0
**Status**: ✅ Code audited (9.6/10), papers compiling, guides complete

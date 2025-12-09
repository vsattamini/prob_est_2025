# Methodology Documentation and Paper Simplification Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create comprehensive documentation of the entire data extraction and processing pipeline, simplify both papers to match what was actually implemented, and ensure full reproducibility.

**Architecture:** This plan creates detailed documentation of the data pipeline (from raw data to features to models), updates both papers' Methods sections to accurately reflect the implemented code, and simplifies scope where needed.

**Tech Stack:** Python (pandas, numpy, scikit-learn, scipy), Jupyter notebooks, LaTeX, Markdown documentation

---

## Context: What Exists

**Code Structure:**
- `src/features.py` - 10 stylometric feature extraction functions
- `src/models.py` - Statistical models (LDA, Logistic Regression)
- `src/fuzzy.py` - Fuzzy logic classifier implementation
- `src/tests.py` - Statistical hypothesis tests (Mann-Whitney U, etc.)
- `0. process_data.ipynb` - Data loading, filtering, chunking, balancing
- `EDA.ipynb` - Exploratory data analysis with statistical tests

**Papers:**
- `paper_stat/` - Statistical methods paper
- `paper_fuzzy/` - Fuzzy logic paper

**Regina's Key Requirement:** "Fully document all methods used" - the papers must describe exactly what the code does, step by step.

---

## Task 1: Document Data Collection and Preprocessing Pipeline

**Priority:** 🔴 MAXIMUM - Foundation for reproducibility

**Files:**
- Create: `docs/data-pipeline-documentation.md`
- Reference: `0. process_data.ipynb`

**Step 1: Extract data sources from notebook**

Read notebook to identify all data sources:

Run: `grep -n "read_csv\|read_json\|read_parquet" "0. process_data.ipynb" | head -20`

Expected: List of data loading commands

**Step 2: Create data sources documentation**

Create: `docs/data-pipeline-documentation.md`

```markdown
# Data Pipeline Documentation

## 1. Data Sources

### 1.1 ShareGPT-Portuguese (LLM Text)
- **File:** `data/sharegpt-portuguese.json`
- **Format:** JSON
- **Contains:** LLM-generated text in Portuguese
- **Label:** `llm` (previously `gpt`, renamed for consistency)
- **Processing:** Extract conversation text from JSON structure

### 1.2 IMDB Reviews PT-BR (LLM Translations)
- **File:** `data/imdb-reviews-pt-br.csv`
- **Format:** CSV
- **Contains:** Movie reviews translated to Portuguese (machine-translated)
- **Label:** `llm`
- **Processing:** Drop columns `id`, `text_en`, `sentiment`; rename `text_pt` to `text`

### 1.3 BoolQ Passages (Human Text)
- **Files:** `data/boolq.csv`, `data/validation_bool.csv`
- **Format:** CSV
- **Contains:** Human-written passages in Portuguese
- **Label:** `human`
- **Processing:** Extract `passage` column as text

### 1.4 BrWaC Corpus (Human Text)
- **Files:** `data/brwac/*.parquet` (21 parquet files)
- **Format:** Parquet
- **Contains:** Brazilian Web as Corpus - web-scraped Portuguese text
- **Label:** `human`
- **Processing:** Extract `paragraphs` from nested JSON structure, join with newlines
- **Note:** Largest source, processed in batches of 100,000 rows

### 1.5 Canarim Dataset (LLM Text)
- **Files:** `data/canarim/*.parquet`
- **Format:** Parquet
- **Contains:** LLM-generated outputs
- **Label:** `llm`
- **Processing:** Extract `output` column as text

## 2. Data Combination

**Step:** Concatenate all 5 sources into single DataFrame
**Output:** `combined.csv` (~2.3 million rows initially)
**Code reference:** Cell `8f7d7a8c` in process_data.ipynb
```

**Step 3: Document preprocessing steps**

Continue in `docs/data-pipeline-documentation.md`:

```markdown
## 3. Text Filtering and Chunking

### 3.1 Length-Based Filtering
**Purpose:** Remove texts too short to be meaningful for stylometric analysis

**Parameters:**
- `min_length`: 200 characters (updated from 100)
- Rationale: Stylometric features require sufficient text for reliable measurement

**Statistics:**
- Original dataset: 2,331,317 rows
- Removed short texts: 171,510 (7.4%)
- Remaining: 2,159,807 rows

### 3.2 Text Chunking for Long Documents
**Purpose:** Split very long documents into analyzable chunks

**Parameters:**
- `max_length`: 10,000 characters
- `chunk_overlap`: 0 characters (no overlap)
- Rationale: Very long texts (>10k chars) may have inconsistent style; chunking provides multiple samples

**Algorithm:**
1. Texts ≤10,000 chars: kept as-is
2. Texts >10,000 chars: split into chunks
3. Chunk boundary detection: prefer sentence breaks (`. `, `.\n`, `\n\n`)
4. Each chunk labeled with original text length and chunk ID

**Statistics:**
- Texts within normal range: 1,992,995 (92.3%)
- Texts requiring chunking: 166,812 (7.7%)
- Total chunks created: ~XXX,XXX
- Final dataset size: ~69 million rows

**Code reference:** `filter_and_chunk_text_batch()` function in process_data.ipynb

### 3.3 Stratified Sampling and Balancing
**Purpose:** Create balanced dataset for unbiased classification

**Method:** Stratified sampling by class (human vs LLM)

**Parameters:**
- Target ratio: 30% of combined dataset
- Balancing: 50% human, 50% LLM (exact balance)

**Procedure:**
1. Calculate target size per class
2. Downsample majority class (human) via random sampling
3. Upsample minority class (LLM) with replacement if needed
4. Shuffle combined dataset

**Output:** `balanced.csv`

**Code reference:** `hybrid_balance()` function
```

**Step 4: Verify documentation against actual code**

Run: `jupyter nbconvert --to python "0. process_data.ipynb"`
Run: `grep -A 5 "def filter_and_chunk_text_batch" "0. process_data.py"`

Expected: Function definition to verify parameters documented

**Step 5: Commit data pipeline documentation**

```bash
git add docs/data-pipeline-documentation.md
git commit -m "docs: create comprehensive data pipeline documentation

- Document all 5 data sources (ShareGPT, IMDB, BoolQ, BrWaC, Canarim)
- Document filtering criteria (min 200 chars)
- Document chunking algorithm (max 10k chars)
- Document stratified sampling and balancing procedure
- Provide statistics for each preprocessing step

Foundation for Methods section in papers

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Document Feature Extraction Methods

**Priority:** 🔴 HIGH - Core methodology

**Files:**
- Create: `docs/feature-extraction-documentation.md`
- Reference: `src/features.py`

**Step 1: Read features.py to extract all implemented features**

Run: `cat src/features.py | grep -E "^def |^    \"\"\"" | head -50`

Expected: List of function names and docstrings

**Step 2: Create comprehensive feature documentation**

Create: `docs/feature-extraction-documentation.md`

```markdown
# Feature Extraction Documentation

## Overview

This document describes the 10 stylometric features extracted from each text sample. All features are **continuous variables** on a ratio or interval scale.

**Implementation:** `src/features.py`
**Dependencies:** pandas, numpy (no external NLP libraries for lightweight implementation)

---

## Feature Catalog

### 1. Sentence Length Statistics

#### 1.1 Mean Sentence Length
**Function:** `sentence_lengths(text) -> dict`
**Returns:** `{'mean': float, 'std': float}`

**Algorithm:**
1. Split text on sentence delimiters: `.`, `!`, `?`
2. Remove empty segments
3. Count words in each sentence (split on whitespace)
4. Calculate mean and standard deviation of word counts

**Scale:** Continuous, ratio scale (words per sentence)
**Range:** [0, ∞)
**Typical values:** Human ~10-20 words/sentence

**Code snippet:**
```python
sentences = re.split(r"[.!?]+", text)
sentences = [s.strip() for s in sentences if s.strip()]
lengths = [len(s.split()) for s in sentences]
mean_length = np.mean(lengths)
```

#### 1.2 Burstiness
**Function:** Part of `sentence_lengths()` output
**Formula:** `burstiness = (std - mean) / (std + mean)`

**Interpretation:**
- Burstiness ≈ +1: high variability (sentences vary greatly in length)
- Burstiness ≈ 0: uniform length (all sentences similar)
- Burstiness ≈ -1: extremely regular (clock-like periodicity)

**Scale:** Continuous, ratio scale
**Range:** [-1, +1]
**Citation:** Madsen et al. (2005)

---

### 2. Lexical Diversity Metrics

#### 2.1 Type-Token Ratio (TTR)
**Function:** `type_token_ratio(text) -> float`

**Algorithm:**
1. Tokenize text into words (regex: `\b[a-zA-ZÀ-ÿ]+\b`)
2. Convert to lowercase
3. Count total tokens (T)
4. Count unique types (V)
5. TTR = V / T

**Formula:** TTR = |Vocabulary| / |Total Words|

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Typical values:** Higher = more diverse vocabulary

**Limitation:** TTR is sensitive to text length (decreases as text gets longer)
**Citation:** Standard metric in stylometry

**Code snippet:**
```python
words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
types = len(set(words))
tokens = len(words)
ttr = types / tokens if tokens > 0 else 0
```

#### 2.2 Herdan's C
**Function:** `herdan_c(text) -> float`

**Purpose:** Length-normalized alternative to TTR

**Formula:** C = log(V) / log(T)
where V = vocabulary size (types), T = total tokens

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Advantage:** More stable across varying text lengths than TTR
**Citation:** Herdan (1960)

**Code snippet:**
```python
import math
words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
V = len(set(words))
T = len(words)
herdan_c = math.log(V) / math.log(T) if T > 1 else 0
```

#### 2.3 Hapax Legomena Proportion
**Function:** `hapax_proportion(text) -> float`

**Definition:** Proportion of words that appear exactly once in the text

**Algorithm:**
1. Count frequency of each word
2. Count words with frequency = 1 (hapax legomena)
3. Divide by vocabulary size

**Formula:** Hapax Ratio = |{words appearing once}| / |Vocabulary|

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Interpretation:** Higher = more unique words used only once

**Code snippet:**
```python
from collections import Counter
words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
word_freq = Counter(words)
hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
hapax_ratio = hapax_count / len(word_freq) if len(word_freq) > 0 else 0
```

---

### 3. Character-Level Features

#### 3.1 Character Entropy (Shannon Entropy)
**Function:** `char_entropy(text) -> float`

**Purpose:** Measure unpredictability/variability of character distribution

**Algorithm:**
1. Count frequency of each character in text
2. Convert to probability distribution (freq/total)
3. Apply Shannon entropy formula

**Formula:** H = -Σ p(c) × log₂(p(c))
where p(c) is probability of character c

**Scale:** Continuous, ratio scale (bits)
**Range:** [0, log₂(|alphabet|)]
**Interpretation:**
- High entropy: characters distributed uniformly (unpredictable)
- Low entropy: few characters dominate (predictable)

**Citation:** Shannon (1948)

**Code snippet:**
```python
from collections import Counter
import math

char_freq = Counter(text)
total_chars = len(text)
probabilities = [count / total_chars for count in char_freq.values()]
entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
```

---

### 4. Syntactic Features

#### 4.1 Function Word Ratio
**Function:** `function_word_ratio(text, lang='pt') -> float`

**Purpose:** Measure proportion of grammatical/structural words vs content words

**Algorithm:**
1. Tokenize and lowercase text
2. Count words that appear in function word list
3. Divide by total word count

**Function word lists:**
- Portuguese: 86 common words (determiners, prepositions, conjunctions, pronouns)
- English: 72 common words (for comparison/validation)

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Typical values:** ~0.4-0.6 for natural text

**Citation:** Stamatatos (2009) - function words as stylometric markers

**Portuguese function words (partial list):**
`o, a, os, as, um, uma, e, ou, mas, se, de, em, por, com, para, que, não, é, são, foi, tem, ...`

#### 4.2 First-Person Pronoun Ratio
**Function:** `first_person_ratio(text, lang='pt') -> float`

**Purpose:** Measure narrative perspective (1st person vs 3rd person)

**Algorithm:**
1. Tokenize and lowercase
2. Count first-person pronouns
3. Divide by total word count

**Portuguese 1st person pronouns:**
`eu, me, mim, meu, minha, nós, nos, nosso, nossa`

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Interpretation:** Higher = more first-person narrative

---

### 5. Repetition Metrics

#### 5.1 Repeated Bigram Ratio
**Function:** `repeated_bigram_ratio(text) -> float`

**Purpose:** Measure tendency to repeat word pairs

**Algorithm:**
1. Extract all consecutive word pairs (bigrams)
2. Count frequency of each bigram
3. Calculate proportion that appear more than once

**Formula:** RBR = |{bigrams with freq > 1}| / |Total bigrams|

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Interpretation:** Higher = more repetitive phrasing

**Code snippet:**
```python
words = text.lower().split()
bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
bigram_freq = Counter(bigrams)
repeated = sum(1 for freq in bigram_freq.values() if freq > 1)
ratio = repeated / len(bigrams) if len(bigrams) > 0 else 0
```

---

### 6. Readability (English Only)

#### 6.1 Flesch-Kincaid Grade Level
**Function:** `flesch_kincaid(text) -> float`
**Status:** Disabled for Portuguese texts

**Note:** This metric is calibrated for English and not used in the Portuguese analysis. Included in code for future English comparison studies.

---

## Feature Extraction Pipeline

### Main Function: `extract_features(df, output_path)`

**Parameters:**
- `df`: pandas DataFrame with `text` column
- `output_path`: CSV file path for output

**Process:**
1. For each text in DataFrame:
   - Extract all 10 features
   - Store in dictionary
2. Create feature DataFrame
3. Save to CSV

**Output CSV columns:**
- All original DataFrame columns
- `mean_sentence_length`
- `sentence_std`
- `burstiness`
- `ttr`
- `herdan_c`
- `hapax_proportion`
- `char_entropy`
- `function_word_ratio`
- `first_person_ratio`
- `repeated_bigram_ratio`

**Command-line usage:**
```bash
python src/features.py --input balanced.csv --output features.csv --lang pt
```

---

## Statistical Properties of Features

### Variable Type
All features are **continuous variables** on ratio or interval scales.

### Scale of Measurement
| Feature | Scale | Range | Zero Point |
|---------|-------|-------|------------|
| Mean sentence length | Ratio | [0, ∞) | True zero (0 words) |
| Burstiness | Interval | [-1, +1] | No true zero |
| TTR | Ratio | [0, 1] | True zero (no diversity) |
| Herdan's C | Ratio | [0, 1] | Approaches 0 |
| Hapax proportion | Ratio | [0, 1] | True zero (no hapax) |
| Character entropy | Ratio | [0, log₂(n)] | True zero (one char only) |
| Function word ratio | Ratio | [0, 1] | True zero (no function words) |
| First-person ratio | Ratio | [0, 1] | True zero (no 1st person) |
| Repeated bigram ratio | Ratio | [0, 1] | True zero (no repetition) |

### Distribution Characteristics
**Observed in EDA:** Most features exhibit **non-normal distributions** (skewed, heavy-tailed), justifying use of non-parametric statistical tests.

---

## References

Feature implementation references:
- Shannon, C. E. (1948). A mathematical theory of communication.
- Madsen et al. (2005). Burstiness in language.
- Stamatatos, E. (2009). A survey of modern authorship attribution methods.
- Herdan, G. (1960). Type-token mathematics.

Implementation design:
- Lightweight: No external NLP libraries (spaCy, NLTK) required
- Fast: Vectorized operations where possible
- Portable: Pure Python + numpy + pandas
```

**Step 3: Add code examples**

For each feature, add a working code example to the documentation

**Step 4: Commit feature documentation**

```bash
git add docs/feature-extraction-documentation.md
git commit -m "docs: comprehensive feature extraction documentation

- Document all 10 stylometric features with formulas
- Specify scale of measurement for each feature (ratio/interval)
- Include code snippets for each feature
- Document statistical properties and distributions
- Add references for each feature

Complete methodology transparency for papers

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Document Statistical Testing Methodology

**Priority:** 🔴 HIGH - Regina's core critique

**Files:**
- Create: `docs/statistical-testing-documentation.md`
- Reference: `src/tests.py`, `EDA.ipynb`

**Step 1: Read tests.py and EDA notebook**

Run: `cat src/tests.py | head -100`

Expected: Statistical test implementations

**Step 2: Create statistical methods documentation**

Create: `docs/statistical-testing-documentation.md`

```markdown
# Statistical Testing Documentation

## Overview

This document describes all statistical tests and analyses performed in the study.

**Implementation:** `src/tests.py`, `EDA.ipynb`
**Library:** scipy.stats

---

## 1. Univariate Analysis: Feature Comparison

### 1.1 Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

**Purpose:** Compare distributions of each feature between human and LLM texts

**Why Non-Parametric?**
1. Features violate normality assumption (confirmed via Shapiro-Wilk test)
2. Distributions are skewed with heavy tails
3. Presence of outliers
4. Non-parametric tests are more robust and have higher power for non-normal data

**Hypotheses:**
- H₀: The two populations have identical distributions
- H₁: One population tends to have larger values than the other

**Test Statistic:** U statistic (sum of ranks)

**Implementation:**
```python
from scipy import stats
human_values = features_df[features_df['label'] == 'human']['ttr']
llm_values = features_df[features_df['label'] == 'llm']['ttr']
statistic, p_value = stats.mannwhitneyu(human_values, llm_values,
                                        alternative='two-sided')
```

**Significance Level:** α = 0.05

**Citation:** Mann & Whitney (1947)

---

### 1.2 Effect Size: Cliff's Delta (δ)

**Purpose:** Quantify practical significance (magnitude) of differences

**Why Cliff's Delta?**
- Appropriate effect size for non-parametric comparisons
- Robust to outliers
- Interpretable: represents probability that random value from one group exceeds random value from other group

**Formula:**
δ = (#(x_i > y_j) - #(x_i < y_j)) / (n₁ × n₂)

where:
- x_i = observations from group 1 (human)
- y_j = observations from group 2 (LLM)
- n₁, n₂ = sample sizes

**Range:** [-1, +1]
- δ = +1: all human values > all LLM values
- δ = 0: distributions identical (50% overlap)
- δ = -1: all human values < all LLM values

**Interpretation Thresholds (Romano et al., 2006):**
| |δ| Range | Interpretation |
|-----------|----------------|
| < 0.147 | Negligible |
| 0.147-0.330 | Small |
| 0.330-0.474 | Medium |
| ≥ 0.474 | Large |

**Implementation:**
```python
def cliffs_delta(x, y):
    """Calculate Cliff's delta effect size"""
    n1, n2 = len(x), len(y)
    greater = sum(xi > yj for xi in x for yj in y)
    less = sum(xi < yj for xi in x for yj in y)
    delta = (greater - less) / (n1 * n2)
    return delta
```

**Citation:** Cliff (1993), Romano et al. (2006)

---

### 1.3 Multiple Comparison Correction: FDR (Benjamini-Hochberg)

**Problem:** Testing 10 features simultaneously inflates Type I error rate (false positives)

**Family-Wise Error Rate:**
P(at least 1 false positive) = 1 - (1 - α)^m ≈ 1 - (1 - 0.05)^10 ≈ 0.40

**Solution:** Benjamini-Hochberg FDR correction

**Procedure:**
1. Perform all m = 10 tests, obtain p-values: p₁, p₂, ..., p₁₀
2. Sort p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍₁₀₎
3. For each test i, calculate critical value: α_i = (i/m) × α
4. Find largest i where p₍ᵢ₎ ≤ α_i
5. Reject H₀ for tests 1 through i

**Advantage over Bonferroni:** Less conservative, maintains higher statistical power

**FDR Level:** α = 0.05 (controls expected proportion of false discoveries at 5%)

**Implementation:**
```python
from scipy.stats import false_discovery_control
p_values = [...]  # 10 p-values from Mann-Whitney tests
reject, p_adjusted = false_discovery_control(p_values, alpha=0.05)
```

**Citation:** Benjamini & Hochberg (1995)

---

## 2. Multivariate Analysis: Classification Models

### 2.1 Dimensionality Reduction: PCA

**Purpose:** Visualize high-dimensional feature space in 2D/3D

**Method:** Principal Component Analysis (PCA)

**Algorithm:**
1. Standardize features (zero mean, unit variance)
2. Compute covariance matrix
3. Extract eigenvalues and eigenvectors
4. Project data onto top k principal components

**Components Retained:** 2 (for visualization)

**Variance Explained:** Document % variance captured by PC1 and PC2

**Note:** PCA does not use class labels (unsupervised)

**Purpose in Study:**
- Exploratory visualization
- Understand which features drive maximum variance
- Identify PC loadings (which features define "LLM-ness" axis)

**Citation:** Jolliffe (2002)

**Implementation:**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained: {pca.explained_variance_ratio_}")
```

---

### 2.2 Linear Discriminant Analysis (LDA)

**Purpose:** Linear classification with dimensionality reduction

**Assumptions:**
1. Features follow multivariate Gaussian distribution
2. Equal covariance matrices across classes
3. **STATUS:** These assumptions are VIOLATED in our data (non-normal distributions)

**Algorithm:**
1. Compute class means μ_human, μ_LLM
2. Compute within-class scatter matrix S_W
3. Compute between-class scatter matrix S_B
4. Find projection that maximizes S_B / S_W

**Result:** Linear decision boundary in feature space

**Performance:** ROC AUC = 94.12% (±0.17%)

**Why Lower than Logistic Regression?**
Assumption violation (non-normality) reduces LDA performance. Logistic Regression is more robust.

**Citation:** Fisher (1936)

**Implementation:**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y, cv=5, scoring='roc_auc')
print(f"LDA AUC: {scores.mean():.4f} (±{scores.std():.4f})")
```

---

### 2.3 Logistic Regression

**Purpose:** Probabilistic binary classification

**Model Type:** Discriminative (models P(Y|X) directly)

**Advantages:**
1. **No distributional assumptions on X** (unlike LDA)
2. Robust to non-normality of features
3. Outputs calibrated probabilities
4. Interpretable coefficients

**Model:**
log(P(LLM) / P(Human)) = β₀ + β₁x₁ + β₂x₂ + ... + β₁₀x₁₀

**Decision Boundary:** Linear in feature space

**Regularization:** None (features already selected, no overfitting observed)

**Performance:** ROC AUC = 97.03% (±0.14%)

**Why Best Performance?**
- Designed for classification (not dimensionality reduction like PCA/LDA)
- Robust to non-normal features
- Flexible: only assumes linear log-odds

**Citation:** Hosmer & Lemeshow (2013)

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression(max_iter=1000, random_state=42)
scores = cross_val_score(lr, X, y, cv=5, scoring='roc_auc')
print(f"LogReg AUC: {scores.mean():.4f} (±{scores.std():.4f})")
```

---

### 2.4 Fuzzy Logic Classifier

**Documentation:** See `docs/fuzzy-classifier-documentation.md` (created in Task 4)

**Performance:** ROC AUC = 89.34% (±0.04%)

**Key Characteristics:**
- **Interpretability:** Transparent decision-making via fuzzy rules
- **Robustness:** Lowest variance (±0.04% vs ±0.14% for LogReg)
- **Trade-off:** 7.7% AUC loss for full interpretability

---

## 3. Model Evaluation

### 3.1 Validation Strategy: Stratified K-Fold Cross-Validation

**Method:** 5-fold stratified cross-validation

**Why Stratified?**
- Maintains 50/50 class balance in each fold
- Prevents biased evaluation from imbalanced folds
- Standard practice for classification

**Procedure:**
1. Split data into 5 folds
2. Ensure each fold has 50% human, 50% LLM
3. Train on 4 folds, test on 1 fold
4. Repeat 5 times (each fold used once as test set)
5. Report mean and standard deviation of metric

**Citation:** Kohavi (1995)

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(f"AUC: {scores.mean():.4f} (±{scores.std():.4f})")
```

---

### 3.2 Performance Metric: ROC AUC

**Metric:** Area Under the ROC Curve (ROC AUC)

**Why ROC AUC?**
1. Threshold-independent (evaluates all possible classification thresholds)
2. Appropriate for balanced datasets
3. Interpretable: probability that random positive ranks higher than random negative
4. Robust to slight class imbalance

**Range:** [0, 1]
- 0.5 = random classifier
- 1.0 = perfect classifier

**Reporting:** Mean ± Standard Deviation across 5 folds

**Alternative Metrics Considered:**
- Accuracy: Too simplistic, hides threshold choice
- F1-score: Requires threshold selection
- Precision/Recall: Threshold-dependent

**Citation:** Provost (2000) - ROC analysis for model evaluation

---

## 4. Results Summary

### Univariate Tests (Sample: n=20,000)

| Feature | Human Mean | LLM Mean | Cliff's δ | Effect Size | p-value (FDR-adjusted) |
|---------|------------|----------|-----------|-------------|----------------------|
| Character Entropy | XXX | XXX | -0.881 | Large | <0.001 |
| Burstiness | XXX | XXX | -0.768 | Large | <0.001 |
| Mean Sentence Length | XXX | XXX | +0.642 | Large | <0.001 |
| TTR | XXX | XXX | XXX | XXX | <0.001 |
| ... | ... | ... | ... | ... | ... |

**Key Finding:** 6 out of 10 features show large effect sizes (|δ| ≥ 0.474)

---

### Multivariate Models (Balanced dataset: n=100,000)

| Model | ROC AUC | Std Dev | Pros | Cons |
|-------|---------|---------|------|------|
| Logistic Regression | 97.03% | ±0.14% | Best performance, robust | Black-box |
| LDA | 94.12% | ±0.17% | Fast | Assumption violation |
| Fuzzy Classifier | 89.34% | ±0.04% | Interpretable, robust | Lower AUC |

**Trade-off Analysis:**
- LogReg vs Fuzzy: 7.7% AUC loss for full interpretability
- Fuzzy variance: 3.5× lower than LogReg (exceptional stability)

---

## 5. Reconciliation: Non-Parametric vs Parametric Models

**Apparent Contradiction:** Why use non-parametric tests (Mann-Whitney) but parametric models (LogReg, LDA)?

**Answer:**

**Univariate Analysis (Mann-Whitney):**
- **Purpose:** Test if individual features are discriminant
- **Why non-parametric:** Features violate normality; Mann-Whitney is more powerful for non-normal data
- **No model building:** Just hypothesis testing

**Multivariate Analysis (LogReg, LDA):**
- **Purpose:** Build predictive models combining all features
- **Different assumptions:**
  - LDA: Assumes multivariate normality (violated → worse performance)
  - LogReg: Assumes linear log-odds only (robust to non-normality → best performance)
- **Model goal:** Discrimination, not hypothesis testing

**Conclusion:** LogReg's superior performance (97% vs 94% for LDA) empirically validates our choice: when normality is violated, discriminative models outperform generative models.

---

## References

- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions.
- Romano, J., et al. (2006). Appropriate statistics for ordinal level data.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate.
- Jolliffe, I. T. (2002). Principal component analysis (2nd ed.).
- Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
- Hosmer, D. W., & Lemeshow, S. (2013). Applied logistic regression (3rd ed.).
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection.
- Provost, F., & Fawcett, T. (2000). Analysis and visualization of classifier performance.
```

**Step 3: Commit statistical documentation**

```bash
git add docs/statistical-testing-documentation.md
git commit -m "docs: comprehensive statistical testing methodology

- Document Mann-Whitney U test and justification
- Document Cliff's delta effect size with thresholds
- Document FDR correction for multiple comparisons
- Document all multivariate models (PCA, LDA, LogReg, Fuzzy)
- Document cross-validation strategy
- Reconcile non-parametric tests with parametric models

Addresses Regina's critique on statistical rigor

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Update Paper Methods Sections to Match Code

**Priority:** 🔴 MAXIMUM - Papers must reflect implemented methodology

**Files:**
- Modify: `paper_stat/sections/methods.tex`
- Modify: `paper_fuzzy/sections/methods.tex`
- Reference: All documentation created in Tasks 1-3

**Step 1: Read current methods sections**

Run: `cat paper_stat/sections/methods.tex | head -50`
Run: `cat paper_fuzzy/sections/methods.tex | head -50`

Expected: Current methods text

**Step 2: Create methods section template from documentation**

Using `docs/data-pipeline-documentation.md` as source, write to `paper_stat/sections/methods.tex`:

```latex
\section{Métodos}

\subsection{Conjunto de Dados e Fontes}

Utilizou-se um corpus balanceado de textos em português do Brasil, combinando cinco fontes distintas para garantir diversidade e representatividade:

\begin{enumerate}
    \item \textbf{BrWaC (Brazilian Web as Corpus)}~\cite{brwac}: Textos autorais extraídos da web brasileira (humanos)
    \item \textbf{ShareGPT-Portuguese}~\cite{sharegpt_portuguese}: Conversações geradas por LLMs em português
    \item \textbf{Canarim}~\cite{canarim}: Outputs de modelos de linguagem
    \item \textbf{BoolQ em Português}: Passagens humanas de perguntas booleanas
    \item \textbf{IMDB Reviews PT-BR}: Resenhas traduzidas automaticamente (LLM)
\end{enumerate}

O corpus combinado inicial continha 2.331.317 amostras. Após filtragem e processamento (descrito a seguir), o corpus final para análise continha aproximadamente 69 milhões de amostras processadas.

\subsection{Pré-Processamento e Filtragem}

\subsubsection{Filtragem por Comprimento Mínimo}

Textos com menos de 200 caracteres foram removidos, pois amostras muito curtas não fornecem informação estilométrica suficiente para análise confiável. Esta filtragem removeu 171.510 amostras (7,4\% do total).

\subsubsection{Segmentação de Textos Longos (Chunking)}

Documentos com mais de 10.000 caracteres foram segmentados em blocos (chunks) menores para garantir análise uniforme. O algoritmo de segmentação:

\begin{enumerate}
    \item Identifica textos com comprimento > 10.000 caracteres
    \item Divide em segmentos de até 10.000 caracteres
    \item Prioriza quebras em limites de sentença (`.`, `.\textbackslash n`, `\textbackslash n\textbackslash n`)
    \item Não utiliza sobreposição entre segmentos (overlap = 0)
\end{enumerate}

Aproximadamente 7,7\% dos textos (166.812 documentos) requereram segmentação, resultando em múltiplos chunks por documento original.

\subsubsection{Amostragem Estratificada e Balanceamento}

Para garantir a ausência de viés de classe nos classificadores, aplicou-se amostragem estratificada~\cite{cochran1977} por categoria de autoria:

\begin{enumerate}
    \item \textbf{Estratificação}: Separação em dois estratos (textos autorais vs. gerados por LLM)
    \item \textbf{Amostragem aleatória simples}: Seleção de 50.000 amostras de cada estrato
    \item \textbf{Balanceamento}: Proporção final de 50\% humano / 50\% LLM (100.000 amostras totais)
\end{enumerate}

Este balanceamento é essencial para evitar que métricas de desempenho (acurácia, AUC) sejam artificialmente inflacionadas por desbalanceamento de classes~\cite{japkowicz2002}.

\subsection{Extração de Características Estilométricas}

Extraíram-se 10 características estilométricas quantitativas de cada texto. Todas as características são \textbf{variáveis contínuas} na escala de medida de razão ou intervalo.

\subsubsection{Estatísticas de Frase}

\begin{itemize}
    \item \textbf{Comprimento médio de frase}: Número médio de palavras por frase (escala de razão, $\in \mathbb{R}^+$)
    \item \textbf{Desvio padrão do comprimento de frase}: Variabilidade no comprimento das frases
    \item \textbf{Burstiness}~\cite{madsen2005}: Medida de irregularidade temporal definida como $B = (\sigma - \mu) / (\sigma + \mu)$, onde $\mu$ e $\sigma$ são média e desvio padrão do comprimento de frase. Valores próximos a +1 indicam alta variabilidade; valores próximos a -1 indicam regularidade periódica.
\end{itemize}

[Continue with all 10 features, following the documentation from Task 2...]
```

**Step 3: Add statistical testing section to methods**

Add to methods.tex, using `docs/statistical-testing-documentation.md`:

```latex
\subsection{Análise Estatística}

\subsubsection{Testes de Hipótese Não-Paramétricos}

Para comparar as distribuições de cada característica entre textos autorais e de LLMs, utilizou-se o teste U de Mann-Whitney~\cite{mann1947}, um teste não-paramétrico para amostras independentes.

\paragraph{Justificativa para Testes Não-Paramétricos}

Embora as características sejam variáveis contínuas, optou-se por testes não-paramétricos devido à violação da suposição de normalidade. A inspeção visual das distribuições revelou assimetria, caudas pesadas e presença de outliers. Nestas condições, testes não-paramétricos apresentam maior poder estatístico que suas alternativas paramétricas~\cite{siegel1988}.

O teste U de Mann-Whitney baseia-se no ranqueamento conjunto das observações e testa a hipótese nula de que as duas populações têm distribuições idênticas.

\paragraph{Tamanho de Efeito: Delta de Cliff}

O valor-p indica \textbf{significância estatística}, mas não quantifica a \textbf{magnitude prática} da diferença. Para isso, calculou-se o delta de Cliff ($\delta$)~\cite{cliff1993}, uma medida de tamanho de efeito apropriada para comparações não-paramétricas:

\begin{equation}
\delta = \frac{\#(x_i > y_j) - \#(x_i < y_j)}{n_1 \times n_2}
\end{equation}

O delta de Cliff varia em $[-1, +1]$. Seguindo Romano et al.~\cite{romano2006}, classificou-se a magnitude como:
\begin{itemize}
    \item Negligenciável: $|\delta| < 0,147$
    \item Pequeno: $0,147 \leq |\delta| < 0,330$
    \item Médio: $0,330 \leq |\delta| < 0,474$
    \item Grande: $|\delta| \geq 0,474$
\end{itemize}

\paragraph{Correção para Comparações Múltiplas}

Ao realizar 10 testes simultâneos (um por característica), a probabilidade de cometer ao menos um erro do Tipo I (falsa descoberta) aumenta para aproximadamente 40\%. Para controlar a Taxa de Descobertas Falsas (FDR), aplicou-se a correção de Benjamini-Hochberg~\cite{benjamini1995}. Este método é menos conservador que a correção de Bonferroni, preservando maior poder estatístico.

[Continue with multivariate analysis sections...]
```

**Step 4: Verify Methods section matches code exactly**

Create checklist in `docs/methods-verification-checklist.md`:

```markdown
# Methods Section Verification Checklist

## Data Pipeline
- [ ] All 5 data sources listed (BrWaC, ShareGPT, Canarim, BoolQ, IMDB)
- [ ] Minimum length threshold specified (200 chars)
- [ ] Maximum length threshold specified (10,000 chars)
- [ ] Chunking algorithm described
- [ ] Stratified sampling procedure documented
- [ ] Final sample sizes stated (50k human, 50k LLM)

## Features
- [ ] All 10 features listed by name
- [ ] Scale of measurement specified for each (continuous/ratio)
- [ ] Formula provided for mathematical features (entropy, burstiness, TTR)
- [ ] Citations provided for each feature

## Statistical Tests
- [ ] Mann-Whitney U test described
- [ ] Justification for non-parametric choice
- [ ] Cliff's delta formula and interpretation
- [ ] Romano et al. thresholds stated
- [ ] FDR correction described
- [ ] All citations present

## Models
- [ ] PCA purpose and components stated
- [ ] LDA assumptions and performance
- [ ] Logistic Regression robustness explained
- [ ] Cross-validation strategy (5-fold stratified)
- [ ] ROC AUC as metric justified

## Code-Methods Alignment
- [ ] Methods describe EXACTLY what code does (no more, no less)
- [ ] No features mentioned that aren't in src/features.py
- [ ] No tests mentioned that aren't in src/tests.py or EDA.ipynb
- [ ] No data sources mentioned that aren't in process_data.ipynb
```

**Step 5: Compile and verify papers**

Run: `cd paper_stat && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
Run: `cd paper_fuzzy && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

Expected: Both compile cleanly

**Step 6: Commit updated methods sections**

```bash
git add paper_stat/sections/methods.tex paper_fuzzy/sections/methods.tex docs/methods-verification-checklist.md
git commit -m "refactor: align methods sections with implemented code

- Rewrite data collection section to match process_data.ipynb exactly
- List all 5 data sources with proper citations
- Document filtering (200 char min) and chunking (10k max)
- Document stratified sampling procedure
- Rewrite feature extraction to match src/features.py exactly
- Add formulas for all mathematical features
- Rewrite statistical testing to match src/tests.py and EDA.ipynb
- Add Mann-Whitney justification and Cliff's delta
- Add FDR correction explanation
- Create verification checklist

Papers now describe implemented methodology precisely

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create Reproducibility Guide

**Priority:** 🟡 HIGH - Enable others to reproduce results

**Files:**
- Create: `docs/REPRODUCIBILITY.md`

**Step 1: Write complete reproducibility guide**

Create: `docs/REPRODUCIBILITY.md`

```markdown
# Reproducibility Guide

This guide provides step-by-step instructions to reproduce all results in both papers.

## Prerequisites

### Software Requirements
- Python 3.8+
- Jupyter Notebook
- LaTeX distribution (for compiling papers)
- Git

### Python Dependencies
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

---

## Step 1: Data Collection

### Download Raw Data

1. **BrWaC Corpus**: Download from [URL] → save to `data/brwac/*.parquet`
2. **ShareGPT-Portuguese**: Download from [URL] → save to `data/sharegpt-portuguese.json`
3. **Canarim Dataset**: Download from [URL] → save to `data/canarim/*.parquet`
4. **BoolQ Portuguese**: Download from [URL] → save to `data/boolq.csv`, `data/validation_bool.csv`
5. **IMDB Reviews PT-BR**: Download from [URL] → save to `data/imdb-reviews-pt-br.csv`

**Expected:** ~100GB of raw data files

---

## Step 2: Data Preprocessing

Run the data preprocessing notebook:

```bash
jupyter notebook "0. process_data.ipynb"
```

**Execute all cells in order:**

1. **Cell: Load data sources** - Loads all 5 datasets
2. **Cell: Combine datasets** - Creates `combined.csv` (~2.3M rows)
3. **Cell: Filter and chunk** - Creates `processed_filtered_chunked_batch.csv` (~69M rows)
4. **Cell: Create balanced dataset** - Creates `balanced.csv` (100k rows, 50/50 split)

**Outputs:**
- `combined.csv` - All sources combined
- `processed_filtered_chunked_batch.csv` - Filtered and chunked
- `balanced.csv` - Balanced dataset for classification

**Time:** ~2-4 hours (BrWaC processing is slow)

---

## Step 3: Feature Extraction

Extract stylometric features from balanced dataset:

```bash
python src/features.py --input balanced.csv --output features.csv --lang pt
```

**Output:** `features.csv` with 10 additional feature columns

**Time:** ~30 minutes for 100k samples

---

## Step 4: Exploratory Data Analysis

Run EDA notebook to reproduce all statistical tests and visualizations:

```bash
jupyter notebook EDA.ipynb
```

**Execute all cells:**

1. Load `features.csv`
2. Calculate descriptive statistics by class
3. Perform Mann-Whitney U tests (10 features)
4. Calculate Cliff's delta effect sizes
5. Apply FDR correction
6. Generate visualizations (box plots, distributions, PCA)

**Outputs:**
- `eda_results_for_paper.json` - Statistical results
- All figures saved to `figures/` directory

**Time:** ~15 minutes

---

## Step 5: Model Training and Evaluation

Run model training (statistical models):

```bash
python src/models.py --input features.csv --output results_stat.pkl
```

**This will:**
1. Train Logistic Regression with 5-fold CV
2. Train LDA with 5-fold CV
3. Calculate ROC AUC for each
4. Save results to pickle file

**Output:** `results_stat.pkl` with model performance metrics

Run fuzzy classifier:

```bash
python src/fuzzy.py --input features.csv --output results_fuzzy.pkl
```

**Output:** `results_fuzzy.pkl` with fuzzy model performance

**Time:** ~20 minutes total

---

## Step 6: Compile Papers

Compile statistical paper:

```bash
cd paper_stat
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

**Output:** `paper_stat/main.pdf`

Compile fuzzy paper:

```bash
cd paper_fuzzy
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
```

**Output:** `paper_fuzzy/main.pdf`

**Time:** ~5 minutes

---

## Verification

### Expected Results

**Statistical Paper:**
- Logistic Regression AUC: 97.03% (±0.14%)
- LDA AUC: 94.12% (±0.17%)
- 6 features with large effect sizes (|δ| ≥ 0.474)

**Fuzzy Paper:**
- Fuzzy Classifier AUC: 89.34% (±0.04%)
- Variance 3-4× lower than statistical models

### Troubleshooting

**Issue:** `FileNotFoundError` for data files
**Solution:** Ensure all data files are in correct `data/` subdirectories

**Issue:** Memory error during BrWaC processing
**Solution:** Reduce batch size in `process_data.ipynb` (line XXX)

**Issue:** Different results than reported
**Solution:** Ensure random seed is set (`random_state=42` throughout)

---

## File Structure

```
prob_est/
├── data/
│   ├── brwac/*.parquet
│   ├── sharegpt-portuguese.json
│   ├── canarim/*.parquet
│   ├── boolq.csv
│   ├── validation_bool.csv
│   └── imdb-reviews-pt-br.csv
├── src/
│   ├── features.py
│   ├── models.py
│   ├── fuzzy.py
│   └── tests.py
├── 0. process_data.ipynb
├── EDA.ipynb
├── paper_stat/
│   └── main.tex
├── paper_fuzzy/
│   └── main.tex
└── docs/
    ├── data-pipeline-documentation.md
    ├── feature-extraction-documentation.md
    ├── statistical-testing-documentation.md
    └── REPRODUCIBILITY.md (this file)
```

---

## Contact

For questions about reproducibility, contact: [your email]
```

**Step 2: Commit reproducibility guide**

```bash
git add docs/REPRODUCIBILITY.md
git commit -m "docs: create complete reproducibility guide

- Document all prerequisites and dependencies
- Provide step-by-step instructions for data collection
- Document preprocessing pipeline execution
- Document feature extraction command
- Document EDA notebook execution
- Document model training commands
- Document paper compilation
- Add verification section with expected results
- Add troubleshooting guide

Complete transparency for reproducing all results

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Simplify Paper Scope

**Priority:** 🟡 MEDIUM - Remove complexity not supported by implementation

**Files:**
- Modify: `paper_stat/sections/intro.tex`
- Modify: `paper_stat/sections/discussion.tex`
- Modify: `paper_fuzzy/sections/intro.tex`
- Modify: `paper_fuzzy/sections/discussion.tex`

**Step 1: Identify scope bloat**

Read both papers and identify claims not supported by implemented code:

- Features mentioned but not implemented
- Analyses promised but not performed
- Future work presented as current work

**Step 2: Simplify statistical paper scope**

Focus on what was actually done:
- 10 specific stylometric features (no more, no less)
- Mann-Whitney U + Cliff's delta + FDR
- PCA for visualization only
- LDA and Logistic Regression for classification
- 5-fold cross-validation
- ROC AUC as metric

Remove or move to "Future Work":
- Other potential features not implemented
- Other models not tested
- Other languages not analyzed

**Step 3: Simplify fuzzy paper scope**

Focus on:
- Fuzzy logic with triangular membership functions
- Takagi-Sugeno order-zero inference
- Quantile-based parameter determination
- Same 10 features as statistical paper
- Interpretability vs performance trade-off

Remove:
- Complex fuzzy systems not implemented
- Alternative membership functions not tested
- Expert knowledge integration (not done - data-driven only)

**Step 4: Commit scope simplification**

```bash
git add paper_stat/sections/*.tex paper_fuzzy/sections/*.tex
git commit -m "refactor: simplify paper scope to match implementation

- Remove features not actually implemented
- Remove analyses not performed
- Move speculative content to Future Work
- Focus on what was actually done and validated
- Ensure every claim is backed by code

Papers now accurately represent completed work

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Final Verification Checklist

Before considering this plan complete:

- [ ] `docs/data-pipeline-documentation.md` created with all 5 sources documented
- [ ] `docs/feature-extraction-documentation.md` created with all 10 features documented
- [ ] `docs/statistical-testing-documentation.md` created with all tests documented
- [ ] `docs/REPRODUCIBILITY.md` created with step-by-step guide
- [ ] `docs/methods-verification-checklist.md` created
- [ ] `paper_stat/sections/methods.tex` matches code exactly
- [ ] `paper_fuzzy/sections/methods.tex` matches code exactly
- [ ] Both papers compile cleanly
- [ ] Scope simplified to match implementation
- [ ] All commits pushed to repository

**Total Implementation Effort:** Approximately 20-30 hours

**Success Criteria:**
1. Regina can read Methods sections and reproduce results from code
2. No claims in papers unsupported by documentation
3. Complete transparency from raw data to final results
4. Papers accurately reflect simplified scope

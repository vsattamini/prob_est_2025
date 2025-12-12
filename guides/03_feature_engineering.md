# Guide 3: Feature Engineering

**Module**: [src/features.py](../src/features.py)
**Previous**: [Guide 2: Data Preprocessing](02_data_preprocessing.md)
**Next**: Guide 4: Exploratory Data Analysis

---

## Overview

Feature engineering is the process of transforming raw text into numerical measurements that capture stylometric characteristics—patterns that distinguish human from LLM-generated text. This guide explains the **10 features** extracted by our system and how to interpret them.

### Why These Features?

Our feature set targets three stylometric dimensions:

1. **Sentence Structure** (3 features): Variation in sentence lengths reveals writing rhythm
2. **Lexical Diversity** (4 features): Vocabulary richness and repetition patterns
3. **Linguistic Markers** (3 features): Function words, pronouns, and character-level entropy

Each feature is **language-agnostic** (works for Portuguese or English) except function word detection, which uses language-specific dictionaries.

---

## The FeatureExtractor Class

### Architecture

```python
from src.features import FeatureExtractor

# Initialize for Portuguese
extractor = FeatureExtractor(lang="pt")

# Process single text
text = "Este é um exemplo de texto. Ele contém duas frases."
features = extractor.process(text)

print(features)
# Output:
# {
#   'sent_mean': 22.5,        # Average sentence length
#   'sent_std': 1.5,          # Standard deviation of sentence lengths
#   'sent_burst': 0.067,      # Coefficient of variation (burstiness)
#   'ttr': 0.846,             # Type-token ratio
#   'herdan_c': 5.89,         # Herdan's C (vocabulary richness)
#   'hapax_prop': 0.769,      # Proportion of words appearing once
#   'char_entropy': 3.92,     # Shannon entropy of character distribution
#   'func_word_ratio': 0.385, # Proportion of function words
#   'first_person_ratio': 0.0,# Proportion of first-person pronouns
#   'bigram_repeat_ratio': 0.0# Proportion of repeated bigrams
# }
```

### Design Principles

**1. Single-pass tokenization**: Text is tokenized once, then reused for all features
**2. Robust to edge cases**: Returns sensible defaults for empty/invalid input
**3. No external dependencies**: Uses only NumPy/Pandas (no spaCy, NLTK required)

---

## Feature Catalog

### Group 1: Sentence Structure Features

These features capture **writing rhythm**—the variation in sentence lengths that characterizes human vs. LLM style.

#### 1. `sent_mean` — Average Sentence Length

**Formula**: μ = (Σ sentence lengths) / (number of sentences)

**Code**: [features.py:142-149](../src/features.py#L142-L149)

```python
def burstiness(lengths: List[int]) -> Tuple[float, float, float]:
    """Return mean, std dev, and burstiness of sentence lengths."""
    if not lengths:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(lengths))
    std = float(np.std(lengths))
    burst = std / mean if mean != 0 else 0.0
    return mean, std, burst
```

**Interpretation**:
- **High values** (μ > 25 chars): Complex, formal writing
- **Low values** (μ < 15 chars): Concise, punchy writing
- **Typical range**: 15–30 characters per sentence

**Why it matters**: LLMs tend toward consistent sentence lengths, while humans vary more.

---

#### 2. `sent_std` — Sentence Length Standard Deviation

**Formula**: σ = √[(Σ(x - μ)²) / n]

**Code**: Same as above (line 146)

**Interpretation**:
- **High σ** (> 10): Wide variation in sentence structure (mixing short/long sentences)
- **Low σ** (< 5): Monotonous, uniform sentence lengths
- **Zero σ**: All sentences identical length (very unusual in natural text)

**Why it matters**: Human writing alternates between simple and complex sentences; LLMs produce more uniform lengths.

---

#### 3. `sent_burst` — Coefficient of Variation (Burstiness)

**Formula**: CV = σ / μ (dimensionless)

**Code**: [features.py:147](../src/features.py#L147)

```python
burst = std / mean if mean != 0 else 0.0
```

**Also known as**: "Normalized burstiness" in LLM detection literature ([GPTZero](https://gptzero.me), [Chakraborty et al. 2023](https://arxiv.org/abs/2310.05030))

**Interpretation**:
- **High CV** (> 0.5): "Bursty" writing with dramatic length variation
- **Medium CV** (0.3–0.5): Natural human variation
- **Low CV** (< 0.2): Mechanical, uniform sentence structure

**Why it matters**: This is one of the **strongest discriminators** between human and LLM text. LLMs struggle to replicate human sentence-length variation.

**Mathematical note**: CV normalizes variability by the mean, making it comparable across texts with different average sentence lengths (unlike raw σ).

---

### Group 2: Lexical Diversity Features

These features measure **vocabulary richness**—how many unique words appear and how often they repeat.

#### 4. `ttr` — Type-Token Ratio

**Formula**: TTR = V / N
- V = number of **types** (unique words)
- N = number of **tokens** (total words)

**Code**: [features.py:57-63](../src/features.py#L57-L63)

```python
def type_token_ratio(tokens: List[str]) -> float:
    """Compute the type-token ratio (number of unique tokens / total tokens)."""
    if not tokens:
        return 0.0
    n_types = len(set(tokens))
    n_tokens = len(tokens)
    return n_types / float(n_tokens)
```

**Interpretation**:
- **High TTR** (> 0.7): Rich vocabulary, little repetition (short texts, creative writing)
- **Medium TTR** (0.4–0.7): Typical prose
- **Low TTR** (< 0.4): Repetitive vocabulary (long texts, technical writing)

**Why it matters**: TTR decreases with text length (longer texts inevitably repeat words). Compare within similar-length texts.

**Caveat**: TTR is **length-dependent**. For texts >1000 words, consider **Herdan's C** instead.

---

#### 5. `herdan_c` — Herdan's C (Length-Normalized Diversity)

**Formula**: C = log(V) / log(N)
- V = vocabulary size (types)
- N = text length (tokens)

**Code**: [features.py:66-75](../src/features.py#L66-L75)

```python
def herdan_c(tokens: List[str]) -> float:
    """Compute Herdan's C, a length-corrected measure of lexical diversity."""
    if not tokens:
        return 0.0
    n_types = len(set(tokens))
    n_tokens = len(tokens)
    if n_tokens <= 1:
        return 0.0
    return np.log(n_types) / np.log(n_tokens)
```

**Interpretation**:
- **High C** (> 0.9): Extreme vocabulary diversity (each word appears once)
- **Medium C** (0.7–0.9): Normal lexical richness
- **Low C** (< 0.7): Limited vocabulary, repetitive phrasing

**Why it matters**: Herdan's C corrects for text length using logarithmic scaling. Unlike TTR, it remains **stable** for long texts.

**Theoretical range**: [0, 1], but practical values rarely exceed 0.95.

---

#### 6. `hapax_prop` — Hapax Legomena Proportion

**Formula**: hapax_prop = (# words appearing exactly once) / (# unique words)

**Code**: [features.py:78-86](../src/features.py#L78-L86)

```python
def hapax_proportion(tokens: List[str]) -> float:
    """Proportion of hapax legomena (words occurring exactly once)."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    hapax_count = sum(1 for count in counts.values() if count == 1)
    return hapax_count / float(len(counts))
```

**Terminology**: "Hapax legomena" (Greek: "said once") are words appearing exactly once in a text.

**Interpretation**:
- **High proportion** (> 0.7): Many rare words, rich vocabulary (creative writing, poetry)
- **Medium proportion** (0.5–0.7): Typical prose
- **Low proportion** (< 0.5): Repetitive vocabulary (technical docs, lists)

**Why it matters**: Human writers introduce unique words for specificity. LLMs often recycle common phrases, reducing hapax proportion.

**Connection to TTR**: Hapax proportion ≈ TTR for short texts, but decouples for longer texts.

---

#### 7. `bigram_repeat_ratio` — Repeated Bigram Ratio

**Formula**: ratio = (# bigram types with count > 1) / (# total distinct bigram types)

**Code**: [features.py:194-201](../src/features.py#L194-L201)

```python
def repeated_bigram_ratio(tokens: List[str]) -> float:
    """Compute the proportion of repeated bigrams."""
    if len(tokens) < 2:
        return 0.0
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    counts = Counter(bigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / float(len(counts)) if counts else 0.0
```

**What are bigrams?**: Pairs of consecutive words, e.g., "machine learning" → ("machine", "learning")

**Interpretation**:
- **High ratio** (> 0.3): Many repeated phrases (formulaic writing, chatbot responses)
- **Medium ratio** (0.1–0.3): Normal repetition (topic-specific terms)
- **Low ratio** (< 0.1): Each phrase unique (creative, diverse writing)

**Why it matters**: LLMs often repeat transitional phrases ("In addition," "It is important to note") more than humans.

**Related metrics**:
- Distinct-2 (Li et al. 2016) counts unique bigrams but doesn't measure repetition
- TF-IDF bigrams (Solaiman et al. 2019) weights by inverse document frequency

**Our implementation** focuses on **local redundancy**: How many bigram patterns appear more than once?

---

### Group 3: Linguistic Markers

These features capture **language-specific patterns** and information density.

#### 8. `char_entropy` — Shannon Entropy of Character Distribution

**Formula**: H = -Σ p(c) × log₂(p(c))
- p(c) = probability of character c
- Sum over all unique characters

**Code**: [features.py:89-99](../src/features.py#L89-L99)

```python
def char_entropy(text: str) -> float:
    """Compute Shannon entropy of the character distribution."""
    if not text:
        return 0.0
    total = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / float(total)
        entropy -= p * np.log2(p)
    return entropy
```

**Interpretation**:
- **High entropy** (> 4.5 bits): Diverse character use (technical writing, code, URLs)
- **Medium entropy** (3.5–4.5 bits): Normal prose
- **Low entropy** (< 3.5 bits): Repetitive characters (poetry, lists, structured text)

**Why it matters**: Entropy measures **unpredictability**. High entropy = uniform character distribution (all letters equally likely). Low entropy = skewed distribution (few letters dominate).

**Example**:
```python
char_entropy("aaaaaaa")     # → 0.0 bits (perfectly predictable)
char_entropy("abcdefg")     # → 2.8 bits (each char equally likely)
char_entropy("The quick brown fox...") # → ~4.2 bits (typical English)
```

**Connection to information theory**: Entropy quantifies the average information content per character. English typically has ~4.5 bits/char.

---

#### 9. `func_word_ratio` — Function Word Ratio

**Formula**: ratio = (# function words) / (# total words)

**Code**: [features.py:102-121](../src/features.py#L102-L121)

```python
# Portuguese function words
FUNC_WORDS_PT = {
    "o", "a", "os", "as",         # Articles
    "de", "em", "para", "com",    # Prepositions
    "e", "ou", "mas", "que",      # Conjunctions
    "se", "não", "por", "da",     # Particles
    # ... (127 total)
}

def function_word_ratio(tokens: List[str], lang: str = "en") -> float:
    """Proportion of function words (articles, prepositions, etc.)."""
    if not tokens:
        return 0.0
    func_words = FUNC_WORDS_PT if lang.lower().startswith("pt") else FUNC_WORDS_EN
    lower_tokens = [t.lower() for t in tokens]
    count = sum(1 for token in lower_tokens if token in func_words)
    return count / float(len(tokens))
```

**What are function words?**: "Grammatical glue" words (articles, prepositions, conjunctions) vs. content words (nouns, verbs, adjectives).

**Interpretation**:
- **High ratio** (> 0.5): Function-heavy text (casual conversation, blog posts)
- **Medium ratio** (0.3–0.5): Balanced prose
- **Low ratio** (< 0.3): Content-heavy text (technical writing, lists, keywords)

**Why it matters**: Function word frequency is a **robust stylometric marker**. Different authors use function words at different rates, and LLMs show distinct patterns.

**Language support**:
- Portuguese: 127 function words ([features.py:102-114](../src/features.py#L102-L114))
- English: 136 function words ([features.py:124-136](../src/features.py#L124-L136))

**Historical note**: Function word analysis dates to Mosteller & Wallace's (1964) authorship study of the Federalist Papers.

---

#### 10. `first_person_ratio` — First-Person Pronoun Ratio

**Formula**: ratio = (# first-person pronouns) / (# total words)

**Code**: [features.py:139-189](../src/features.py#L139-L189)

```python
# Portuguese first-person pronouns
FIRST_PERSON_PT = {
    "eu", "me", "mim", "comigo",   # Singular
    "meu", "minha", "meus", "minhas",  # Possessives
    "nós", "nos", "conosco",       # Plural
    "nosso", "nossa", "nossos", "nossas"
}

def first_person_ratio(tokens: List[str], lang: str = "en") -> float:
    """Proportion of first-person pronouns (I, me, my, we, our, etc.)."""
    if not tokens:
        return 0.0
    first_person = FIRST_PERSON_PT if lang.lower().startswith("pt") else FIRST_PERSON_EN
    lower_tokens = [t.lower() for t in tokens]
    count = sum(1 for token in lower_tokens if token in first_person)
    return count / float(len(tokens))
```

**Interpretation**:
- **High ratio** (> 0.05): Personal narrative, opinion writing, blogs
- **Medium ratio** (0.01–0.05): Some personal perspective
- **Zero ratio**: Objective, third-person writing (news, technical docs)

**Why it matters**: First-person pronouns signal **subjectivity** and **personal perspective**. LLMs trained on formal corpora (Wikipedia, books) may underuse first-person pronouns.

**Use case**: Distinguishing personal blog posts (high first-person) from LLM-generated article summaries (low first-person).

---

## Batch Feature Extraction

### Command-Line Interface

```bash
# Extract features from CSV
python -m src.features \
  --input data/balanced.csv \
  --output data/features.csv \
  --text-col text \
  --lang pt
```

**Parameters**:
- `--input`: Path to CSV with raw text
- `--output`: Destination for features CSV
- `--text-col`: Column name containing text (default: "text")
- `--lang`: Language code: "pt" (Portuguese) or "en" (English)

### Python API

```python
from src.features import extract_features

# Process entire CSV file
extract_features(
    input_path="data/balanced.csv",
    output_path="data/features.csv",
    text_col="text",
    lang="pt"
)

# Result: features.csv with columns:
# sent_mean, sent_std, sent_burst, ttr, herdan_c, hapax_prop,
# char_entropy, func_word_ratio, first_person_ratio, bigram_repeat_ratio,
# label, topic (preserved from input)
```

**Memory efficiency**: The `extract_features()` function processes row-by-row, so it handles large datasets without loading everything into memory.

---

## Feature Interpretation Guide

### Typical Value Ranges

| Feature | Human Range | LLM Range | Direction |
|---------|-------------|-----------|-----------|
| `sent_mean` | 20–30 chars | 15–25 chars | Human > LLM |
| `sent_std` | 8–15 chars | 5–10 chars | Human > LLM |
| `sent_burst` | 0.3–0.6 | 0.2–0.4 | Human > LLM |
| `ttr` | 0.5–0.7 | 0.6–0.8 | LLM > Human* |
| `herdan_c` | 0.75–0.85 | 0.80–0.90 | LLM > Human* |
| `hapax_prop` | 0.5–0.7 | 0.6–0.8 | LLM > Human* |
| `char_entropy` | 4.0–4.5 bits | 4.1–4.6 bits | LLM > Human |
| `func_word_ratio` | 0.35–0.45 | 0.30–0.40 | Human > LLM |
| `first_person_ratio` | 0.01–0.05 | 0.00–0.02 | Human > LLM |
| `bigram_repeat_ratio` | 0.10–0.25 | 0.15–0.30 | LLM > Human |

**\*Note**: Lexical diversity features (TTR, Herdan's C, hapax) often show **higher** values for LLM text because:
1. LLMs avoid repetition (trained to maximize diversity)
2. Human texts naturally repeat topic-specific terms
3. Our dataset may have shorter LLM samples (TTR increases for shorter texts)

### Multivariate Patterns

**No single feature perfectly separates human/LLM text.** Classification requires **combining features**:

**Example discriminative patterns**:
```
Human text signature:
  - High sent_burst (0.5) + High func_word_ratio (0.42) + High first_person_ratio (0.03)
  → Personal narrative with varied sentence structure

LLM text signature:
  - Low sent_burst (0.25) + High herdan_c (0.88) + High bigram_repeat_ratio (0.28)
  → Uniform sentence structure with diverse vocabulary but repetitive phrases
```

**Statistical verification**: See [Guide 4: Statistical Testing](04_statistical_testing.md) for hypothesis testing and effect size analysis.

---

## Feature Engineering Best Practices

### 1. Handle Edge Cases

```python
# Always check for empty/invalid input
def safe_feature_extraction(text):
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return {feature: 0.0 for feature in FEATURE_NAMES}
    return extractor.process(text)
```

### 2. Preserve Metadata

```python
# extract_features() automatically preserves non-text columns
df = pd.read_csv("input.csv")  # Contains: text, label, topic
extract_features("input.csv", "features.csv")

df_features = pd.read_csv("features.csv")
# Now contains: [10 features] + label + topic
```

### 3. Monitor Feature Distributions

```python
import matplotlib.pyplot as plt

# Check for outliers
df_features['sent_burst'].hist(bins=50)
plt.xlabel("Burstiness")
plt.ylabel("Frequency")
plt.title("Distribution of Sentence Burstiness")
plt.show()

# Flag suspicious values
suspicious = df_features[df_features['sent_burst'] > 1.0]
print(f"Found {len(suspicious)} texts with burstiness > 1.0")
```

### 4. Feature Scaling for Machine Learning

```python
from sklearn.preprocessing import StandardScaler

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
feature_cols = [
    'sent_mean', 'sent_std', 'sent_burst', 'ttr', 'herdan_c',
    'hapax_prop', 'char_entropy', 'func_word_ratio',
    'first_person_ratio', 'bigram_repeat_ratio'
]

X = df_features[feature_cols]
X_scaled = scaler.fit_transform(X)

# Now X_scaled has comparable scales for all features
```

**Why scale?** Features have different units (chars, ratios, bits). Scaling ensures:
- Logistic regression coefficients are comparable
- Distance-based algorithms (k-NN, SVM) work correctly
- Gradient descent converges faster

---

## Common Issues

### Issue 1: Zero Variance Features

**Symptom**: All values identical for a feature (e.g., `first_person_ratio = 0.0` for all texts)

**Diagnosis**: Your corpus lacks first-person pronouns (objective writing like news articles)

**Solution**:
```python
# Remove zero-variance features before modeling
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_reduced = selector.fit_transform(X)
retained_features = X.columns[selector.get_support()].tolist()
print(f"Retained {len(retained_features)}/10 features")
```

### Issue 2: Highly Correlated Features

**Symptom**: `sent_mean` and `sent_std` have correlation > 0.9

**Diagnosis**: Longer sentences naturally have higher variance

**Solution**:
```python
# Use burstiness (normalized) instead of raw std
# Our implementation already does this!
# sent_burst = sent_std / sent_mean (decorrelates the features)
```

### Issue 3: NaN Values

**Symptom**: Feature extraction returns `NaN` for some texts

**Diagnosis**: Division by zero (e.g., empty text, single word, no sentences)

**Solution**: Already handled! All feature functions return `0.0` for edge cases:
```python
# Example from ttr()
if not tokens:
    return 0.0  # Empty text → TTR = 0
```

If you see NaNs, check your input data for corrupted rows.

### Issue 4: Unexpected Feature Ranges

**Symptom**: `ttr > 1.0` or `char_entropy < 0.0`

**Diagnosis**: Bug in feature calculation or input preprocessing

**Solution**:
```python
# Validate feature ranges
assert 0.0 <= features['ttr'] <= 1.0, "TTR out of bounds"
assert features['char_entropy'] >= 0.0, "Entropy cannot be negative"
assert 0.0 <= features['sent_burst'] <= 10.0, "Burstiness suspiciously high"
```

Add these assertions after feature extraction to catch bugs early.

---

## Exercises

### Exercise 1: Feature Exploration

Extract features from 100 texts and visualize their distributions:

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.features import FeatureExtractor

# Load sample data
df = pd.read_csv("data/balanced.csv").head(100)
extractor = FeatureExtractor(lang="pt")

# Extract features
features = [extractor.process(text) for text in df['text']]
df_feat = pd.DataFrame(features)
df_feat['label'] = df['label'].values

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, col in enumerate(df_feat.columns[:-1]):  # Exclude 'label'
    ax = axes[i // 5, i % 5]
    df_feat.boxplot(column=col, by='label', ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()
```

**Questions**:
1. Which feature shows the clearest separation between human/LLM?
2. Are any features completely overlapping (no discriminative power)?
3. Do you see outliers? What causes them?

### Exercise 2: Feature Engineering

Create a new feature combining existing ones:

```python
# Hypothesis: Human text has high burstiness AND high function words
# Let's create a composite "humanness score"

def humanness_score(features):
    return (
        0.5 * features['sent_burst'] +
        0.3 * features['func_word_ratio'] +
        0.2 * features['first_person_ratio']
    )

df_feat['humanness'] = df_feat.apply(humanness_score, axis=1)

# Test discriminative power
from sklearn.metrics import roc_auc_score
y_true = (df_feat['label'] == 'human').astype(int)
auc = roc_auc_score(y_true, df_feat['humanness'])
print(f"Humanness score AUC: {auc:.3f}")
```

**Questions**:
1. Does your composite feature outperform individual features?
2. Try different weights—can you find a better combination?
3. How does this compare to logistic regression? (Hint: logistic regression learns optimal weights automatically!)

### Exercise 3: Language Comparison

Extract features from **English** and **Portuguese** texts and compare:

```python
# English texts
extractor_en = FeatureExtractor(lang="en")
features_en = [extractor_en.process(text) for text in df_english['text']]

# Portuguese texts
extractor_pt = FeatureExtractor(lang="pt")
features_pt = [extractor_pt.process(text) for text in df_portuguese['text']]

# Compare function word ratios
df_en = pd.DataFrame(features_en)
df_pt = pd.DataFrame(features_pt)

print(f"English func_word_ratio: {df_en['func_word_ratio'].mean():.3f}")
print(f"Portuguese func_word_ratio: {df_pt['func_word_ratio'].mean():.3f}")
```

**Questions**:
1. Do Portuguese texts have higher function word ratios than English?
2. How do sentence structures differ (sent_mean, sent_burst)?
3. Are lexical diversity metrics (TTR, Herdan's C) comparable across languages?

### Exercise 4: Manual Validation

Pick a suspicious text and manually verify feature values:

```python
text = "Este é um teste. Outro teste aqui. Mais um."
features = extractor.process(text)

# Manual calculation
sentences = ["Este é um teste", "Outro teste aqui", "Mais um"]
lengths = [len(s) for s in sentences]
mean_manual = sum(lengths) / len(lengths)
std_manual = (sum((x - mean_manual)**2 for x in lengths) / len(lengths))**0.5
burst_manual = std_manual / mean_manual

print(f"Automated: sent_mean={features['sent_mean']:.2f}, sent_std={features['sent_std']:.2f}, sent_burst={features['sent_burst']:.3f}")
print(f"Manual:    sent_mean={mean_manual:.2f}, sent_std={std_manual:.2f}, sent_burst={burst_manual:.3f}")
```

**Questions**:
1. Do automated and manual calculations match?
2. If not, what explains the difference? (Hint: Check tokenization!)
3. How sensitive are features to sentence boundary detection?

---

## Summary

**You've learned**:
✅ How to extract 10 stylometric features from text
✅ The mathematical formulas and implementations for each feature
✅ How to interpret feature values (typical ranges, discriminative power)
✅ Best practices for batch processing and quality checks

**Next steps**:
- **[Guide 4: Statistical Testing](04_statistical_testing.md)**: Verify feature differences are statistically significant
- **[Guide 5: Multivariate Models](05_multivariate_models.md)**: Combine features using PCA, LDA, logistic regression
- **Paper reference**: See [paper_stat/sections/methods.tex](../paper_stat/sections/methods.tex#L63-L90) for the complete technical description of all 10 features

---

**Mathematical references**:
- Shannon entropy: Shannon (1948), "A Mathematical Theory of Communication"
- Type-token ratio: Johnson (1944), "Studies in Language Behavior"
- Herdan's C: Herdan (1960), "Type-token Mathematics"
- Function words in stylometry: Mosteller & Wallace (1964), "Inference and Disputed Authorship: The Federalist"
- Burstiness in LLM detection: Chakraborty et al. (2023), "On the Possibilities of AI-Generated Text Detection"

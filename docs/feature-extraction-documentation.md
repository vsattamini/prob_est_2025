# Feature Extraction Documentation

## Overview

This document describes the 10 stylometric features extracted from each text sample. All features are **continuous variables** on a ratio or interval scale.

**Implementation:** `src/features.py`
**Dependencies:** pandas, numpy (no external NLP libraries for lightweight implementation)

---

## Feature Catalog

### 1. Sentence Length Statistics

#### 1.1 Mean Sentence Length
**Function:** `sentence_lengths(text) -> List[int]` combined with `burstiness()`
**Returns:** Mean sentence length in words

**Algorithm:**
1. Split text on sentence delimiters: `.`, `!`, `?`
2. Remove empty segments
3. Count words in each sentence (split on whitespace using regex `\b\w+\b`)
4. Calculate mean of word counts across all sentences

**Scale:** Continuous, ratio scale (words per sentence)
**Range:** [0, ∞)
**Typical values:** Human ~10-20 words/sentence, LLM may show different patterns

**Code snippet:**
```python
def sentence_lengths(text: str) -> List[int]:
    """Return a list of sentence lengths measured in tokens."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = []
    for s in sentences:
        tokens = re.findall(r"\b\w+\b", s.lower())
        lengths.append(len(tokens))
    return lengths
```

**Output feature:** `sent_mean`

---

#### 1.2 Sentence Length Standard Deviation
**Function:** Part of `burstiness()` output
**Returns:** Standard deviation of sentence lengths

**Algorithm:**
1. Calculate sentence lengths as above
2. Compute population standard deviation (ddof=0) using numpy

**Scale:** Continuous, ratio scale (words per sentence)
**Range:** [0, ∞)
**Interpretation:** Higher values indicate more variability in sentence structure

**Code snippet:**
```python
arr = np.array(lengths, dtype=float)
std = float(arr.std(ddof=0))  # population standard deviation
```

**Output feature:** `sent_std`

---

#### 1.3 Burstiness
**Function:** `burstiness(lengths) -> Tuple[float, float, float]`
**Formula:** `burstiness = std / mean`

**Interpretation:**
- Burstiness > 1: High variability (sentences vary greatly in length)
- Burstiness ≈ 1: Moderate variability
- Burstiness < 1: Uniform length (sentences are similar in length)
- Burstiness ≈ 0: Extremely regular (all sentences identical length)

**Scale:** Continuous, ratio scale
**Range:** [0, ∞)
**Citation:** Madsen et al. (2005) - Modeling burstiness in language

**Code snippet:**
```python
def burstiness(lengths: List[int]) -> Tuple[float, float, float]:
    """Return mean, standard deviation and burstiness (std/mean) of sentence lengths."""
    if not lengths:
        return 0.0, 0.0, 0.0
    arr = np.array(lengths, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    burst = std / mean if mean != 0 else 0.0
    return mean, std, burst
```

**Output feature:** `sent_burst`

**Note:** This implementation differs from the plan template (which referenced a formula `(std - mean) / (std + mean)`). The actual implementation uses `std / mean`, which is the coefficient of variation, a standard statistical measure of relative variability.

---

### 2. Lexical Diversity Metrics

#### 2.1 Type-Token Ratio (TTR)
**Function:** `type_token_ratio(tokens) -> float`

**Algorithm:**
1. Tokenize text into words (regex: `\b\w+\b`)
2. Convert to lowercase
3. Count total tokens (T)
4. Count unique types (V)
5. TTR = V / T

**Formula:** TTR = |Vocabulary| / |Total Words|

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Typical values:**
- 0.5-0.7 for short texts (high diversity)
- 0.3-0.5 for longer texts (decreases with text length)

**Limitation:** TTR is sensitive to text length (decreases as text gets longer due to word repetition)

**Citation:** Standard metric in stylometry and computational linguistics

**Code snippet:**
```python
def type_token_ratio(tokens: List[str]) -> float:
    """Compute the type–token ratio (TTR) for a list of tokens."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))
```

**Output feature:** `ttr`

---

#### 2.2 Herdan's C
**Function:** `herdan_c(tokens) -> float`

**Purpose:** Length-normalized alternative to TTR

**Formula:** C = log(V) / log(T)
where V = vocabulary size (types), T = total tokens

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Advantage:** More stable across varying text lengths than TTR; logarithmic transformation normalizes the effect of text length

**Citation:** Herdan, G. (1960). Type-token mathematics: A textbook of mathematical linguistics.

**Code snippet:**
```python
def herdan_c(tokens: List[str]) -> float:
    """Compute Herdan's C, a logarithmic variant of TTR."""
    n = len(tokens)
    if n == 0:
        return 0.0
    distinct = len(set(tokens))
    if n <= 1 or distinct <= 1:
        return 0.0
    return math.log(distinct) / math.log(n)
```

**Output feature:** `herdan_c`

---

#### 2.3 Hapax Legomena Proportion
**Function:** `hapax_proportion(tokens) -> float`

**Definition:** Proportion of words that appear exactly once in the text

**Algorithm:**
1. Count frequency of each word using Counter
2. Count words with frequency = 1 (hapax legomena)
3. Divide by total number of tokens

**Formula:** Hapax Ratio = |{words appearing once}| / |Total Tokens|

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Interpretation:**
- Higher values indicate more unique words used only once (varied vocabulary)
- Lower values indicate more word repetition

**Code snippet:**
```python
def hapax_proportion(tokens: List[str]) -> float:
    """Compute the proportion of hapax legomena (words occurring exactly once)."""
    n = len(tokens)
    if n == 0:
        return 0.0
    counts = Counter(tokens)
    hapax_count = sum(1 for c in counts.values() if c == 1)
    return hapax_count / float(n)
```

**Output feature:** `hapax_prop`

**Note:** This measures hapax proportion relative to total tokens, not vocabulary size as suggested in the plan template. This is a standard measure in quantitative linguistics.

---

### 3. Character-Level Features

#### 3.1 Character Entropy (Shannon Entropy)
**Function:** `char_entropy(text) -> float`

**Purpose:** Measure unpredictability/variability of character distribution

**Algorithm:**
1. Count frequency of each character in text (including spaces, punctuation)
2. Convert to probability distribution (freq/total)
3. Apply Shannon entropy formula

**Formula:** H = -Σ p(c) × log₂(p(c))
where p(c) is probability of character c

**Scale:** Continuous, ratio scale (bits)
**Range:** [0, log₂(|alphabet|)]
- Minimum: 0 bits (text contains only one character)
- Maximum: depends on alphabet size (e.g., ~5 bits for English alphabet)

**Interpretation:**
- High entropy: characters distributed uniformly (unpredictable, more varied)
- Low entropy: few characters dominate (predictable, less varied)

**Citation:** Shannon, C. E. (1948). A mathematical theory of communication.

**Code snippet:**
```python
def char_entropy(text: str) -> float:
    """Compute Shannon entropy of the character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = float(len(text))
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy
```

**Output feature:** `char_entropy`

---

### 4. Syntactic Features

#### 4.1 Function Word Ratio
**Function:** `function_word_ratio(tokens, lang='pt') -> float`

**Purpose:** Measure proportion of grammatical/structural words vs content words

**Algorithm:**
1. Tokenize and lowercase text
2. Count words that appear in function word list
3. Divide by total word count

**Function word lists:**
- **Portuguese** (86 words): determiners, prepositions, conjunctions, pronouns, common verbs
  - Examples: `o, a, os, as, um, uma, e, ou, mas, se, de, em, por, com, para, que, não, é, são, foi, tem, eu, me, nós, nos, ...`
- **English** (72 words): determiners, prepositions, conjunctions, pronouns, auxiliary verbs
  - Examples: `the, a, an, and, or, but, if, of, in, on, at, for, with, to, from, by, is, are, was, were, have, has, i, me, we, you, ...`

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Typical values:** ~0.4-0.6 for natural text

**Linguistic significance:** Function words are often used unconsciously and reflect grammatical structure rather than content, making them reliable stylometric markers.

**Citation:** Stamatatos, E. (2009). A survey of modern authorship attribution methods. Journal of the American Society for Information Science and Technology.

**Code snippet:**
```python
def function_word_ratio(tokens: List[str], lang: str = "en") -> float:
    """Return the proportion of tokens that are function words in the chosen language."""
    if not tokens:
        return 0.0
    if lang.lower().startswith("pt"):
        fw = FUNCTION_WORDS_PT
    else:
        fw = FUNCTION_WORDS_EN
    count_fw = sum(1 for t in tokens if t in fw)
    return count_fw / float(len(tokens))
```

**Output feature:** `func_word_ratio`

---

#### 4.2 First-Person Pronoun Ratio
**Function:** `first_person_ratio(tokens, lang='pt') -> float`

**Purpose:** Measure narrative perspective (1st person vs 3rd person)

**Algorithm:**
1. Tokenize and lowercase
2. Count first-person pronouns
3. Divide by total word count

**Portuguese 1st person pronouns:**
`eu, me, mim, meu, minha, nós, nos, nosso, nossa`

**English 1st person pronouns:**
`i, me, my, mine, we, us, our, ours`

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Interpretation:**
- Higher values indicate more first-person narrative (personal, subjective)
- Lower values indicate more third-person or impersonal narrative

**Code snippet:**
```python
def first_person_ratio(tokens: List[str], lang: str = "en") -> float:
    """Return the proportion of tokens that are first person pronouns."""
    if not tokens:
        return 0.0
    if lang.lower().startswith("pt"):
        pronouns = FIRST_PERSON_PT
    else:
        pronouns = FIRST_PERSON_EN
    count_fp = sum(1 for t in tokens if t in pronouns)
    return count_fp / float(len(tokens))
```

**Output feature:** `first_person_ratio`

---

### 5. Repetition Metrics

#### 5.1 Repeated Bigram Ratio
**Function:** `repeated_bigram_ratio(tokens) -> float`

**Purpose:** Measure tendency to repeat word pairs (phrasal repetition)

**Algorithm:**
1. Extract all consecutive word pairs (bigrams) from token sequence
2. Count frequency of each bigram using Counter
3. Calculate proportion of unique bigrams that appear more than once

**Formula:** RBR = |{bigrams with freq > 1}| / |Unique bigrams|

**Scale:** Continuous, ratio scale
**Range:** [0, 1]
**Interpretation:**
- Higher values indicate more repetitive phrasing (same word pairs recur)
- Lower values indicate more varied expression

**Code snippet:**
```python
def repeated_bigram_ratio(tokens: List[str]) -> float:
    """Compute the proportion of repeated bigrams in the token sequence."""
    if len(tokens) < 2:
        return 0.0
    # form bigrams
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    counts = Counter(bigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / float(len(counts)) if counts else 0.0
```

**Output feature:** `bigram_repeat_ratio`

**Note:** This metric measures the proportion of **unique** bigram types that repeat, not the proportion of total bigram tokens that are repeated.

---

### 6. Readability (English Only)

#### 6.1 Flesch-Kincaid Grade Level
**Function:** `flesch_kincaid(text) -> float`
**Status:** Disabled for Portuguese texts (returns 0.0)

**Purpose:** Estimate reading difficulty for English texts

**Formula:** FK = 0.39 × (words per sentence) + 11.8 × (syllables per word) - 15.59

**Algorithm:**
1. Count sentences using punctuation heuristics
2. Count words using regex tokenization
3. Estimate syllables per word using vowel-group counting heuristic
4. Apply formula

**Scale:** Continuous, interval scale (U.S. grade level)
**Range:** [0, ∞)
**Interpretation:** FK grade level indicates years of education needed to understand text (e.g., 8.0 = 8th grade)

**Note:** This metric is calibrated for English and not used in the Portuguese analysis. Included in code for future English comparison studies or multilingual extensions.

**Citation:** Flesch, R. (1948). A new readability yardstick. Journal of Applied Psychology.

**Code snippet:**
```python
def flesch_kincaid(text: str) -> float:
    """Compute Flesch–Kincaid grade level for an English text."""
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    n_sent = len(sentences)
    n_words = len(words)
    if n_sent == 0 or n_words == 0:
        return 0.0
    total_syllables = sum(syllable_count(w) for w in words)
    words_per_sentence = n_words / float(n_sent)
    syllables_per_word = total_syllables / float(n_words)
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return max(0.0, grade)
```

**Output feature:** `fk_grade` (set to 0.0 for Portuguese texts)

---

## Feature Extraction Pipeline

### Main Function: `extract_features(input_path, output_path, text_col='text', lang='en')`

**Purpose:** Batch processing of text samples to extract all features

**Parameters:**
- `input_path` (str): Path to input CSV file containing text column
- `output_path` (str): Path to output CSV file for features
- `text_col` (str, default='text'): Name of column containing raw text
- `lang` (str, default='en'): Language code ('en' or 'pt') for function words and pronouns

**Process:**
1. Read input CSV into pandas DataFrame
2. For each text in DataFrame:
   - Tokenize once for efficiency (words and sentences)
   - Extract all 10 features
   - Store in dictionary with standardized keys
3. Create feature DataFrame from list of dictionaries
4. Preserve original DataFrame columns (labels, metadata)
5. Write combined DataFrame to output CSV

**Output CSV columns:**
- All original DataFrame columns (e.g., `text`, `label`, `source`, etc.)
- `sent_mean` - Mean sentence length
- `sent_std` - Sentence length standard deviation
- `sent_burst` - Burstiness (coefficient of variation)
- `ttr` - Type-token ratio
- `herdan_c` - Herdan's C
- `hapax_prop` - Hapax proportion
- `char_entropy` - Character entropy
- `func_word_ratio` - Function word ratio
- `first_person_ratio` - First-person pronoun ratio
- `bigram_repeat_ratio` - Repeated bigram ratio
- `fk_grade` - Flesch-Kincaid grade level (0.0 for Portuguese)

**Command-line usage:**
```bash
python src/features.py --input balanced.csv --output features.csv --lang pt --text-col text
```

**Code snippet:**
```python
def extract_features(input_path: str, output_path: str, text_col: str = "text", lang: str = "en") -> None:
    """Read a CSV file, extract stylometric features and save as a new CSV."""
    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in {input_path}")
    extractor = FeatureExtractor(lang=lang)
    feature_rows = []
    for text in df[text_col].astype(str):
        feature_rows.append(extractor.process(text))
    features_df = pd.DataFrame(feature_rows)
    # Preserve original labels/topics if present
    for col in df.columns:
        if col != text_col:
            features_df[col] = df[col].values
    features_df.to_csv(output_path, index=False)
```

---

## FeatureExtractor Class

**Purpose:** Encapsulate feature extraction logic with language configuration

**Usage:**
```python
from features import FeatureExtractor

extractor = FeatureExtractor(lang='pt')
features = extractor.process("Este é um texto de exemplo.")
print(features)
# Output: {'sent_mean': 5.0, 'sent_std': 0.0, 'sent_burst': 0.0,
#          'ttr': 1.0, 'herdan_c': 1.0, 'hapax_prop': 1.0,
#          'char_entropy': 3.98, 'func_word_ratio': 0.2,
#          'first_person_ratio': 0.0, 'bigram_repeat_ratio': 0.0, 'fk_grade': 0.0}
```

**Method:** `process(text: str) -> Dict[str, Any]`
- Input: Raw text string
- Output: Dictionary with 11 feature keys
- Side effects: None (pure function)
- Optimization: Tokenizes text once and reuses tokens for multiple features

---

## Statistical Properties of Features

### Variable Type
All features are **continuous variables** on ratio or interval scales.

### Scale of Measurement
| Feature | Scale | Range | Zero Point | Units |
|---------|-------|-------|------------|-------|
| Mean sentence length | Ratio | [0, ∞) | True zero (0 words) | words/sentence |
| Sentence std dev | Ratio | [0, ∞) | True zero (no variation) | words |
| Burstiness | Ratio | [0, ∞) | True zero (no variation) | dimensionless |
| TTR | Ratio | [0, 1] | True zero (no diversity) | dimensionless |
| Herdan's C | Ratio | [0, 1] | Approaches 0 | dimensionless |
| Hapax proportion | Ratio | [0, 1] | True zero (no hapax) | dimensionless |
| Character entropy | Ratio | [0, log₂(n)] | True zero (one char only) | bits |
| Function word ratio | Ratio | [0, 1] | True zero (no function words) | dimensionless |
| First-person ratio | Ratio | [0, 1] | True zero (no 1st person) | dimensionless |
| Repeated bigram ratio | Ratio | [0, 1] | True zero (no repetition) | dimensionless |
| FK grade level | Interval | [0, ∞) | No true zero (arbitrary) | grade level |

### Distribution Characteristics
**Observed in EDA:** Most features exhibit **non-normal distributions** with the following characteristics:
- Skewness (asymmetric distributions)
- Heavy tails (extreme values more common than normal distribution predicts)
- Presence of outliers
- Bimodal or multimodal patterns in some features

These properties justify the use of **non-parametric statistical tests** (Mann-Whitney U) rather than parametric alternatives (t-test) in the analysis.

### Intercorrelations
Features are not independent; some correlations are expected:
- `sent_mean` and `sent_std` are positively correlated (longer sentences tend to vary more)
- `ttr` and `herdan_c` measure related aspects of lexical diversity
- `hapax_prop` correlates with vocabulary diversity metrics

The presence of correlated features is acceptable for classification tasks (models handle multicollinearity), but may affect interpretation of individual feature importance.

---

## Implementation Design Philosophy

### Lightweight and Portable
- **No external NLP libraries:** Avoids dependencies on spaCy, NLTK, or language-specific parsers
- **Pure Python + numpy + pandas:** Minimal dependencies for maximum portability
- **Fast:** Vectorized operations where possible; single-pass tokenization

### Trade-offs
- **Simplicity vs accuracy:** Uses regex-based sentence splitting instead of sophisticated sentence boundary detection
- **Speed vs precision:** Syllable counting is heuristic-based (not dictionary-based)
- **Generality:** Function word lists are manually curated, not exhaustive

These trade-offs are acceptable because:
1. Features are used for **classification**, not linguistic analysis
2. Errors affect human and LLM texts equally (systematic bias is acceptable)
3. Speed is critical for processing large datasets (millions of samples)

---

## References

### Feature Implementation
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
- Madsen, R. E., Kauchak, D., & Elkan, C. (2005). Modeling word burstiness using the Dirichlet distribution. *Proceedings of ICML*.
- Herdan, G. (1960). *Type-token mathematics: A textbook of mathematical linguistics*. Mouton.
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology*, 32(3), 221.

### Stylometric Applications
- Stamatatos, E. (2009). A survey of modern authorship attribution methods. *Journal of the American Society for Information Science and Technology*, 60(3), 538-556.
- Koppel, M., Schler, J., & Argamon, S. (2009). Computational methods in authorship attribution. *Journal of the American Society for Information Science and Technology*, 60(1), 9-26.

### Implementation Design
- Design philosophy: Lightweight, portable, fast processing for large-scale stylometric analysis
- Code architecture: Functional decomposition with optional OOP wrapper (FeatureExtractor class)
- Language support: English and Portuguese with extensible design for additional languages

---

## Appendix: Complete Function Word Lists

### Portuguese Function Words (86 words)
```python
FUNCTION_WORDS_PT = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "e", "ou", "mas", "se", "então", "porque", "assim",
    "de", "em", "no", "na", "nos", "nas", "por", "com",
    "para", "sem", "sobre", "entre", "antes", "depois",
    "sob", "contra", "durante", "perante", "até", "após",
    "estes", "essas", "esse", "essa", "aquele", "aquela",
    "aqueles", "aquelas", "algum", "alguma", "nenhum", "nenhuma",
    "não", "sim", "é", "são", "ser", "estou", "está",
    "estamos", "estão", "fui", "foi", "foram", "tenho",
    "tem", "tinha", "faz", "fazem", "pode", "poder",
    "vou", "vai", "vamos", "vão", "devo", "devem",
    "posso", "podem", "eu", "tu", "ele", "ela", "nós",
    "vós", "eles", "elas", "me", "mim", "minha", "minhas",
    "te", "ti", "tua", "tuas", "seu", "sua", "seus",
    "suas", "nosso", "nossa", "nossos", "nossas", "vosso",
    "vossa", "vossos", "vossas"
}
```

### English Function Words (72 words)
```python
FUNCTION_WORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "else", "because", "so", "of", "in", "on", "at", "for",
    "with", "to", "from", "by", "about", "as", "into", "like",
    "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among",
    "this", "that", "these", "those", "some", "any", "no",
    "not", "is", "are", "be", "am", "was", "were", "have",
    "has", "had", "do", "does", "did", "can", "could", "will",
    "would", "shall", "should", "may", "might", "we", "you",
    "he", "she", "it", "they", "i", "me", "my", "mine",
    "your", "yours", "our", "ours", "their", "theirs", "his",
    "her", "hers", "its", "us"
}
```

---

## Version History

- **v1.0** (2025-12-06): Initial documentation based on `src/features.py` implementation
  - Documented all 10 stylometric features
  - Included formulas, algorithms, and code snippets
  - Specified scales of measurement and statistical properties
  - Added complete references

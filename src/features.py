"""
Feature extraction for stylometric analysis.

This module defines a collection of functions that compute stylometric
features from text samples.  The `extract_features` function applies these
metrics to a pandas DataFrame and writes the result to CSV.  A command
line interface is provided via argparse.

The following metrics are implemented:

* **Sentence statistics:** mean length, standard deviation and burstiness
  (ratio of standard deviation to mean length).
* **Lexical diversity:** type–token ratio (TTR), Herdan's C, hapax
  proportion (proportion of words that appear exactly once).
* **Character entropy:** Shannon entropy of the character distribution.
* **Function word proportion:** fraction of words that belong to a list of
  English or Portuguese function words (use `--lang` to switch).
* **First person pronoun ratio:** share of tokens that are first person
  pronouns (e.g. "I", "me", "we", etc.).
* **N‑gram repetition:** proportion of repeated bigrams (can be extended).
* **Flesch–Kincaid grade level:** readability measure for English texts
  (disabled for Portuguese).

The module is intentionally light‑weight: it avoids external dependencies
beyond pandas and numpy.  All functions operate on plain Python lists or
strings and can be reused for other tasks.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

__all__ = [
    "tokenize_sentences",
    "tokenize_words",
    "sentence_lengths",
    "type_token_ratio",
    "herdan_c",
    "hapax_proportion",
    "burstiness",
    "char_entropy",
    "function_word_ratio",
    "first_person_ratio",
    "repeated_bigram_ratio",
    "flesch_kincaid",
    "extract_features",
]

# English and Portuguese function word lists.  These lists are not exhaustive,
# but they cover common determiners, conjunctions, prepositions and pronouns.
FUNCTION_WORDS_EN = set(
    [
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "because",
        "so", "of", "in", "on", "at", "for", "with", "to", "from", "by", "about",
        "as", "into", "like", "through", "after", "over", "between", "out", "against",
        "during", "without", "before", "under", "around", "among", "this", "that",
        "these", "those", "some", "any", "no", "not", "is", "are", "be", "am",
        "was", "were", "have", "has", "had", "do", "does", "did", "can", "could",
        "will", "would", "shall", "should", "may", "might", "we", "you", "he",
        "she", "it", "they", "i", "me", "my", "mine", "your", "yours", "our",
        "ours", "their", "theirs", "his", "her", "hers", "its", "us",
    ]
)

FUNCTION_WORDS_PT = set(
    [
        "o", "a", "os", "as", "um", "uma", "uns", "umas", "e", "ou", "mas", "se",
        "então", "porque", "assim", "de", "em", "no", "na", "nos", "nas", "por",
        "com", "para", "sem", "sobre", "entre", "antes", "depois", "sob", "contra",
        "durante", "perante", "até", "após", "estes", "essas", "esse", "essa", "aquele",
        "aquela", "aqueles", "aquelas", "algum", "alguma", "nenhum", "nenhuma",
        "não", "sim", "é", "são", "ser", "estou", "está", "estamos", "estão", "fui",
        "foi", "foram", "tenho", "tem", "tinha", "faz", "fazem", "pode", "poder",
        "vou", "vai", "vamos", "vão", "devo", "devem", "posso", "podem", "eu",
        "tu", "ele", "ela", "nós", "vós", "eles", "elas", "me", "mim", "minha",
        "minhas", "te", "ti", "tua", "tuas", "seu", "sua", "seus", "suas", "nosso",
        "nossa", "nossos", "nossas", "vosso", "vossa", "vossos", "vossas",
    ]
)

FIRST_PERSON_EN = set(["i", "me", "my", "mine", "we", "us", "our", "ours"])
FIRST_PERSON_PT = set(["eu", "me", "mim", "meu", "minha", "nós", "nos", "nosso", "nossa"])


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using simple punctuation heuristics."""
    # Split on punctuation that typically ends a sentence.  Keep question marks and
    # exclamation marks as delimiters.  Remove empty segments.
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> List[str]:
    """Split text into lowercase word tokens using a regex."""
    return re.findall(r"\b\w+\b", text.lower())


def sentence_lengths(text: str) -> List[int]:
    """Return a list of sentence lengths measured in tokens."""
    sentences = tokenize_sentences(text)
    lengths: List[int] = []
    for s in sentences:
        tokens = tokenize_words(s)
        lengths.append(len(tokens))
    return lengths


def type_token_ratio(tokens: List[str]) -> float:
    """Compute the type–token ratio (TTR) for a list of tokens."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))


def herdan_c(tokens: List[str]) -> float:
    """Compute Herdan's C, a logarithmic variant of TTR."""
    n = len(tokens)
    if n == 0:
        return 0.0
    distinct = len(set(tokens))
    # Avoid division by zero; log(1) == 0 so C=0 when distinct=1 or n=1
    if n <= 1 or distinct <= 1:
        return 0.0
    return math.log(distinct) / math.log(n)


def hapax_proportion(tokens: List[str]) -> float:
    """Compute the proportion of hapax legomena (words occurring exactly once)."""
    n = len(tokens)
    if n == 0:
        return 0.0
    counts = Counter(tokens)
    hapax_count = sum(1 for c in counts.values() if c == 1)
    return hapax_count / float(n)


def burstiness(lengths: List[int]) -> Tuple[float, float, float]:
    """Return mean, standard deviation and burstiness (std/mean) of sentence lengths."""
    if not lengths:
        return 0.0, 0.0, 0.0
    arr = np.array(lengths, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))  # population standard deviation
    burst = std / mean if mean != 0 else 0.0
    return mean, std, burst


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


def repeated_bigram_ratio(tokens: List[str]) -> float:
    """Compute the proportion of repeated bigrams in the token sequence."""
    if len(tokens) < 2:
        return 0.0
    # form bigrams
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    counts = Counter(bigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / float(len(counts)) if counts else 0.0


def syllable_count(word: str) -> int:
    """Estimate the number of syllables in an English word.

    This is a naive heuristic used solely for computing the Flesch–Kincaid
    grade level.  It counts vowel groups (a, e, i, o, u, y) and applies a
    few simple corrections for silent 'e' endings.
    """
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel
    # subtract one syllable if word ends with 'e' and has more than one syllable
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def flesch_kincaid(text: str) -> float:
    """Compute Flesch–Kincaid grade level for an English text.

    Returns zero for non‑English languages (there is no agreed FK formula
    for Portuguese).  The FK grade level is:

      0.39 * (words per sentence) + 11.8 * (syllables per word) - 15.59

    where sentences, words and syllables are estimated heuristically.
    """
    # If text is very short, return 0
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    n_sent = len(sentences)
    n_words = len(words)
    if n_sent == 0 or n_words == 0:
        return 0.0
    # Estimate syllables for each word
    total_syllables = sum(syllable_count(w) for w in words)
    words_per_sentence = n_words / float(n_sent)
    syllables_per_word = total_syllables / float(n_words)
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    return max(0.0, grade)


@dataclass
class FeatureExtractor:
    """A stateful class to extract features from text samples.

    This class encapsulates the language selection and exposes a single
    `process` method that returns a dictionary of feature values for a
    given text.  Instances can be reused to process multiple samples.
    """
    lang: str = "en"

    def process(self, text: str) -> Dict[str, Any]:
        # Tokenise once for reuse
        tokens = tokenize_words(text)
        sent_lengths = sentence_lengths(text)
        mean_len, std_len, burst = burstiness(sent_lengths)
        features = {
            "sent_mean": mean_len,
            "sent_std": std_len,
            "sent_burst": burst,
            "ttr": type_token_ratio(tokens),
            "herdan_c": herdan_c(tokens),
            "hapax_prop": hapax_proportion(tokens),
            "char_entropy": char_entropy(text),
            "func_word_ratio": function_word_ratio(tokens, self.lang),
            "first_person_ratio": first_person_ratio(tokens, self.lang),
            "bigram_repeat_ratio": repeated_bigram_ratio(tokens),
        }
        # Only compute Flesch–Kincaid for English texts
        if self.lang.lower().startswith("en"):
            features["fk_grade"] = flesch_kincaid(text)
        else:
            features["fk_grade"] = 0.0
        return features


def extract_features(input_path: str, output_path: str, text_col: str = "text", lang: str = "en") -> None:
    """Read a CSV file, extract stylometric features and save as a new CSV.

    Parameters
    ----------
    input_path: str
        Path to the input CSV containing at least a text column.
    output_path: str
        Destination path where the features CSV will be saved.
    text_col: str, default "text"
        Name of the column containing raw text.
    lang: str, default "en"
        Language code used for function words and pronouns.
    """
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


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract stylometric features from a CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV containing raw text.")
    parser.add_argument("--output", required=True, help="Path to output CSV where features will be saved.")
    parser.add_argument("--text-col", default="text", help="Name of the text column in the input file.")
    parser.add_argument("--lang", default="en", help="Language code ('en' or 'pt') for function words.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    extract_features(args.input, args.output, args.text_col, args.lang)


if __name__ == "__main__":
    main()
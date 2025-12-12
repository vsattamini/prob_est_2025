# Guide 1: Data Collection & Processing

**Learning Objectives**:
- Understand the 5 data sources used in this research
- Learn how to download and organize research data
- Understand data format specifications
- Perform quality assessment on text datasets

**Prerequisites**: Basic command line skills, understanding of CSV/JSON/Parquet formats

**Estimated Time**: 2-3 hours (including downloads)

---

## Table of Contents

1. [Overview of Data Sources](#1-overview-of-data-sources)
2. [Data Source Details](#2-data-source-details)
3. [Download Instructions](#3-download-instructions)
4. [Data Format Specifications](#4-data-format-specifications)
5. [Quality Assessment](#5-quality-assessment)
6. [File Organization](#6-file-organization)
7. [Exercises](#7-exercises)

---

## 1. Overview of Data Sources

This research uses **5 distinct data sources** to create a balanced corpus of human and LLM-generated Portuguese text.

### Data Sources Summary

| Source | Type | Size | Language | Label |
|--------|------|------|----------|-------|
| **BrWaC** | Web corpus | ~21 GB (21 files) | Portuguese (BR) | human |
| **BoolQ** | QA passages | ~2 MB | Portuguese | human |
| **ShareGPT-Portuguese** | Conversations | ~500 MB | Portuguese | llm |
| **IMDB Reviews** | Movie reviews (translated) | ~10 MB | Portuguese | llm |
| **Canarim-Instruct** | Instructions + outputs | ~3 GB | Portuguese | llm |

**Total Raw Data**: ~25 GB
**Final Balanced Dataset**: 100,000 samples (50,000 human + 50,000 LLM)

---

## 2. Data Source Details

### 2.1 BrWaC (Brazilian Web as Corpus) - HUMAN

**What it is**: Large-scale web corpus of Brazilian Portuguese text scraped from .br domains.

**Why we use it**: Represents authentic, naturally-occurring human-written Portuguese from diverse web sources (news, blogs, forums, etc.).

**Citation**: Wagner Filho, J. A., Wilkens, R., Idiart, M., & Villavicencio, A. (2018). The brWaC Corpus: A New Open Resource for Brazilian Portuguese. *Proceedings of LREC 2018*.

**Format**: 21 Parquet files containing structured data
```python
# Column structure:
{
    "url": str,           # Source URL
    "paragraphs": list,   # List of text paragraphs
    "doc_id": str,        # Unique document identifier
    "source": str         # Domain name
}
```

**Characteristics**:
- ✅ **High quality**: Web-scraped text with quality filters
- ✅ **Diverse genres**: News, blogs, educational, forums
- ✅ **Large scale**: Millions of documents
- ⚠️ **Size**: 21 GB requires significant disk space

**Processing needed**:
1. Extract paragraphs from nested JSON structure
2. Join paragraphs with newlines
3. Filter by length (200-10,000 characters)

---

### 2.2 BoolQ Portuguese - HUMAN

**What it is**: Boolean question-answering dataset with passage + question + answer structure. We use only the **passages** (human-written explanatory text).

**Why we use it**: High-quality, edited human writing with clear, informative style.

**Original source**: BoolQ (English) by Google
**Portuguese version**: Machine-translated with quality checks

**Format**: CSV with columns:
```python
{
    "question": str,    # Boolean question (not used)
    "passage": str,     # Source passage (USED)
    "answer": bool,     # True/False (not used)
    "title": str        # Article title
}
```

**Characteristics**:
- ✅ **High quality**: Wikipedia-derived passages
- ✅ **Informative style**: Explanatory, encyclopedia-like
- ✅ **Clean**: Pre-filtered, edited text
- ⚠️ **Homogeneous**: Single genre (encyclopedic)

---

### 2.3 ShareGPT-Portuguese - LLM

**What it is**: Real conversations between users and ChatGPT, shared via ShareGPT platform, filtered for Portuguese language.

**Why we use it**: Authentic LLM outputs from real usage scenarios.

**Format**: JSON with conversation structure:
```python
{
    "conversations": [
        {"from": "human", "value": "User message"},
        {"from": "gpt", "value": "ChatGPT response"}
    ],
    "id": str,
    "language": "pt"
}
```

**Characteristics**:
- ✅ **Authentic LLM**: Real ChatGPT outputs
- ✅ **Conversational**: Natural dialogue style
- ✅ **Diverse topics**: User questions span many domains
- ⚠️ **Privacy**: Public sharing implies consent

**Processing needed**:
1. Extract GPT responses only (discard human messages)
2. Join multi-turn responses if needed
3. Filter by length

---

### 2.4 IMDB Reviews Portuguese - LLM

**What it is**: Movie reviews from IMDB database, machine-translated from English to Portuguese.

**Why we use it**: Represents machine-translated text, which is a form of LLM-generated content.

**Format**: CSV with columns:
```python
{
    "id": int,
    "text_en": str,      # Original English (not used)
    "text_pt": str,      # Portuguese translation (USED)
    "sentiment": str     # positive/negative (not used)
}
```

**Characteristics**:
- ✅ **Machine-translated**: Modern MT systems (likely Google Translate or similar)
- ✅ **Opinion text**: Review/sentiment writing style
- ⚠️ **Translation artifacts**: May contain MT-specific patterns

**Processing needed**:
1. Drop unnecessary columns (id, text_en, sentiment)
2. Rename text_pt → text
3. Standard length filtering

---

### 2.5 Canarim-Instruct-PTBR - LLM

**What it is**: Instruction-following dataset where LLMs generate outputs based on prompts.

**Format**: Parquet files with structure:
```python
{
    "instruction": str,   # Task description
    "input": str,         # Additional context
    "output": str,        # LLM-generated response (USED)
    "model": str          # Which LLM generated it
}
```

**Characteristics**:
- ✅ **Instruction-following**: Task-oriented generation
- ✅ **Multiple LLMs**: May include outputs from different models
- ✅ **Formal style**: Often informative/educational tone
- ⚠️ **Large files**: Parquet format, multiple GB

**Processing needed**:
1. Extract 'output' column only
2. Batch processing due to size

---

## 3. Download Instructions

### Step 1: Create Data Directory

```bash
cd /home/vlofgren/Projects/mestrado/prob_est
mkdir -p data/brwac data/canarim
```

### Step 2: Download Each Source

#### BrWaC

```bash
# Download from official repository
wget -P data/brwac https://...  # (Full URL from BrWaC project page)

# Or if using Hugging Face Datasets:
python3 << EOF
from datasets import load_dataset
dataset = load_dataset("brwac")
dataset.save_to_disk("data/brwac")
EOF
```

**Expected**: 21 parquet files in `data/brwac/`

#### BoolQ

```bash
wget -P data https://storage.googleapis.com/boolq/train.jsonl
wget -P data https://storage.googleapis.com/boolq/dev.jsonl
```

**Convert to Portuguese** (if not already):
```python
# Translation script (if needed)
from deep_translator import GoogleTranslator
import pandas as pd

df = pd.read_json('data/train.jsonl', lines=True)
translator = GoogleTranslator(source='en', target='pt')
df['passage_pt'] = df['passage'].apply(translator.translate)
df.to_csv('data/boolq_pt.csv', index=False)
```

#### ShareGPT-Portuguese

```bash
# Check Hugging Face or ShareGPT archives
wget -P data https://huggingface.co/datasets/.../sharegpt_portuguese.json
```

#### IMDB Reviews

```bash
# Download from Kaggle or similar source
# Already translated to Portuguese
wget -P data https://.../imdb_reviews_pt.csv
```

#### Canarim

```bash
# Download from Hugging Face
wget -P data/canarim https://huggingface.co/datasets/.../canarim_instruct.parquet
```

---

## 4. Data Format Specifications

### 4.1 Expected Directory Structure

After downloading, your directory should look like:

```
data/
├── brwac/
│   ├── part_00.parquet
│   ├── part_01.parquet
│   ├── ... (21 files total)
│   └── part_20.parquet
├── canarim/
│   ├── train_00.parquet
│   ├── train_01.parquet
│   └── ... (multiple files)
├── boolq_pt.csv
├── sharegpt_portuguese.json
└── imdb_reviews_pt.csv
```

### 4.2 Column Requirements

Each source after initial processing should have:
- `text` column: The actual text content
- `label` column: Either "human" or "llm"
- `source` column: Dataset name (e.g., "brwac", "sharegpt")

---

## 5. Quality Assessment

### 5.1 Check File Integrity

```bash
# Verify all files downloaded
ls -lh data/brwac/*.parquet | wc -l  # Should be 21
ls -lh data/canarim/*.parquet | wc -l # Should be multiple files
ls -lh data/*.csv data/*.json
```

### 5.2 Quick Data Inspection

```python
import pandas as pd

# Check BoolQ
df_bool = pd.read_csv('data/boolq_pt.csv')
print(f"BoolQ shape: {df_bool.shape}")
print(df_bool.head())

# Check ShareGPT
import json
with open('data/sharegpt_portuguese.json') as f:
    data = json.load(f)
print(f"ShareGPT entries: {len(data)}")

# Check IMDB
df_imdb = pd.read_csv('data/imdb_reviews_pt.csv')
print(f"IMDB shape: {df_imdb.shape}")
```

### 5.3 Text Length Distribution

```python
# Check typical text lengths
texts = df_bool['passage']  # or whatever column
lengths = texts.str.len()
print(f"Length statistics:")
print(lengths.describe())
print(f"Texts < 200 chars: {(lengths < 200).sum()}")
print(f"Texts > 10000 chars: {(lengths > 10000).sum()}")
```

**Expected**:
- Mean length: 500-2000 characters
- Some very short texts (will be filtered)
- Some very long texts (will be chunked)

---

## 6. File Organization

### 6.1 Organize by Type

Create separate folders for raw vs processed data:

```bash
mkdir -p data/raw data/processed
mv data/*.csv data/*.json data/raw/
mv data/brwac data/canarim data/raw/
```

### 6.2 Documentation

Create a data manifest:

```bash
cat > data/DATA_MANIFEST.md << 'EOF'
# Data Manifest

## Raw Data Sources

1. BrWaC: `raw/brwac/*.parquet` (21 files, ~21 GB)
2. BoolQ: `raw/boolq_pt.csv` (~2 MB)
3. ShareGPT: `raw/sharegpt_portuguese.json` (~500 MB)
4. IMDB: `raw/imdb_reviews_pt.csv` (~10 MB)
5. Canarim: `raw/canarim/*.parquet` (~3 GB)

## Processed Data (after pipeline)

- `processed/combined.csv` - All sources combined
- `processed/filtered_chunked.csv` - After filtering & chunking
- `processed/balanced.csv` - Final balanced dataset (100k samples)

## Download Dates

- BrWaC: 2024-XX-XX
- BoolQ: 2024-XX-XX
- (etc.)
EOF
```

---

## 7. Exercises

### Exercise 1: Verify Download Completeness

**Task**: Write a script that checks all required files exist and reports their sizes.

```python
import os

required_files = {
    'brwac': ('data/raw/brwac', 21, '.parquet'),
    'boolq': ('data/raw/boolq_pt.csv', 1, '.csv'),
    'sharegpt': ('data/raw/sharegpt_portuguese.json', 1, '.json'),
    'imdb': ('data/raw/imdb_reviews_pt.csv', 1, '.csv'),
    'canarim': ('data/raw/canarim', 'multiple', '.parquet'),
}

# TODO: Implement verification function
def verify_downloads():
    pass
```

### Exercise 2: Sample Inspection

**Task**: Extract 10 random samples from each source and visually inspect for quality issues.

**What to look for**:
- Encoding issues (mojibake characters)
- HTML/XML tags not removed
- Very short or very long texts
- Language mixing (Portuguese + other)

### Exercise 3: Length Distribution Analysis

**Task**: Create histograms showing text length distribution for each source.

```python
import matplotlib.pyplot as plt

# TODO: Plot length distributions
# Compare across sources
# Identify outliers
```

---

## Next Steps

Once data collection is complete, proceed to:
- **[Guide 2: Data Preprocessing](./02_preprocessing.md)** - Filtering, chunking, and balancing

---

## Summary Checklist

- [ ] All 5 data sources downloaded
- [ ] File integrity verified (no corrupted files)
- [ ] Sample inspection completed
- [ ] Length distributions analyzed
- [ ] Data organized in proper directory structure
- [ ] DATA_MANIFEST.md created
- [ ] Ready to proceed to preprocessing

---

## Common Issues & Solutions

### Issue 1: BrWaC files too large for memory

**Solution**: Process in batches using pandas chunking:
```python
for chunk in pd.read_parquet('file.parquet', chunksize=10000):
    process(chunk)
```

### Issue 2: Character encoding errors

**Solution**: Specify encoding explicitly:
```python
pd.read_csv('file.csv', encoding='utf-8')
```

### Issue 3: Download timeouts

**Solution**: Use `wget` with retry:
```bash
wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 0 URL
```

---

## References

- BrWaC: Wagner Filho et al. (2018)
- BoolQ: Clark et al. (2019)
- Dataset documentation: See `docs/data-pipeline-documentation.md`


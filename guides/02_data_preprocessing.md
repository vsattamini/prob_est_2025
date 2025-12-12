# Guide 2: Data Preprocessing

**Purpose**: Transform raw data from 5 sources into a clean, balanced dataset ready for feature extraction
**Notebook**: [`0. process_data.ipynb`](../0.%20process_data.ipynb)
**Input**: Raw data files from 5 sources
**Output**: `balanced_processed.csv` - Clean, balanced dataset (~1.2M samples)

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources Integration](#data-sources-integration)
3. [Text Filtering & Chunking](#text-filtering--chunking)
4. [Dataset Balancing](#dataset-balancing)
5. [Memory-Efficient Processing](#memory-efficient-processing)
6. [Quality Checks](#quality-checks)
7. [Common Issues](#common-issues)
8. [Exercises](#exercises)

---

## Overview

### What is Data Preprocessing?

Data preprocessing transforms raw, heterogeneous text data into a uniform format suitable for analysis. This involves:

1. **Integration**: Combining data from multiple sources
2. **Cleaning**: Removing invalid or too-short texts
3. **Normalization**: Handling variable-length texts via chunking
4. **Balancing**: Creating equal representation of human/LLM texts

### Why This Matters

**Without preprocessing**:
- ❌ Different file formats (JSON, CSV, Parquet) cause errors
- ❌ Very short texts (< 100 chars) lack stylometric signal
- ❌ Very long texts (> 10,000 chars) dominate statistics
- ❌ Imbalanced data (70% human, 30% LLM) leads to biased models

**After preprocessing**:
- ✅ Uniform CSV format
- ✅ All texts 100-10,000 characters (optimal for stylometry)
- ✅ Long texts split into manageable chunks
- ✅ 50/50 human/LLM balance prevents class bias

---

## Data Sources Integration

### Step 1: Load Individual Sources

Each source has unique format - we standardize to `{'label': str, 'text': str}`:

```python
# Source 1: ShareGPT-Portuguese (JSON)
df = pd.read_json("data/sharegpt-portuguese.json")
source = ['llm' if label == 'gpt' else label for label in source]

# Source 2: IMDB Reviews (CSV)
df_1 = pd.read_csv("data/imdb-reviews-pt-br.csv")
df_1 = df_1.rename(columns={"text_pt": "text"})
df_1['label'] = 'llm'  # Translated by model

# Source 3 & 4: BoolQ (CSV - human-written passages)
df_2 = pd.read_csv("data/boolq.csv")
label = ['human' for _ in range(len(df_2))]
text = [row['passage'] for _, row in df_2.iterrows()]

# Source 5: BrWaC (Parquet - large corpus)
parquet_files = glob.glob("data/brwac/*.parquet")
for parquet_file in parquet_files:
    parquet_obj = pq.ParquetFile(parquet_file)
    for batch in parquet_obj.iter_batches(batch_size=100000):
        # Process paragraphs into single text
        processed_text = process_paragraphs_to_text(row['text'])

# Source 6: Canarim (Parquet - LLM-generated)
parquet_files = glob.glob("data/canarim/*.parquet")
df_5 = pd.read_parquet(parquet_file)
labels.append('llm')
text.append(row['output'])
```

**Key Function**: `process_paragraphs_to_text()`
```python
def process_paragraphs_to_text(data_dict):
    """
    BrWaC stores text as {'paragraphs': [['text1'], ['text2'], ...]}
    This flattens it to a single text block.
    """
    all_text = []
    for paragraph in data_dict['paragraphs']:
        if isinstance(paragraph, list):
            paragraph_text = ' '.join(paragraph)
        else:
            paragraph_text = str(paragraph)
        all_text.append(paragraph_text)

    return '\n'.join(all_text)  # Join paragraphs with newlines
```

### Step 2: Combine Sources

```python
# Concatenate all DataFrames
df_combined = pd.concat([df, df_1, df_2, df_3, df_4, df_5], ignore_index=True)

# Save combined dataset
df_combined.to_csv("combined.csv", index=False)
```

**Result**: ~2.3M samples from 5 sources

**Label Distribution Before Balancing**:
```
human:  1,650,000 (70.8%)
llm:      681,317 (29.2%)
Total:  2,331,317
```

---

## Text Filtering & Chunking

### The Problem: Variable Text Lengths

**Distribution of text lengths**:
- Min: 1 character (empty or near-empty)
- Max: 150,000+ characters (entire articles)
- Mean: ~1,200 characters
- Median: ~450 characters

**Issues**:
1. **Too short** (< 100 chars): Insufficient for stylometric analysis
   - Example: "Sim." (3 chars) - can't extract 10 features reliably

2. **Too long** (> 10,000 chars): Computationally expensive, dominate statistics
   - Example: 50,000-char article creates outliers in all metrics

### Solution: Filter + Chunk

**Step 1: Filter Short Texts**
```python
min_length = 100  # Characters
df_filtered = df[df['text'].str.len() >= min_length]
```
**Removed**: 171,510 texts (7.4%)

**Step 2: Split Long Texts**
```python
max_length = 10,000  # Characters
chunk_overlap = 0    # No overlap between chunks

def create_text_chunks(text, max_length, overlap):
    """
    Split text into chunks at natural breakpoints.

    Strategy:
    1. Try to break at sentence end ('. ')
    2. Fall back to paragraph break ('\n\n')
    3. Last resort: space character
    4. Ensure we don't break too early (at least half max_length)
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length

        if end < len(text):
            # Find best break point
            break_points = ['. ', '.\n', '\n\n', ' ']
            best_break = end

            for break_char in break_points:
                last_break = text.rfind(break_char, start, end)
                if last_break > start + max_length // 2:
                    best_break = last_break + len(break_char)
                    break

            chunk = text[start:best_break].strip()
        else:
            chunk = text[start:].strip()

        if chunk:
            chunks.append(chunk)

        start = max(start + max_length - overlap, best_break - overlap)

    return chunks
```

**Example**:
```
Original text (15,000 chars):
  "This is a long article about AI. It has many paragraphs..."

After chunking → 2 chunks:
  Chunk 1 (9,500 chars): "This is a long article... first section."
  Chunk 2 (5,500 chars): "The second section continues... end."
```

**Metadata Added**:
- `original_length`: Length before chunking
- `chunk_id`: Unique identifier (e.g., `"1234_0"`, `"1234_1"`)

**Results**:
- **Normal texts** (100-10,000 chars): 1,992,995 (kept as-is)
- **Chunked texts**: 166,812 → created ~350,000 chunks
- **Final dataset**: ~2.3M samples

---

## Dataset Balancing

### Why Balance?

**Imbalanced Data Problem**:
```
Training data:  70% human, 30% LLM
Model learns:   "Just predict 'human' and be right 70% of the time"
Result:         High accuracy (70%) but useless classifier
```

**Balanced Data Solution**:
```
Training data:  50% human, 50% LLM
Model learns:   Actual stylometric differences
Result:         True discrimination ability
```

### Hybrid Balancing Strategy

**Goal**: Create 30% sample with 50/50 balance

```python
def hybrid_balance(df, target_ratio=0.3):
    """
    Hybrid approach: Downsample majority + Upsample minority

    Strategy:
    1. Count samples: Human (1,992,995), LLM (367,812)
    2. Calculate target: 30% of total = ~708,000 samples
    3. Target per class: 354,000 each
    4. Downsample human: 1,992,995 → 354,000 (random sample)
    5. Upsample LLM: 367,812 → 354,000 (sample with replacement)
    6. Shuffle combined dataset
    """

    human_samples = df[df['label'] == 'human']
    llm_samples = df[df['label'] == 'llm']

    target_size = int((len(human_samples) + len(llm_samples)) * target_ratio)

    # Downsample human (plenty available)
    human_balanced = human_samples.sample(n=target_size, random_state=42)

    # Upsample LLM (need more via resampling)
    llm_upsampled = llm_samples.sample(
        n=target_size,
        replace=True,  # Allow duplicates
        random_state=42
    )

    # Combine and shuffle
    df_balanced = pd.concat([human_balanced, llm_upsampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced
```

**Result**:
```
Balanced dataset: 708,000 samples
  human: 354,000 (50.0%)
  llm:   354,000 (50.0%)
```

**Upsampling Caveat**: Some LLM samples appear 2-3 times due to replacement. This is acceptable because:
1. LLM samples are diverse (from multiple models/sources)
2. Statistical tests remain valid (samples are independent draws)
3. Alternative (discarding human data) wastes valuable samples

---

## Memory-Efficient Processing

### The Challenge

**Problem**: Processing 2.3M samples × ~1KB/sample = ~2.3GB RAM
- With chunking overhead: ~4.6GB RAM
- Plus pandas overhead: ~7GB total
- Many machines have < 8GB RAM available

### Solution: Batch Processing

**Strategy**: Process data in chunks, save intermediate results

```python
def filter_and_chunk_text_batch(input_file, output_file,
                                batch_size=50000,
                                intermediate_save_every=5):
    """
    Process large CSV files in batches to avoid memory issues.

    Workflow:
    1. Read 50,000 rows (batch)
    2. Filter short texts
    3. Chunk long texts
    4. Save intermediate result every 5 batches
    5. Clear memory and continue
    6. Combine all intermediate files at end
    """

    batch_number = 0
    processed_batches = []

    # Read file in batches
    chunk_iter = pd.read_csv(input_file, chunksize=batch_size)

    for batch_df in tqdm(chunk_iter, desc="Processing batches"):
        batch_number += 1

        # Process current batch
        batch_result = process_single_batch(
            batch_df,
            min_length=100,
            max_length=10000,
            chunk_overlap=0
        )

        processed_batches.append(batch_result['data'])

        # Intermediate save
        if batch_number % intermediate_save_every == 0:
            intermediate_file = f"intermediate_batch_{batch_number}.csv"
            combined = pd.concat(processed_batches, ignore_index=True)
            combined.to_csv(intermediate_file, index=False)

            # Clear memory
            del combined
            processed_batches = []
            gc.collect()

    # Combine all intermediate files
    # (code to merge intermediate results)
```

**Memory Estimation**:
```python
def estimate_memory_usage(input_file, batch_size=50000):
    """
    Predict memory usage before processing.
    """
    # Read 1000-row sample
    sample = pd.read_csv(input_file, nrows=1000)
    avg_row_size = sample.memory_usage(deep=True).sum() / len(sample)

    # Estimate batch memory with chunking overhead
    batch_memory_mb = (avg_row_size * batch_size * 2.0) / (1024**2)

    print(f"Estimated batch memory: {batch_memory_mb:.1f} MB")

    if batch_memory_mb > 2000:  # > 2GB
        print("WARNING: Consider reducing batch_size to 25000")
```

**Best Practices**:
1. **Start with memory estimation** - know before you run
2. **Save intermediate results** - don't lose progress if crash
3. **Use garbage collection** - `gc.collect()` after each batch
4. **Monitor progress** - `tqdm` shows time remaining

---

## Quality Checks

### Post-Processing Validation

**Always verify**:

```python
def analyze_processed_results(processed_file):
    """
    Validate preprocessing results without loading full dataset.
    """
    total_rows = 0
    label_counts = {'human': 0, 'llm': 0}
    chunk_counts = 0
    length_stats = []

    # Read in small chunks
    for chunk in pd.read_csv(processed_file, chunksize=10000):
        total_rows += len(chunk)
        label_counts.update(chunk['label'].value_counts())
        chunk_counts += len(chunk[chunk['chunk_id'] != ''])
        length_stats.extend(chunk['text'].str.len().tolist()[:5000])

    # Report
    print(f"Total rows: {total_rows:,}")
    print(f"Chunked entries: {chunk_counts:,} ({chunk_counts/total_rows*100:.1f}%)")
    print(f"Label balance: {label_counts}")
    print(f"Length: min={min(length_stats)}, max={max(length_stats)}, "
          f"mean={np.mean(length_stats):.0f}")
```

**Expected Outputs**:
```
✅ Total rows: ~2,340,000
✅ Chunked entries: ~350,000 (15%)
✅ Labels: human ≈ llm (if balanced)
✅ Length: min=100, max=10,000, mean=~800
```

### Sanity Checks

**1. No empty texts**:
```python
assert (df['text'].str.len() > 0).all(), "Found empty texts!"
```

**2. All labels valid**:
```python
assert set(df['label'].unique()) == {'human', 'llm'}, "Invalid labels!"
```

**3. Length constraints**:
```python
lengths = df['text'].str.len()
assert lengths.min() >= 100, "Texts too short!"
assert lengths.max() <= 10000, "Texts too long!"
```

**4. No duplicates** (if desired):
```python
duplicates = df.duplicated(subset=['text']).sum()
print(f"Duplicate texts: {duplicates} ({duplicates/len(df)*100:.2f}%)")
```

---

## Common Issues

### Issue 1: Out of Memory

**Symptoms**: Python crashes with `MemoryError` or `Killed`

**Solutions**:
1. **Reduce batch size**: 50,000 → 25,000 or 10,000
2. **Close other programs**: Free up RAM
3. **Use batch processing**: Don't load entire dataset
4. **Process on server**: If available, use machine with more RAM

```python
# If 50K batch fails, try 25K
filter_and_chunk_text_batch(
    input_file="combined.csv",
    batch_size=25000,  # Reduced
    intermediate_save_every=3  # Save more frequently
)
```

### Issue 2: Parquet Files Too Large

**Symptoms**: BrWaC processing takes hours or crashes

**Solution**: Process in smaller batches within each file

```python
# Process each parquet file in batches
for parquet_file in parquet_files:
    pq_file = pq.ParquetFile(parquet_file)

    for batch in pq_file.iter_batches(batch_size=100000):
        df_batch = batch.to_pandas()
        # Process batch
        del df_batch, batch  # Free memory immediately
        gc.collect()
```

### Issue 3: Text Encoding Errors

**Symptoms**: `UnicodeDecodeError` when reading CSV

**Solution**: Specify encoding explicitly

```python
# Try UTF-8 first
df = pd.read_csv("data/file.csv", encoding='utf-8')

# If fails, try latin-1
df = pd.read_csv("data/file.csv", encoding='latin-1')

# Or use error handling
df = pd.read_csv("data/file.csv", encoding='utf-8', errors='ignore')
```

### Issue 4: Chunking Creates Tiny Fragments

**Symptoms**: Some chunks are only 100-200 characters

**Cause**: Breaking at first space creates small fragments

**Solution**: Enforce minimum chunk size

```python
def create_text_chunks(text, max_length, overlap, min_chunk_size=500):
    """Add minimum chunk size constraint"""
    chunks = []
    # ... chunking logic ...

    # Only add chunks that meet minimum size
    if len(chunk) >= min_chunk_size:
        chunks.append(chunk)

    return chunks
```

---

## Exercises

### Exercise 1: Analyze Source Distribution

**Task**: How many samples come from each source?

```python
# Add source column during loading
df['source'] = 'sharegpt'  # Or 'imdb', 'boolq', 'brwac', 'canarim'

# After combining
source_counts = df_combined['source'].value_counts()
print(source_counts)

# Plot distribution
source_counts.plot(kind='bar')
plt.title('Samples per Source')
plt.ylabel('Count')
plt.show()
```

**Expected**: BrWaC dominates (>50%), others smaller

### Exercise 2: Length Distribution Analysis

**Task**: Visualize text length distribution before and after filtering

```python
import matplotlib.pyplot as plt

# Before filtering
lengths_before = df_combined['text'].str.len()

# After filtering
lengths_after = df_filtered['text'].str.len()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(lengths_before, bins=100, edgecolor='black')
axes[0].set_title('Before Filtering')
axes[0].set_xlabel('Text Length (chars)')
axes[0].set_ylabel('Frequency')
axes[0].set_xlim(0, 15000)

axes[1].hist(lengths_after, bins=100, edgecolor='black', color='green')
axes[1].set_title('After Filtering (100-10,000)')
axes[1].set_xlabel('Text Length (chars)')
axes[1].set_xlim(0, 15000)

plt.tight_layout()
plt.show()
```

### Exercise 3: Chunking Impact

**Task**: What percentage of data gets chunked? How many chunks created?

```python
# Separate normal vs chunked
normal_count = len(df_filtered[df_filtered['text'].str.len() <= 10000])
chunked_count = len(df_filtered[df_filtered['text'].str.len() > 10000])

# After chunking
final_chunks = len(df_processed[df_processed['chunk_id'] != ''])

print(f"Texts requiring chunking: {chunked_count:,}")
print(f"Chunks created: {final_chunks:,}")
print(f"Expansion factor: {final_chunks / chunked_count:.2f}x")
```

**Expected**: ~166K texts → ~350K chunks (2.1x expansion)

### Exercise 4: Balance Quality

**Task**: Verify balanced dataset maintains source diversity

```python
# Check source distribution in balanced dataset
balanced_sources = df_balanced.groupby(['label', 'source']).size()
print(balanced_sources)

# Plot
balanced_sources.unstack().plot(kind='bar', stacked=True)
plt.title('Source Distribution in Balanced Dataset')
plt.ylabel('Sample Count')
plt.legend(title='Source')
plt.show()
```

**Goal**: Both human and LLM should have diverse sources

---

## Summary

**Data Preprocessing Pipeline**:

```
Raw Data (5 sources, varied formats)
    ↓
Load & Standardize (→ CSV with label + text)
    ↓
Combine (2.3M samples)
    ↓
Filter Short (<100 chars) → Remove 171K
    ↓
Chunk Long (>10K chars) → Split 167K → 350K chunks
    ↓
Balanced Sample (30% = 708K, 50/50 split)
    ↓
Clean Dataset Ready for Feature Extraction ✅
```

**Key Takeaways**:

1. ✅ **Standardize first** - uniform format prevents errors
2. ✅ **Filter extremes** - too short or too long hurts quality
3. ✅ **Chunk intelligently** - break at sentences, not mid-word
4. ✅ **Balance carefully** - upsample minority, downsample majority
5. ✅ **Process in batches** - handle large data without crashes
6. ✅ **Validate thoroughly** - check counts, lengths, labels

**Next Step**: Guide 3 covers feature extraction from this clean dataset.

---

**Files Created**:
- `combined.csv` - All sources merged (2.3M samples)
- `processed_filtered_chunked.csv` - Filtered and chunked (2.34M samples)
- `balanced_processed.csv` - Final balanced dataset (708K samples)

**Ready for**: Feature extraction with `src/features.py`

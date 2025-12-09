# Instructions for Reviewing Agent

## Overview

You are reviewing two academic papers from a master's thesis project on detecting LLM-generated text in Brazilian Portuguese using statistical and fuzzy logic approaches. Both papers have already undergone one round of bibliography review and fixes.

## Your Task

Perform a comprehensive review focusing on:

1. **Bibliography Quality & Completeness**
2. **Citation Appropriateness & Currency**
3. **Argument Validity & Consistency**
4. **Methodological Soundness**
5. **Missing Literature (via Web Search)**

---

## Files Provided

### Primary Documents
- `paper_stat/main.pdf` - Statistical Analysis Paper (15 pages)
- `paper_fuzzy/main.pdf` - Fuzzy Logic Paper (13 pages)

### Supporting Documents
- `PROJECT_COMPLETION_SUMMARY.md` - Full project context, datasets, methods, results
- `BIBLIOGRAPHY_REVIEW_AND_FIXES.md` - Previous review findings and all fixes applied
- `paper_stat/refs.bib` - Shared bibliography (39 references)

### Context
- Both papers use the **same dataset**: 100K balanced Portuguese texts (50K human, 50K LLM)
- Both papers extract the **same 10 stylometric features**
- Statistical paper: Mann-Whitney U, Cliff's delta, PCA, LDA, Logistic Regression (97% AUC)
- Fuzzy paper: Triangular membership functions, data-driven quantile approach (89% AUC)

---

## Review Checklist

### 1. Bibliography Quality Assessment

#### A. Verify All References Are Appropriate
- [ ] All **statistical methods** cite original or authoritative sources
  - Mann-Whitney U test → Mann & Whitney 1947
  - Cliff's delta → Cliff 1993
  - FDR correction → Benjamini & Hochberg 1995
  - PCA → Jolliffe 2002 or Fisher 1936
  - LDA → Fisher 1936, McLachlan 2004
  - Logistic Regression → Hosmer & Lemeshow 2013

- [ ] All **fuzzy logic concepts** cite peer-reviewed sources
  - Fuzzy sets foundation → Zadeh 1965 ✓
  - Triangular membership functions → Pedrycz 1994, Ross 2010 ✓
  - Takagi-Sugeno systems → Takagi & Sugeno 1985 ✓
  - Data-driven fuzzy → Jang 1993 (ANFIS) ✓
  - General fuzzy theory → Klir & Yuan 1995 ✓

- [ ] **Dataset citations** are traceable
  - BrWaC → Wagner Filho et al. 2018
  - BoolQ → Clark et al. 2019
  - ShareGPT-Portuguese → Hugging Face URL provided ✓
  - Canarim → Hugging Face URL provided ✓

- [ ] **Prior work** on LLM detection
  - Herbold et al. 2023 → Now Scientific Data publication (Nature) ✓
  - Are there other recent papers (2023-2024) on stylometric LLM detection?

#### B. Check for Weak/Informal Sources
Previous review eliminated:
- ✓ ResearchHubs/ResearchPod blog posts → Replaced with Pedrycz, Ross, Klir
- ✓ arXiv-only Herbold citation → Upgraded to Scientific Data DOI

**Your job:** Look for any remaining issues:
- [ ] Are there any citations to personal websites, blogs, or non-peer-reviewed sources?
- [ ] Do all conference papers have venue information?
- [ ] Do dataset citations include DOIs or permanent URLs?

#### C. Identify Missing Metadata
Check these entries flagged in previous review:
- [ ] `davis2006` - Missing booktitle (Precision-Recall vs ROC curves)
- [ ] `kohavi1995` - Missing journal/venue (Cross-validation study)
- [ ] `romano2006` - Missing journal (Effect size study)
- [ ] `pandas` - Missing conference venue (should be SciPy 2010)
- [ ] `brwac` - Missing publication venue or DOI
- [ ] `boolq` - Missing conference venue (should be NAACL 2019)

**Action:** For each, perform a web search to find:
1. Full publication venue
2. DOI if available
3. Complete author list if "et al." is used

---

### 2. Citation Currency & Completeness

#### A. Search for Newer Publications
For each key topic, search for papers published in **2023-2024**:

**Statistical LLM Detection:**
- [ ] Search: "stylometric detection LLM generated text 2023 2024"
- [ ] Search: "ChatGPT detection stylometry Portuguese"
- [ ] Search: "authorship verification LLM machine learning"
- Are there more recent studies than Herbold 2023?

**Fuzzy Logic for Text Classification:**
- [ ] Search: "fuzzy logic text classification 2023 2024"
- [ ] Search: "fuzzy systems natural language processing"
- [ ] Search: "interpretable machine learning fuzzy"
- Are there recent fuzzy NLP applications the authors should cite?

**Portuguese NLP:**
- [ ] Search: "Brazilian Portuguese NLP datasets 2023 2024"
- [ ] Search: "Portuguese language models evaluation"
- Are there newer PT-BR corpora or benchmarks?

**Stylometric Features:**
- [ ] Search: "type-token ratio lexical diversity 2023"
- [ ] Search: "entropy stylometry linguistic features"
- Any recent papers on the specific features used (TTR, burstiness, entropy)?

#### B. Check for Seminal Papers Missing
- [ ] Is Zipf's law mentioned but not cited? (relevant for lexical diversity)
- [ ] Are there classic stylometry papers missing? (e.g., Burrows, Juola)
- [ ] For PCA, is Kaiser criterion or scree plot cited if used?
- [ ] For cross-validation, is Stone 1974 or Geisser 1975 cited?

---

### 3. Argument Validation

#### A. Statistical Paper Claims
Verify these key claims are properly supported:

1. **"9 of 10 features differ significantly at p<0.001"**
   - [ ] Check Table 1 in paper matches this claim
   - [ ] Verify FDR correction is mentioned
   - [ ] Confirm effect sizes (Cliff's delta) are reported

2. **"6 features have large effect sizes (|δ|≥0.474)"**
   - [ ] Cross-reference with Romano et al. 2006 guidelines
   - [ ] Check if threshold is correctly cited

3. **"Logistic regression achieves 97.03% AUC"**
   - [ ] Verify 5-fold CV is mentioned
   - [ ] Check if results are mean±SD
   - [ ] Confirm train/test split described

4. **"First comprehensive stylometric analysis in Portuguese"**
   - [ ] Perform web search to verify this claim
   - [ ] Search: "Portuguese LLM detection" "Brazilian Portuguese ChatGPT detection"
   - [ ] If other studies exist, authors should acknowledge them

#### B. Fuzzy Paper Claims
Verify these key claims:

1. **"First application of fuzzy logic to LLM detection"**
   - [ ] Search: "fuzzy logic LLM detection" "fuzzy classifier ChatGPT"
   - [ ] Confirm no prior fuzzy approaches exist
   - [ ] If any exist, note them

2. **"Fuzzy achieves 89.34% AUC with ±0.04% variance (3-4× lower than statistical models)"**
   - [ ] Check if comparison is fair (same CV protocol)
   - [ ] Verify variance calculation is correct
   - [ ] Confirm this is presented as a stability advantage

3. **"Triangular functions chosen for simplicity and efficiency"**
   - [ ] Is this claim supported by Pedrycz 1994 or Ross 2010?
   - [ ] Are alternative MF shapes mentioned (Gaussian, trapezoidal)?

4. **"Data-driven quantile approach (33%, 50%, 66%)"**
   - [ ] Is this methodology clearly described?
   - [ ] Are there citations for quantile-based fuzzy MF design?
   - [ ] Should cite Jang 1993 (ANFIS) for data-driven fuzzy?

---

### 4. Methodological Consistency

#### A. Cross-Paper Consistency
Both papers claim to use the **same data and features**. Verify:
- [ ] Do both papers describe the same 100K sample size?
- [ ] Do both list the same 10 features?
- [ ] Do both use 5-fold stratified CV?
- [ ] Are feature definitions identical across papers?

#### B. Statistical Rigor
- [ ] Is the choice of Mann-Whitney U justified (non-parametric)?
- [ ] Is FDR correction applied correctly (10 tests)?
- [ ] Is Cliff's delta the appropriate effect size for this test?
- [ ] Is PCA standardization mentioned?
- [ ] Is CV stratification mentioned to maintain class balance?

#### C. Fuzzy Logic Rigor
- [ ] Are membership functions mathematically defined (equations)?
- [ ] Is the inference method specified (Takagi-Sugeno type-0)?
- [ ] Is the aggregation method clear (arithmetic mean)?
- [ ] Is the orientation rule (direct vs inverse) explained?
- [ ] Are fuzzy MFs fit on training data only (no data leakage)?

---

### 5. Missing Literature - Web Search Tasks

#### High-Priority Searches

**A. Recent LLM Detection Studies (2023-2025)**
```
Search queries:
1. "GPT detection stylometry 2024"
2. "AI-generated text detection linguistic features"
3. "ChatGPT detector machine learning 2023"
4. "LLM authorship attribution 2024"
5. "BERT GPT detector comparison"
```
**Goal:** Find 3-5 recent papers the authors should cite or compare against.

**B. Fuzzy NLP Applications (Recent)**
```
Search queries:
1. "fuzzy logic sentiment analysis 2023"
2. "fuzzy text classification interpretable 2024"
3. "neuro-fuzzy natural language processing"
4. "type-2 fuzzy systems NLP"
```
**Goal:** Find if anyone has applied fuzzy logic to similar NLP tasks recently.

**C. Portuguese NLP Resources**
```
Search queries:
1. "BERTimbau GPT Portuguese benchmark"
2. "Portuguese text generation evaluation"
3. "Brazilian Portuguese LLM dataset 2024"
```
**Goal:** Identify newer PT-BR datasets or models the authors might compare to.

**D. Stylometric Features Literature**
```
Search queries:
1. "type-token ratio criticism limitations"
2. "burstiness text analysis citation"
3. "character entropy stylometry"
4. "hapax legomena authorship attribution"
```
**Goal:** Ensure each feature has a proper citation for its use in stylometry.

**E. Cross-Validation Best Practices**
```
Search queries:
1. "stratified k-fold cross-validation citation"
2. "cross-validation grouped data NLP"
3. "topic-based cross-validation text classification"
```
**Goal:** Check if the CV approach is optimal (papers mention lacking topic labels).

---

### 6. Specific Issues to Investigate

#### A. Dataset Imbalance Claim
Papers claim "balanced dataset" achieved by downsampling majority class.
- [ ] Search: "class imbalance text classification best practices"
- [ ] Check if downsampling is optimal or if SMOTE/class weights would be better
- [ ] Verify if this is standard practice in stylometry

#### B. Feature Selection Justification
Papers use 10 specific features.
- [ ] Search: "stylometric features LLM detection which"
- [ ] Are these the standard features in the field?
- [ ] Are any important features missing (e.g., n-gram diversity, perplexity)?

#### C. Comparison Fairness
Fuzzy paper compares to Herbold et al. (31 features, Random Forest).
- [ ] Is this a fair comparison (different feature count)?
- [ ] Should authors compare fuzzy to simpler baselines?
- [ ] Are there other interpretable methods (decision trees, rule-based) to compare?

#### D. Generalization Claims
Papers focus on generic Portuguese text.
- [ ] Do authors acknowledge domain shift concerns?
- [ ] Should they test on different domains (news, academic, social media)?
- [ ] Are there citations for domain adaptation in stylometry?

---

### 7. Output Format

Provide your review as a structured report:

```markdown
# Academic Review Report

## 1. Bibliography Quality
### ✅ Strengths
- [List what's good]

### ⚠️ Issues Found
- [List each issue with entry name and suggested fix]

### 🔍 Missing Metadata
- [List entries needing completion with web search results]

## 2. Citation Currency
### Recent Papers to Add (2023-2024)
- [Title, Authors, Venue, Year, DOI, Why relevant]

### Classic Papers Missing
- [Title, Authors, Year, Why it should be cited]

## 3. Argument Validation
### Statistical Paper
- [Verify each claim, note if unsupported]

### Fuzzy Paper
- [Verify each claim, note if unsupported]

## 4. Methodological Issues
- [List any concerns about methods, consistency, rigor]

## 5. Recommended Additions
### New References to Add
- [Full BibTeX entries for suggested additions]

### Text Changes Needed
- [Specific locations where citations should be added/changed]

## 6. Overall Assessment
- **Ready for submission?** Yes/No (with caveats)
- **Major revisions needed?** List them
- **Minor revisions needed?** List them

## 7. Priority Action Items
1. [Most critical fix]
2. [Second priority]
3. [etc.]
```

---

## Notes

- **Language:** Papers are in Portuguese, but you can review methodology/citations in English
- **Scope:** Focus on academic rigor, not writing style or grammar
- **Standard:** Papers should meet requirements for:
  - Master's thesis defense
  - Conference submission (ACL, EMNLP, NAACL)
  - Journal submission (Computational Linguistics, etc.)

- **Previous fixes:** See `BIBLIOGRAPHY_REVIEW_AND_FIXES.md` for what was already addressed
  - All weak web sources removed ✓
  - Herbold upgraded to Scientific Data ✓
  - Dataset URLs added ✓
  - Fuzzy references strengthened ✓

## Questions to Answer

1. Are there any **2024 papers** on LLM detection the authors should cite?
2. Are the **10 stylometric features** standard in the field or arbitrary?
3. Is the **fuzzy approach** truly novel or has it been done before?
4. Are there **better comparison baselines** than Herbold et al.?
5. Should authors test on **multiple domains** to claim generalization?
6. Are there **missing citations** for specific claims (e.g., "first in Portuguese")?
7. Do the **statistical tests** match best practices in NLP evaluation?
8. Is the **CV protocol** optimal given the data structure?

---

**Begin your review. Be thorough, critical, and constructive.**

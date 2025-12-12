# Pedagogical Guides for Academic Research Project

**Project**: Statistical and Fuzzy Logic Analysis of LLM-generated Text Detection
**Purpose**: Comprehensive, didactic guides for understanding and reproducing the research
**Audience**: Students, researchers, and anyone wanting to understand the methodology

---

## Guide Structure

These guides are designed to be read sequentially, building knowledge step-by-step:

### Foundation Guides (Must Read First)

1. **[Data Collection & Processing](./01_data_collection.md)**
   - Understanding data sources
   - Downloading and organizing data
   - Quality assessment
   - Format specifications

2. **[Data Preprocessing](./02_preprocessing.md)**
   - Text filtering logic
   - Chunking algorithm
   - Stratified sampling mathematics
   - Balancing procedures

3. **[Feature Engineering](./03_feature_engineering.md)**
   - Each of the 10 stylometric features explained
   - Mathematical formulas with examples
   - Scale of measurement understanding
   - Implementation details

### Statistical Analysis Guides

4. **[Statistical Testing](./04_statistical_testing.md)**
   - Non-parametric vs parametric tests
   - Mann-Whitney U test step-by-step
   - Effect sizes (Cliff's Delta)
   - Multiple comparison corrections (FDR)

5. **[Multivariate Analysis](./05_multivariate_analysis.md)**
   - PCA: Theory and practice
   - LDA: Discriminant analysis
   - Logistic Regression: Probabilistic classification
   - When to use each method

6. **[Model Evaluation](./07_model_evaluation.md)**
   - Cross-validation strategies
   - Performance metrics (AUC, Precision, Recall)
   - ROC and PR curves
   - Model comparison techniques

### Fuzzy Logic Guides

7. **[Fuzzy Logic Classifier](./06_fuzzy_classifier.md)**
   - Fuzzy set theory foundations
   - Membership functions explained
   - Fuzzy inference systems
   - Defuzzification methods
   - Building interpretable classifiers

### Communication & Reproducibility

8. **[Paper Writing from Code](./08_paper_writing.md)**
   - Methods section structure
   - Results reporting standards
   - Statistical language guidelines
   - Common mistakes to avoid
   - LaTeX tips for academic writing

9. **[Reproducibility Checklist](./09_reproducibility.md)**
   - Environment setup
   - Running the complete pipeline
   - Verification procedures
   - Troubleshooting common issues

10. **[Common Pitfalls & Solutions](./10_pitfalls.md)**
    - Statistical errors to avoid
    - Code debugging strategies
    - LaTeX compilation issues
    - Citation management

---

## How to Use These Guides

### For Students Learning

Read guides 1-3 first to understand the data pipeline. Then:
- **If interested in statistics**: Read guides 4-6
- **If interested in fuzzy logic**: Read guide 7
- **For thesis writing**: Read guide 8
- **To reproduce results**: Read guide 9

### For Reviewers/Researchers

- **Quick overview**: Read guide 9 (Reproducibility) first
- **Methodology verification**: Guides 2-5
- **Code review**: Guides 3, 4, 6, 7
- **Paper assessment**: Guide 8

### For Reproducers

Follow this exact sequence:
1. Guide 9 (setup environment)
2. Guide 1 (get data)
3. Guide 2 (preprocess)
4. Guide 3 (features)
5. Guides 4-7 (run analyses)
6. Verify results match papers

---

## Pedagogical Philosophy

These guides follow these principles:

1. **Explain the "Why"** - Not just "do this," but "do this because..."
2. **Show Examples** - Real code snippets and worked examples
3. **Connect to Literature** - Every technique cited with references
4. **Visual Learning** - Diagrams and visualizations where helpful
5. **Build Intuition** - Start simple, add complexity gradually
6. **No Hand-Waving** - Mathematical rigor with accessible explanations

---

## Prerequisites

### Required Knowledge

- **Basic statistics**: Mean, standard deviation, distributions
- **Basic Python**: Variables, loops, functions
- **Basic command line**: Running scripts, navigating directories

### Helpful But Not Required

- Machine learning concepts
- LaTeX basics
- Fuzzy logic (we'll teach this from scratch)

---

## Getting Help

If something is unclear:
1. Check the **Common Pitfalls** guide (Guide 10)
2. Review the corresponding section in the papers
3. Examine the actual code in `/src/` directory
4. Check the technical documentation in `/docs/` directory

---

## Document Status

| Guide | Status | Last Updated | Reviewer |
|-------|--------|--------------|----------|
| 01 - Data Collection | ⚠️ IN PROGRESS | 2025-12-11 | Pending |
| 02 - Preprocessing | ⬜ TODO | - | - |
| 03 - Feature Engineering | ⬜ TODO | - | - |
| 04 - Statistical Testing | ⬜ TODO | - | - |
| 05 - Multivariate Analysis | ⬜ TODO | - | - |
| 06 - Fuzzy Classifier | ⬜ TODO | - | - |
| 07 - Model Evaluation | ⬜ TODO | - | - |
| 08 - Paper Writing | ⬜ TODO | - | - |
| 09 - Reproducibility | ⬜ TODO | - | - |
| 10 - Common Pitfalls | ⬜ TODO | - | - |

---

## Contributing

These guides are living documents. If you find:
- Unclear explanations
- Missing examples
- Errors or typos
- Outdated information

Please note them for updates.

---

## License & Citation

If these guides help your research, please cite the associated papers:

```bibtex
[STATISTICAL PAPER CITATION]
[FUZZY PAPER CITATION]
```


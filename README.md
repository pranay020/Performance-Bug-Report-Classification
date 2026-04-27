# Performance-Bug-Report-Classification
## Overview
This project implements an intelligent software engineering tool for classifying whether a bug report is performance-related or not. The study was developed for the Intelligent Software Engineering coursework and focuses on bug report classification using text-based machine learning methods.

The project compares:
- A baseline model: `TF-IDF + Multinomial Naive Bayes`
- An improved model: `TF-IDF + Logistic Regression` with `class_weight="balanced"`

The goal is to determine whether the improved model can better detect minority-class performance bug reports in highly imbalanced datasets.

## Datasets
The experiments use five real-world bug report datasets from deep learning projects:

- `caffe.csv`
- `incubator-mxnet.csv`
- `keras.csv`
- `pytorch.csv`
- `tensorflow.csv`

For each bug report:
- `Title` and `Body` are used as textual input
- `related` is used as the binary target label
  - `1` = performance-related
  - `0` = not performance-related

## Method
### Preprocessing
- Missing values in `Title` and `Body` are replaced with empty strings
- `Title` and `Body` are merged into one text feature
- The text is transformed into numeric vectors using `TF-IDF`

### Models
1. **Baseline**
   - `TfidfVectorizer`
   - `MultinomialNB`

2. **Improved Model**
   - `TfidfVectorizer`
   - `LogisticRegression(max_iter=1000, class_weight="balanced")`

## Experimental Setup
- Train-test split: `70% training / 30% testing`
- Stratified sampling is used to preserve class distribution
- Each experiment is repeated `30 times`
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
- Statistical significance test:
  - Wilcoxon signed-rank test

## Main Findings
- The baseline model achieved an average F1-score of `0.0` on all five datasets
- The improved model performed better on:
  - `incubator-mxnet`
  - `pytorch`
  - `tensorflow`
- The strongest improvement was observed on `tensorflow`
- Severe class imbalance remained a major challenge, especially for `caffe` and `keras`

## Repository Structure
```text
.
├── data/
│   ├── caffe.csv
│   ├── incubator-mxnet.csv
│   ├── keras.csv
│   ├── pytorch.csv
│   └── tensorflow.csv
├── notebooks/
│   └── bug_report_classification.ipynb
├── results/
│   ├── final_results_summary.csv
└── README.md

# MSBD566-Assignment-3/Final Project
# MSBD566 – Midterm/Final Project: Breast Cancer Classification (Random Forest)/Dimensionality Reduction and Neural Network

**Author:** James Walton  
**Course:** MSBD566 – Predictive Modeling and Analytics  
**Date:** December 2025  

---

## Project Overview
This project focuses on classifying breast cancer tumors as benign or malignant using diagnostic measurements collected from fine-needle aspirate (FNA) samples of breast tissue. Building on the previous analysis, which used a Random Forest Classifier, this extended study incorporates two additional approaches: a dimensionality reduction step using Principal Component Analysis (PCA) and a neural network model. The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains 30 numerical features describing the physical characteristics of cell nuclei captured from digitized medical images.

The purpose of this expanded analysis is to compare how different modeling strategies handle the same diagnostic information and to evaluate whether reducing the number of features affects classification performance. Understanding how well these methods identify tumors that are likely to be malignant contributes to early detection efforts and may support clinical decision-making by highlighting consistent patterns in the underlying diagnostic measurements.

---

## Dataset Information

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Description:** Fine-needle aspirate (FNA) samples of breast tissue analyzed for cell nuclei features.
- **Target Variable:**  
  - `M` = Malignant  
  - `B` = Benign  
- **Features:** 30 numeric attributes describing morphology (e.g., radius, texture, smoothness, concavity).

---

## Methods

### Model Used
- **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)
  - 300 estimators
  - Random state = 42
  - Parallel computation (`n_jobs=-1`)

### Rationale
Random Forest was selected because it:
- Handles non-linear relationships effectively  
- Provides feature importance estimates  
- Is robust against overfitting  

---

## Visualization

### Confusion Matrix
A **confusion matrix heatmap** illustrates model performance by comparing actual vs. predicted classes:

| Metric | Meaning |
|--------|----------|
| **True Positive (TP)** | Correctly predicted malignant cases |
| **True Negative (TN)** | Correctly predicted benign cases |
| **False Positive (FP)** | Benign predicted as malignant |
| **False Negative (FN)** | Malignant predicted as benign |

Interpretation:  
- High **TP** and **TN** values → strong classification ability  
- Low **FP/FN** values → minimal diagnostic errors  

---

## Evaluation Metrics

| Metric | Description |
|---------|--------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Fraction of predicted malignant cases that were correct |
| **Recall (Sensitivity)** | Fraction of actual malignant cases correctly identified |
| **F1-score** | Harmonic mean of precision and recall |

All metrics are printed in the notebook output using `classification_report()`.

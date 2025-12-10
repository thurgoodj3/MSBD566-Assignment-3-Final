# MSBD566 – Final Project  
### Breast Cancer Classification Using Random Forests, PCA, and Neural Networks

**Author:** James Walton  
**Course:** MSBD566 – Predictive Modeling and Analytics  
**Date:** December 2025  

---

## Project Overview

This project investigates multiple modeling strategies for classifying breast cancer tumors as **benign** or **malignant** using diagnostic measurements derived from fine-needle aspirate (FNA) images.

Building on an earlier analysis that relied solely on a Random Forest classifier, this extended project explores two additional approaches:

1. **Dimensionality Reduction using Principal Component Analysis (PCA)**
2. **A Feed-Forward Neural Network Classifier**

The goal is to compare how different modeling strategies handle the same biomedical dataset and how these choices affect diagnostic accuracy—especially the ability to correctly identify malignant tumors, where recall is medically crucial.

---

## Dataset Information

**Source:** UCI Machine Learning Repository  
**Description:** Diagnostic features extracted from digitized FNA images.  
**Classes:**  
- `M` = Malignant  
- `B` = Benign  

**Features:**  
30 numerical attributes describing cell-nuclei morphology, including:
- radius  
- texture  
- smoothness  
- concavity  
- compactness  
- symmetry  

Standardization (z-score normalization) is applied to all numerical features before modeling to ensure consistent scaling.  
:contentReference[oaicite:1]{index=1}

---

## Methodology

### Baseline Model — Random Forest Classifier

**Implementation:** `sklearn.ensemble.RandomForestClassifier`

**Configuration**
- `n_estimators=300`
- `random_state=42`
- `n_jobs=-1` (parallel computation)

**Why Random Forest?**
- Handles nonlinear feature interactions  
- Provides feature importance rankings  
- Robust to noise and moderate overfitting  

This model forms the baseline for comparison with PCA-reduced and neural-network models.

---

### PCA + Random Forest Pipeline

PCA was used to reduce the 30-dimensional input space into a smaller set of orthogonal components.

**Workflow**
1. Standardize all numerical features.  
2. Fit PCA on the full dataset.  
3. Select the number of components explaining **≈95%** of variance.  
4. Train a Random Forest on the PCA components.

**Performance**
- **Test Accuracy:** 94.7%  
- **Recall (Malignant):** 0.90  
- The lower recall indicates **missed malignant cases**, making PCA less suitable for sensitive diagnostic tasks.  
:contentReference[oaicite:2]{index=2}

**Interpretation:**  
While PCA removes redundancy and noise, it also flattens subtle nonlinear patterns critical for identifying malignancies.

---

### Neural Network Classifier

A simple feed-forward network was trained on the original standardized features without dimensionality reduction.

**Architecture (from report)**
- **Input:** 30 standardized features  
- **Hidden Layer 1:** ReLU  
- **Hidden Layer 2:** ReLU  
- **Output:** Sigmoid (binary classification)  

**Training Notes**
- 80/20 stratified train-test split  
- Early runs overfit quickly  
- Adjustments included lowering the learning rate and using smaller batch sizes  
- Final model converged quickly and consistently  
:contentReference[oaicite:3]{index=3}

**Performance**
- **Accuracy:** 99.1%  
- **Precision/Recall:** nearly perfect  
- **Only one malignant sample misclassified**  

**Interpretation:**  
The network captured complex, multidimensional patterns that PCA-based models struggled with—highlighting the power of nonlinear modeling in biomedical classification tasks.

---

## Evaluation Metrics

The following metrics were used across all models:

| Metric | Meaning |
|--------|---------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Proportion of predicted malignant cases that were correct |
| **Recall (Sensitivity)** | Proportion of actual malignant cases correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Distribution of TP, FP, TN, FN |

Confusion matrices were visualized to compare how each model handled misclassifications.

---

##  Key Takeaways

- **Random Forest:** Strong baseline performance; interpretable and reliable.  
- **PCA + Random Forest:** Simpler feature space but reduced sensitivity to malignancies.  
- **Neural Network:** Best overall performance, capturing nonlinear relationships lost during PCA compression.  

**Conclusion:**  
Dimensionality reduction does not always improve diagnostic performance. In this case, retaining the full set of features allowed the neural network to detect subtle patterns essential for accurate malignancy classification.


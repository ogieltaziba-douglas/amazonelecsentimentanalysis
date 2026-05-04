# Amazon Electronics Sentiment Classification

---

## Project Overview

This notebook implements a **multi-class sentiment classification system** for Amazon Electronics product reviews. It compares three traditional ML baselines (Logistic Regression, Multinomial Naive Bayes, Linear SVM) against two deep learning architectures (Bi-LSTM with Self-Attention, Transformer Encoder) to classify reviews as **Positive**, **Neutral**, or **Negative**.

## Dataset

| Property | Value |
| --- | --- |
| **Source** | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (HuggingFace) |
| **Category** | Electronics |
| **Sampled** | 50,000 reviews (streamed from full corpus) |
| **After Cleaning** | 47,444 reviews (duplicates removed) |
| **Classes** | Positive (~80%), Negative (~13%), Neutral (~7%) |
| **Split** | 70% Train / 15% Validation / 15% Test (stratified) |

## How to Run

### Environment

- **Platform:** Azure Machine Learning Compute Instance
- **Kernel:** Python 3.10 — AzureML
- **GPU:** CUDA-enabled GPU recommended (training will use CPU if unavailable)

### Instructions

1. Upload `AIDL_Individual.ipynb` to your Azure ML workspace.
2. Open the notebook and select the **Python 3.10 - AzureML** kernel.
3. Click **Run All** (⏩) — the notebook is fully self-contained:
   - All dependencies are installed in the first code cell (`pip install`).
   - The dataset is streamed directly from HuggingFace (no manual download required).
   - Pre-trained GloVe embeddings are downloaded automatically.
   - All random seeds are fixed (`RANDOM_SEED = 42`) for full reproducibility.
4. **Expected runtime:** ~10-5 minutes on a GPU instance (varies by hardware).
5. Output files (`*.png` charts) are saved to the working directory.

### Notebook Structure

| Section | Cells | Description |
| --- | --- | --- |
| **Data Preprocessing & EDA** | 0–30 | Data loading, cleaning, class mapping, visualisation, train/val/test split |
| **ML Baselines** | 31–42 | TF-IDF + Logistic Regression, Naive Bayes, Linear SVM |
| **Bi-LSTM ** | 43–61 | Vocabulary, GloVe embeddings, model architecture, training |
| **Transformer ** | 62–75 | Encoder-only Transformer, training, architecture comparison |
| **Evaluation** | 76–92 | Test-set metrics, confusion matrices, ROC/AUC, per-class accuracy, attention heatmaps |
| **Runtime** | 93–94 | Total notebook execution time |

---

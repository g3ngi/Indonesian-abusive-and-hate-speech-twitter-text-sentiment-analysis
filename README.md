# Model Comparison: IndoBERT vs Bi-LSTM for Indonesian Abusive and Hate Speech Detection on Twitter

A deep learning comparative study to detect **hate speech** and **abusive language** in Indonesian-language tweets. This project benchmarks a classical Bi-LSTM architecture against the transformer-based **IndoBERT** (`indobenchmark/indobert-base-p2`) on a real-world Indonesian Twitter dataset.

> 🏆 **Result highlight:** IndoBERT outperforms Bi-LSTM with **87.57%** test accuracy vs **82.74%**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Example Predictions](#-example-predictions)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Acknowledgments](#-acknowledgments)

---

## 🔎 Overview

Hate speech detection in Indonesian poses unique challenges: code-mixing with English, heavy use of slang (*bahasa alay*), abbreviations, and non-standard spelling. This project addresses those challenges by:

1. **Cleaning & normalizing** raw tweet text (URLs, mentions, hashtags, punctuation, digits).
2. **Slang normalization** using a *kamus alay* (slang dictionary) mapping.
3. **Balancing** the imbalanced dataset via random under-sampling.
4. **Training two models** with different paradigms — sequence learning (Bi-LSTM) vs contextual transformer (IndoBERT).
5. **Comparing performance** on the same held-out test set.

---

## 📊 Dataset

- **Source:** [Indonesian Abusive and Hate Speech Twitter Text](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text) (Kaggle)
- **Target label used:** `HS` (Hate Speech — binary: 0 = non-hate, 1 = hate)
- **Other available labels:** `Abusive`, `HS_Individual`, `HS_Group`, `HS_Religion`, `HS_Race`, `HS_Physical`, `HS_Gender`, `HS_Other`, `HS_Weak`, `HS_Moderate`, `HS_Strong`
- **Auxiliary file:** `new_kamusalay.csv` — slang → formal Indonesian mapping dictionary.

### Data Split (after under-sampling)

| Split      | Ratio |
| ---------- | ----- |
| Train      | 70%   |
| Validation | 15%   |
| Test       | 15%   |

Stratified split to preserve label distribution across sets.

---

## 🛠️ Methodology

### 1. Preprocessing pipeline

- Lowercase conversion
- Remove digits, URLs (`http(s)://...`, `www...`)
- Remove mentions (`@user`) and hashtags (`#tag`)
- Strip non-ASCII bytes (`\x80-\xFF`) and punctuation
- Drop residual artifacts (`xf`, `x`, `xe`, `f`, `e`, `user`/`USER`)
- **Slang normalization** via `new_kamusalay.csv` lookup
- Collapse whitespace

### 2. Class balancing

The `HS` label is imbalanced in the raw data. We use `RandomUnderSampler` from `imbalanced-learn` to downsample the majority class to match the minority — preventing the model from collapsing to a trivial classifier.

### 3. Models

#### 🧠 Bi-LSTM (Keras / TensorFlow)

| Hyperparameter      | Value           |
| ------------------- | --------------- |
| Vocab size          | 10,000          |
| Max sequence length | 150             |
| Embedding dim       | 128             |
| LSTM units          | 64 (× 2 stacked Bi-LSTM layers) |
| Dropout             | 0.3             |
| Dense head          | 64 (ReLU) → 1 (sigmoid) |
| Optimizer           | Adam            |
| Loss                | Binary cross-entropy |
| Callbacks           | EarlyStopping (patience=5), ReduceLROnPlateau |
| Max epochs          | 20              |
| Batch size          | 32              |

#### 🤖 IndoBERT (PyTorch / HuggingFace `transformers`)

| Hyperparameter      | Value                              |
| ------------------- | ---------------------------------- |
| Pretrained model    | `indobenchmark/indobert-base-p2`   |
| Max sequence length | 128                                |
| Optimizer           | AdamW (lr=2e-5, eps=1e-8)          |
| Scheduler           | Linear warmup                      |
| Gradient clipping   | 1.0                                |
| Epochs              | 5 (with early stopping, patience=3) |
| Batch size          | 32                                 |

---

## 📈 Results

### Test set performance (1,669 samples, balanced)

| Model        | Accuracy  | Precision (macro) | Recall (macro) | F1 (macro) | Test Loss |
| ------------ | --------- | ----------------- | -------------- | ---------- | --------- |
| Bi-LSTM      | 82.74%    | 0.83              | 0.83           | 0.83       | 0.3948    |
| **IndoBERT** | **87.57%** | **0.88**         | **0.88**       | **0.88**   | 0.5233    |

### Per-class breakdown — Bi-LSTM

| Class               | Precision | Recall | F1   | Support |
| ------------------- | --------- | ------ | ---- | ------- |
| 0 — Non-Hate Speech | 0.81      | 0.85   | 0.83 | 835     |
| 1 — Hate Speech     | 0.84      | 0.80   | 0.82 | 834     |

### Per-class breakdown — IndoBERT

| Class               | Precision | Recall | F1   | Support |
| ------------------- | --------- | ------ | ---- | ------- |
| 0 — Non-Hate Speech | 0.90      | 0.86   | 0.88 | 835     |
| 1 — Hate Speech     | 0.86      | 0.90   | 0.88 | 834     |

### Key takeaways

- **IndoBERT wins on every metric** (+4.83 percentage points in accuracy, +0.05 macro F1).
- IndoBERT shows **better recall on the hate-speech class** (0.90 vs 0.80) — important for moderation use cases where false negatives are costly.
- The Bi-LSTM showed signs of overfitting early (train acc reached 99% while val acc stalled at ~81%), suggesting it memorized rather than generalized.
- IndoBERT's contextual embeddings appear better at handling Indonesian slang and code-mixing.

---

## 💻 Tech Stack

- **Languages:** Python 3.x
- **Deep Learning:** TensorFlow / Keras, PyTorch
- **NLP:** HuggingFace `transformers`, IndoBERT
- **Data:** `pandas`, `numpy`
- **Class balancing:** `imbalanced-learn`
- **Evaluation & viz:** `scikit-learn`, `matplotlib`, `seaborn`, `wordcloud`
- **Environment:** Google Colab (GPU recommended for IndoBERT)

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```text
tensorflow
torch
transformers
scikit-learn
imbalanced-learn
pandas
numpy
matplotlib
seaborn
wordcloud
kaggle
```

---

## 🚀 How to Run

The notebook is designed to run on **Google Colab** (recommended — free GPU).

1. **Set up Kaggle API**
   - Download your `kaggle.json` from your Kaggle account settings.
   - When prompted by the notebook, upload `kaggle.json`.

2. **Open the notebook**
   ```
   AoL_2_deep_learning-2.ipynb
   ```

3. **Run all cells in order.** The notebook will:
   - Download the dataset from Kaggle
   - Preprocess and balance the data
   - Train Bi-LSTM (~5–10 min on GPU)
   - Train IndoBERT (~15–25 min on GPU)
   - Evaluate both and print classification reports + confusion matrices
   - Run example predictions

> 💡 **Tip:** Enable GPU in Colab via `Runtime → Change runtime type → T4 GPU`.

---

## 🎯 Example Predictions

The notebook ends with side-by-side inference on sample Indonesian tweets:

| Input Tweet                              | Bi-LSTM        | IndoBERT       |
| ---------------------------------------- | -------------- | -------------- |
| *"dasar kampungan, jangan sok bersih lah"* | Hate Speech    | Hate Speech    |
| *"saya sangat senang dengan pelayanan ini"* | Non-Hate Speech | Non-Hate Speech |
| *"ini adalah contoh kalimat netral"*       | Non-Hate Speech | Non-Hate Speech |

Both models agree on these clear-cut examples, but IndoBERT generally provides more confident and accurate probabilities on ambiguous cases.

---

## 📁 Project Structure

```
.
├── AoL_2_deep_learning-2.ipynb    # Main notebook (training + evaluation)
├── data.csv                        # Downloaded dataset (from Kaggle)
├── new_kamusalay.csv               # Slang normalization dictionary
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- [ ] Multi-label classification covering all 12 label columns (not just `HS`)
- [ ] Try **IndoBERT-large** or **XLM-RoBERTa** as stronger baselines
- [ ] Compare with traditional ML baselines (SVM, Logistic Regression with TF-IDF)
- [ ] Address class imbalance with **class weighting** or **focal loss** instead of under-sampling (to keep more data)
- [ ] Add **k-fold cross-validation** for more robust evaluation
- [ ] Deploy as a REST API (FastAPI) or interactive demo (Streamlit / Gradio)
- [ ] Explainability with **LIME** or **SHAP** to understand model decisions
- [ ] Data augmentation with back-translation for the minority class

---

## 🙏 Acknowledgments

- **Dataset:** [Ilham Firdausi Putra](https://www.kaggle.com/ilhamfp31) — *Indonesian Abusive and Hate Speech Twitter Text* (Kaggle)
- **Pretrained model:** [IndoBenchmark](https://huggingface.co/indobenchmark/indobert-base-p2) — IndoBERT base p2
- **Slang dictionary:** `new_kamusalay.csv` from the same Kaggle dataset
- Built as part of an **Assurance of Learning (AoL)** Deep Learning project.

---

## 📜 License

This project is released for academic and educational purposes. Please respect the original dataset's license when reusing the data.

---

> ⚠️ **Ethical note:** Hate speech detection models can encode biases present in their training data. This model is intended for research and educational purposes and should not be deployed in production moderation systems without extensive bias auditing, human-in-the-loop review, and fairness testing.


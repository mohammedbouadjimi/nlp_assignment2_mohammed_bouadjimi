# NLP Assignment 2 – Bot or Not?
**Author**: Mohammed Bouadjimi  

---

## 🧩 Problem Definition

The rise of automated accounts (bots) on social media platforms has raised serious concerns about the spread of misinformation, manipulation of public opinion, and violation of user trust. These bots mimic human behavior and are often difficult to detect using traditional rule-based systems. This project focuses on designing a Natural Language Processing (NLP) pipeline for **automated bot detection** using machine learning models that process user-generated content and metadata.

The problem is framed as a **binary classification task**, where the model predicts whether a given Twitter account is a bot or a human. The model learns to distinguish patterns in the text (tweets, bios) and behavior (follower ratio, number of tweets) to make this decision.

Goals:
- Develop a transformer-based classification model using DistilBERT.
- Achieve better performance than traditional machine learning baselines.
- Package and document the entire NLP pipeline.

---

## 📚 Dataset Overview

The dataset used for this task is a labeled collection of Twitter user profiles. Each instance includes:

- **User description (text)**
- **Tweet content (text)**
- **Profile metadata**: follower count, following count, status count, etc.
- **Label**: `1` for bots, `0` for humans

Files included:
- `train.csv`: Used for training the model
- `test.csv`: Used for evaluating the model's generalization

The dataset contains thousands of samples and shows a slightly imbalanced distribution, where the number of bots and humans are not equal, justifying the use of F1-score and precision/recall in evaluation.

---

## 🧪 Evaluation Metrics

We evaluate model performance using the following metrics:

- **Accuracy**: Measures overall correctness
- **Precision**: Measures true positives out of predicted positives
- **Recall**: Measures true positives out of actual positives
- **F1 Score**: Harmonic mean of precision and recall (critical when data is imbalanced)

These metrics are calculated both on validation during training and final testing.

---

## 🧠 Model Explanation

The classification model was built using the HuggingFace `transformers` library. The steps include:

### Preprocessing:
- Removal of URLs, user mentions, hashtags, and special characters
- Lowercasing all text
- Tokenization using `DistilBERTTokenizerFast`
- Padding and truncation to a fixed maximum sequence length

### Model:
- **Architecture**: `DistilBERTForSequenceClassification`
- **Base model**: `distilbert-base-uncased`
- **Output Layer**: A single linear layer with softmax activation
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `AdamW`
- **Learning rate**: `2e-5`
- **Batch size**: 16
- **Epochs**: 3

Training is done using `Trainer` API with a custom training loop and validation logic. The training loop includes evaluation at the end of every epoch.

---

## ⚙️ Technical Stack

- **Python**: 3.10
- **Libraries**:
    - `transformers`
    - `datasets`
    - `torch`
    - `sklearn`
    - `matplotlib`, `seaborn`
    - `pandas`, `numpy`
- **Hardware**: Google Colab GPU runtime (Tesla T4)

---

## 🧪 Results and Comparison

After fine-tuning the DistilBERT model, we achieved the following results:

### Test Set Performance:

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 94.6%     |
| Precision  | 94.2%     |
| Recall     | 94.9%     |
| F1 Score   | 94.5%     |

### Comparison with Baseline:

A baseline using TF-IDF with an SVM classifier was implemented for comparison.

| Model                | Accuracy | F1 Score |
|---------------------|----------|----------|
| **DistilBERT**       | 94.6%    | 0.945    |
| **TF-IDF + SVM**     | 87.2%    | 0.865    |

The transformer-based approach outperforms the classical baseline significantly in all metrics, indicating the benefit of using contextual word embeddings.

---

## 🗂️ Project Structure

```bash
.
├── bot_or_not_model.zip               # Compressed trained model
├── nlp_assignment2_mohammed_bouadjimi.ipynb  # Full notebook with code and results
├── .gitattributes                     # Git LFS tracking file
├── README.md                          # This file
└── requirements.txt                   # Dependencies
```

---

## 🧪 How to Run This Project

1. **Clone the Repository**
```bash
git clone https://github.com/mohammedbouadjimi/nlp_assignment2_mohammed_bouadjimi.git
cd nlp_assignment2_mohammed_bouadjimi
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Extract the Model (Optional for inference)**
```bash
unzip bot_or_not_model.zip
```

4. **Launch the Notebook**
Open `nlp_assignment2_mohammed_bouadjimi.ipynb` using Jupyter or Colab and follow cell-by-cell execution.

---

## 🧾 Future Work

- Incorporate metadata features into the transformer model
- Extend to multi-class classification (e.g., spam bots, political bots, etc.)
- Deploy the model using FastAPI or Flask with a front-end for demo
- Create a live dashboard to analyze prediction trends

---

## 📬 Acknowledgments

- HuggingFace for providing free access to transformer models
- Google Colab for GPU acceleration
- scikit-learn and matplotlib teams for visualization tools

---

## 🔗 Project Link

[GitHub Repository](https://github.com/mohammedbouadjimi/nlp_assignment2_mohammed_bouadjimi)

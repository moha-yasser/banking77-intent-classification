
# Banking77 Intent Classification using Classical NLP and Machine Learning

## Overview

This project builds an end-to-end Natural Language Processing (NLP) pipeline for intent classification using the Banking77 dataset. The objective is to automatically classify customer support queries into one of 77 predefined banking intents, enabling faster and more efficient handling of customer requests.

The project follows a complete machine learning workflow, including exploratory data analysis (EDA), text preprocessing, feature extraction with TF-IDF, model training, hyperparameter tuning using cross-validation, and comprehensive model evaluation. Multiple classical machine learning algorithms are compared to identify the best-performing model based on macro-averaged evaluation metrics.

The repository is designed to demonstrate best practices in organizing an NLP project, documenting each stage of the workflow, and building a reproducible text classification pipeline using Python and scikit-learn.

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/sssonnn/banking77/code
- **Number of Samples:** 10,003
- **Number of Features:** 4
- **Target Variable:** `label_text`

---

## Project Structure

```text
banking77-intent-classification/
│
├── data/
│   ├── raw.csv
│   ├── eda.parquet
│   ├── processed.parquet
│   └── test.parquet
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_post_preprocessing_eda.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── models/
│   └── best_model.pkl
│
├── requirements.txt
└── README.md
```
---

## Workflow

### 1. Data Loading

- Load the Banking77 dataset.
- Inspect the dataset structure and data types.

### 2. Exploratory Data Analysis (EDA)

- Analyze class distribution and dataset balance.
- Assess class imbalance using the Coefficient of Variation (CV).
- Explore text statistics and create exploratory features (`word_count`, `char_count`, and `utterance_type`).
- Analyze utterance type distribution across banking intents.
- Identify common words, text length patterns, and potential outliers.

### 3. Data Preprocessing

- Clean and normalize the text.
- Prepare the text for feature extraction.

### 4. Post-Preprocessing Analysis

- Analyze the processed text.
- Identify representative terms for each intent using TF-IDF.

### 5. Feature Extraction

- Generate numerical text representations using CountVectorizer and TF-IDF.

### 6. Model Training

- Train and compare multiple classical machine learning models.

### 7. Hyperparameter Tuning

- Optimize model parameters using GridSearchCV with Stratified Cross-Validation.

### 8. Model Evaluation

- Evaluate models primarily using Macro F1-score (Macro), with Accuracy as a complementary metric.
- Report Precision and Recall for comprehensive performance analysis.

### 9. Error Analysis

- Investigate common misclassification patterns and analyze challenging intent pairs.

### 10. Final Model

- Save the best-performing pipeline for future inference.

---

## Methodology

### Text Representation

Two text vectorization techniques were evaluated:

* **CountVectorizer** was used with **Multinomial Naive Bayes** as a baseline representation, since this algorithm is commonly applied to raw term-frequency features in text classification.
* **TF-IDF Vectorizer** was used with **Logistic Regression** and **Linear SVM** to produce weighted feature representations that emphasize more informative terms while reducing the influence of very common words.


### Machine Learning Models

The following classical machine learning algorithms were evaluated:

* Multinomial Naive Bayes
* Logistic Regression
* Linear Support Vector Machine (Linear SVM)

### Model Selection

Hyperparameter optimization was performed using **GridSearchCV** with **Stratified K-Fold Cross-Validation** to ensure reliable model comparison while preserving the class distribution across folds.

### Evaluation Strategy

Model performance was primarily assessed using **Macro F1-score**, as the Banking77 dataset contains multiple intent classes with varying sample sizes. **Accuracy** was used as a complementary metric, while **Precision**, **Recall**, and the **Classification Report** were reported to provide a more comprehensive evaluation. An additional error analysis was conducted to investigate common misclassification patterns and identify opportunities for improvement.

### Reproducibility

A scikit-learn **Pipeline** was used to combine text vectorization and model training into a single workflow, ensuring consistent preprocessing during hyperparameter tuning, cross-validation, and final evaluation.

---

## Models Evaluated

## Results

| Model                   | Vectorizer      | Validation Macro F1 | Validation Accuracy |
| ----------------------- | --------------- | ------------------: | ------------------: |
| Logistic Regression     | TF-IDF          |          **0.8601** |          **0.8622** |
| Linear SVM              | TF-IDF          |              0.8483 |              0.8523 |
| Multinomial Naive Bayes | CountVectorizer |              0.8404 |              0.8449 |


All models were trained using the same data split and evaluated under the same cross-validation strategy to ensure a fair comparison. Hyperparameter tuning was performed using GridSearchCV, and the final model was selected based on **Macro F1-score** while also considering Accuracy and other evaluation metrics.

---

## Best Model

The **Logistic Regression** model combined with **TF-IDF** achieved the best validation performance and was selected as the final model.

### Test Set Performance

| Metric         |      Score |
| -------------- | ---------: |
| Accuracy       | **0.8571** |
| Macro F1-score | **0.8551** |

---

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- NLTK
- spaCy
- Contractions
- Joblib

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/moha-yasser/banking77-intent-classification.git
cd banking77-project
pip install -r requirements.txt
```

---

## Usage

Launch Jupyter Notebook and open the notebooks in numerical order:

```bash
jupyter notebook
```

Run the notebooks sequentially:

1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_post_preprocessing_eda.ipynb`
4. `04_modeling.ipynb`
5. `05_evaluation.ipynb`

---

## Future Improvements

* Evaluate additional text representation techniques, such as word embeddings.
* Compare the performance of transformer-based models (e.g., BERT) with classical machine learning approaches.
* Deploy the final model as a REST API or web application.
* Experiment with advanced hyperparameter optimization methods.
* Extend the pipeline to support real-time intent classification.

---

## Author

**Mohamed Yasser Anwer**

 Junior Data Scientist & NLP Enthusiast

* GitHub: https://github.com/moha-yasser
* LinkedIn: https://linkedin.com/in/mohamed-yasser-063a143b0

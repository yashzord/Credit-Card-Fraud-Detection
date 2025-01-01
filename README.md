# Credit Card Fraud Detection

A demonstration of building a Credit Card Fraud Detection system using Python, scikit-learn (and imblearn), XGBoost, and Flask. The project showcases:

- Data ingestion from a SQLite database  
- Preprocessing with pipelines (handling numeric and categorical features)  
- Dealing with imbalanced classes via SMOTE  
- Model experimentation (automatically training multiple classifiers)  
- Final model artifact (.joblib) for real-time inference  
- REST API endpoint (via Flask) to classify transactions in real time  

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Folder Structure](#2-folder-structure)  
3. [Requirements](#3-requirements)  
4. [Data Source](#4-data-source)  
5. [Model Training & Generation](#5-model-training--generation)  
6. [Flask API Usage](#6-flask-api-usage)  
7. [Testing the API](#7-testing-the-api)  
8. [Notes on FinalClassifier.joblib](#8-notes-on-finalclassifierjoblib)  
9. [Project Notebooks](#9-project-notebooks)  
10. [Contributing](#10-contributing)  
11. [License](#11-license)   

---

## 1. Overview

This repository contains code to predict whether a credit card transaction is fraudulent or not. We use:

- **scikit-learn** pipelines for data preprocessing and SMOTE for class rebalancing.  
- **RandomForest** and other estimators (including **XGBoost**) to compare model performance.  
- A **Flask** service to expose the trained model as a REST API.

In practice, a real fraud detection system would involve further domain-specific feature engineering, advanced model architectures, and thorough data security measures. However, this project demonstrates a working end-to-end pipeline for educational or demonstration purposes.

---

## 2. Folder Structure

```bash
.
├── classification/
│   ├── tests.py                       # Script to train multiple classifiers automatically
│   ├── utils.py                       # Pipeline & model generation utility functions
│   ├── CoreCard.joblib               # (Example model artifact, if generated)
│   ├── RFClassifier.joblib           # (Example model artifact, if generated)
│   ├── preprocessing.joblib          # (Example of saving only the preprocessing, if used)
│   ├── scores.txt                    # Holds metrics/results for each trained classifier
│   └── FinalClassifier.joblib        # Not pushed to Git (too large); kept locally
├── data_corecard_imbpipeline.ipynb   # (Notebook: alternative pipeline focusing on imbalanced data)
├── data_corecard_normpipeline.ipynb  # (Notebook: demonstration of data loading, EDA, model training)
├── example_input.csv                 # Example CSV input with columns matching the DB
├── app.py                            # Flask app to serve the final model
├── tests.py                          # Python script to test the running Flask API (POST requests)
├── fraud_test.json                   # Sample JSON for a known fraudulent transaction
├── non_fraud_test.json               # Sample JSON for a known non-fraudulent transaction
├── requirements.txt                  # Python dependencies
├── corecard_credit_card_transactions.db    # Local SQLite DB (not in repo, but you need it locally)
└── README.md                         # (You are here!)
```

---

## 3. Requirements

1. Python 3.7+  
2. Packages listed in requirements.txt:  
```bash
pip install -r requirements.txt
```
3. A SQLite database named corecard_credit_card_transactions.db (not included in this repo).

---

## 4. Data Source

1. All transaction data is pulled from a local SQLite database:
```python
conn = sqlite3.connect("corecard_credit_card_transactions.db")
df = pd.read_sql("SELECT * FROM transactions", conn)
```
2. Each row contains credit card transaction details (e.g., TransactionAmount, MerchantType, timestamps, etc.), along with a FraudIndicator column that labels a transaction as fraudulent (1.0) or non-fraudulent (0.0).

---

## 5. Model Training & Generation

1. To train multiple classifiers (including random forests, logistic regression, XGBoost, etc.) and compare their performance:

- Update paths as needed in `classification/tests.py` and `classification/utils.py` to point to your local DB.
- From the command line, run:
```bash
cd classification
python tests.py
```
- This calls run_all_classifiers():
- Iterates over all scikit-learn classifiers (and XGBoost).
- Preprocesses data via pipelines (numeric/categorical features, SMOTE).
- Trains each classifier and saves it as <ClassifierName>.joblib.
- Appends metrics to scores.txt.

2. To create a single specific final model (e.g., a FinalClassifier.joblib):

- In classification/utils.py, inside model_generator(...), change:
```python
model_name="FinalClassifier"
```
- or call it like this from a custom script/notebook:
```python
from classification import utils
from sklearn.ensemble import RandomForestClassifier
import sqlite3, pandas as pd

conn = sqlite3.connect("corecard_credit_card_transactions.db")
df = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

utils.model_generator(df, RandomForestClassifier, model_name="FinalClassifier")
```
- This will produce FinalClassifier.joblib in the classification/ folder.

---

## 6. Flask API Usage

1. The file app.py provides a Flask API to serve a single trained model (defaulting to classification/FinalClassifier.joblib).
2. Ensure FinalClassifier.joblib exists locally in classification/.
3. Launch the Flask server:
```bash
python app.py
```
- By default, it runs at http://127.0.0.1:5000.

4. POST a transaction JSON to http://127.0.0.1:5000/classify with the relevant keys matching the training data. For example:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d @fraud_test.json \
     http://127.0.0.1:5000/classify
```
- or use a tool like Postman.

---

## 7. Testing the API

1. After the server is running, you can test it in multiple ways:
### Using `tests.py`
The `tests.py` script (located in the root directory, not to be confused with `classification/tests.py`):
```bash
python tests.py
```

2. This script:
- Reads a sample JSON file (`test.json`).
- Submits it via `requests.post` to your running Flask server.
- Prints the response, including the prediction and confidence score.

### Manually Testing
1. You can also test the API manually using sample files like `fraud_test.json` or `non_fraud_test.json`.
```bash
curl -X POST -H "Content-Type: application/json" \
     -d @fraud_test.json \
     http://127.0.0.1:5000/classify
```

2. Expect a JSON response from the API similar to:
```json
{
  "prediction": [1.0],
  "confidence": [[0.3, 0.7]]
}
```

---

## 8. Notes on FinalClassifier.joblib

1. The file `FinalClassifier.joblib` is not in this repository due to its large size (>250 MB).

2. To get it locally, you need to train and save it. You can do this by:

- Running a script or notebook that calls `model_generator(...)` with `model_name="FinalClassifier"`.
- Or any other scikit-learn pipeline training code that includes `joblib.dump(...)` to produce `FinalClassifier.joblib`.

3. Once you have it, place it in the `classification/` folder so that `app.py` can load it:
```python
model = joblib.load("classification/FinalClassifier.joblib")
```

---

## 9. Project Notebooks

- `data_corecard_normpipeline.ipynb` (also shared as `.py`):
  - Demonstrates data loading, EDA, and a basic `RandomForestClassifier` pipeline.
  - Explores confusion matrix, accuracy, F1, and AUC metrics.

- `data_corecard_imbpipeline.ipynb`:
  - Potentially focuses on dealing with an imbalanced dataset, employing SMOTE, etc.
  - The exact content is not fully shown, but it follows a similar approach.

---

## 10. Contributing

Contributions are welcome! If you encounter issues or want to propose enhancements, feel free to open a pull request or create an issue.

---

## 11. License

This project is distributed under the MIT License. You can use and modify it for your own purposes, but please give appropriate credit.

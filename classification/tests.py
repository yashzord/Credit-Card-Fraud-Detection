from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators
import xgboost 
import sqlite3
import pandas as pd
import utils

conn = sqlite3.connect(r"C:\Users\yashu\OneDrive\Desktop\Github_Projects\Credit_Card_Fraud_Detection\corecard_credit_card_transactions.db")
df = pd.read_sql("SELECT * FROM transactions", conn)


def run_all_classifiers():
    classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
    classifiers.append(("XGBoost", "xgboost.XGBClassifier"))
    for name, model in classifiers:
        try:
            utils.model_generator(df, model, name)
        except:
            pass
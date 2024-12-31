import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Note that this Pipeline is imported from imblearn, not sklearn
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)




numerical_data = ['TransactionAmount', 'OutstandingAmount', 'CurrentBalance', 'TotalOutStgAuthAmt', 'ActualReversalAmount', 'CalcOTB', 'ATC']
categorical_data = ['TranType', 'PrimaryCurrencyCode', 'MessageTypeIdentifier', 'ProcCode', 'ProcCodeFromAccType', 'MerchantType', 'IResponseCode', 'ResponseCode', 'AuthStatus', 'POSTransactionStatusInd', 'TxnCategory', 'AdvReasonCode', 'AVResponse', 'PostingRef', 'TerminalType', 'Field112', 'AuthVarianceException', 'TxnSubCategory', 'PinExist', 'CardAcceptorNameLocation', 'CrossBorderTxnIndicator', 'MerchantName', 'MerchantGroup']
nominal_data = ['TxnAcctId', 'UniqueID', 'AccountNumber']
date_time = ['TranTime', 'PostTime', 'TransmissionDateTime', 'TimeLocalTransaction', 'DateLocalTransaction', 'EffectiveDate_ForAgeOff', 'PurgeDate']

label = 'FraudIndicator'


def model_generator(dataset, model, model_name = "FinalClassifier"):
    with joblib.parallel_backend('threading', n_jobs = -1):
        num_pipe = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  
        ])

        cat_pipe = Pipeline([
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
        ])
        preprocessing = ColumnTransformer(
            [
                #do not put label pipe here
                ('num', num_pipe, numerical_data),
                ('cat', cat_pipe, categorical_data)
            ]
            )
        clf = ImbPipeline(
        [
            ('preprocess', preprocessing),
            ('smote', SMOTE(random_state=42)),
            ('clf', model())
        ]
    )
        X = dataset.drop(label, axis=1)
        y = dataset[label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

        clf.fit(X_train, y_train)
        joblib.dump(clf, model_name +".joblib")

        y_pred = clf.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        class_names=[False, True] # name  of classes
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        tick_marks = [0.5, 1.5]
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        plt.savefig(model_name + "_confusion_matrix.png")


        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        b_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        r_squared = r2_score(y_test, y_pred)

        scores = {
            "Model": model_name,
            "Accuracy" : accuracy,
            "Balanced Accuracy": b_accuracy,
            "F1 Score": f1,
            "R-Squared": r_squared
        }
        with open("scores.txt", "a") as data:
            data.write(str(scores) + "\n")





    

            

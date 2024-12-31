import requests
import sqlite3
import json


def get_test_json():
    conn = sqlite3.connect(r"C:\Users\yashu\OneDrive\Desktop\Github_Projects\Credit_Card_Fraud_Detection\corecard_credit_card_transactions.db")
    cursor = conn.cursor()

    row = cursor.execute('SELECT * FROM transactions LIMIT 1').fetchone()

    row_dict = dict(zip([desc[0] for desc in cursor.description], row))

    with open('row.json', 'w') as json_file:
        json.dump(row_dict, json_file)

    conn.close()

def test_single_prediction():
    url = 'http://localhost:5000/classify'
    with open('test.json') as json_file:
        json_request = json.load(json_file)
    r = requests.post(url, json = json_request)
    print(r.json())


if __name__ == '__main__':
    test_single_prediction()
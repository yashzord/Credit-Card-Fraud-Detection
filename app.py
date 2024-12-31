from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json 
import ast

app = Flask(__name__)

model = joblib.load("classification/FinalClassifier.joblib")




@app.route('/classify',methods=['POST'])
def predict():
    data = request.get_json(force = True)
    if data is None:
        return jsonify({'error': 'No JSON data found'})
    else:
        input_data = pd.DataFrame(data, index =[0] )
        prediction = model.predict(input_data)
        confidence = model.predict_proba(input_data)
        output = {'prediction': prediction.tolist(), 'confidence': confidence.tolist()}
        return jsonify(output)



if __name__ == '__main__':
    app.run(port=5000, debug=True)



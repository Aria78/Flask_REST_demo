import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    query = pd.get_dummies(query_df)

    classifier = joblib.load('titanic_classifier.pkl')
    prediction = classifier.predict(query)

    return render_template('page.html',prediction_display_area='Predict result：{}'.format(jsonify({'prediction': str(list(prediction))})))

if __name__ == "__main__":
    app.run(port=3000,debug = True)
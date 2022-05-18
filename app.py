from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import pandas as pd
import csv

app = Flask(__name__)
model = pickle.load(open('titanic_classifier.pkl','rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Create variable for uploaded file
        f = request.files['fileupload']
        print('yes')
        # store the file contents as a string
        fstring = f.read()

        # create list of dictionaries keyed by header row
        csv_dicts = [{k: v for k, v in row.items()} for row in csv.DictReader(fstring.splitlines(), skipinitialspace=True)]

        print(csv_dicts)

    json_ = request.json
    query_df = pd.DataFrame(json_)
    query = pd.get_dummies(query_df)

    classifier = joblib.load('titanic_classifier.pkl')
    prediction = classifier.predict(query)

    return render_template('page.html',prediction_display_area='Predict result：{}'.format(jsonify({'prediction': str(list(prediction))})))

if __name__ == "__main__":
    app.run(port=3000,debug = True)
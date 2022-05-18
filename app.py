from flask import Flask, request, jsonify, render_template,url_for
import joblib
import pandas as pd
import os
from sklearn.metrics import classification_report

app = Flask(__name__)


UPLOAD_FOLDER  = '/Users/aria/Desktop'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # First save the uploaded csv file and pkl file.
    if request.method == 'POST':
        upload_file = request.files['file']
        upload_model = request.files['model']
        if upload_file.filename != '' and upload_model.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_file.filename)
            model_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_model.filename)
            upload_file.save(file_path)
            upload_model.save(model_path)

        df = pd.read_csv(file_path)
        print(file_path)
        print('Here')

        x = df[df.columns.difference(['Survived'])]
        y_true = df['Survived']

        classifier = joblib.load(model_path)
        prediction = classifier.predict(x[:])

        y_pred = list(prediction)

        return render_template('page.html',prediction_display_area='Result：{}'.format(classification_report(y_true, y_pred)))



if __name__ == "__main__":
    app.run(port=3000,debug = True)
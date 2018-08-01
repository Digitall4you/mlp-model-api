import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import pandas as pd
from scipy import misc

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route("/accurate", methods=['POST'])
def make_accurate():
    if request.method == 'POST':
        X_t = np.array(X_test_df)
        y_t = np.array(y_test_df)
        y_pred = clf.predict(X_t)
        accuracy = accuracy_score(y_t, y_pred)
        return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        file = request.files['image']
        if not file: return render_template('index.html', label="No file")
        img = misc.imread(file)
        img = img.reshape(-1, 64)
        prediction = clf.predict(img)
        label = str(np.squeeze(prediction))
        if label == '10': label = '0'
        return render_template('index.html', label=label)

if __name__ == '__main__':
    clf = joblib.load('model_MLP.pkl')
    X_test_df = pd.read_csv('X_test.csv')
    y_test_df = pd.read_csv('y_test.csv')
    app.run(port=8000, debug=True)
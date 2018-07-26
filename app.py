# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:54:04 2018

@author: Andry
"""
import numpy as np
from flask import Flask, abort, jsonify, request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/predict_api', methods=['POST'])
def predict():
     
     data = request.get_json(force=True)
     query = [data['X']]
     query = np.array(query)
     y = clf.predict(query)
     output = [y]
     return jsonify(results=output)

if __name__ == '__main__':
    clf = joblib.load('model_MLP.pkl')
     app.run(port = 9000, debug = True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 23:09:24 2020

@author: akhilreddyalluri
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app = Flask(__name__)
Swagger(app)

pickle_in = open('regressor.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/predict', methods=["Get"])
def predict_price():
    
    
    """Forecasting the medical price to analyze by insurance carrier.
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: bmi
        in: query
        type: number
        required: true
      - name: children
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    age = int(request.args.get("age"))
    bmi = float(request.args.get("bmi"))
    children = int(request.args.get("children"))
    
    y_predictions = model.predict([[age, bmi, children]])
    return " forecasted value is {}".format(y_predictions)

if __name__ == '__main__':
    app.run(debug=True)

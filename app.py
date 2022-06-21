# -*- coding: utf-8 -*-

import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
from model import run_model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def redirect_predict():
    return redirect(url_for('home'))


@app.route('/predict', methods=['POST'])
def predict():
    market = int(float(str(request.form.get('market') or 0)))
    funding_total_usd = int(
        float(str(request.form.get('funding_total_usd') or 0)))
    region = int(float(str(request.form.get('region') or 0)))
    city = int(float(str(request.form.get('city') or 0)))
    funding_rounds = int(float(str(request.form.get('funding_rounds') or 0)))
    founding_year = int(float(str(request.form.get('founding_year') or 0)))
    first_funding_year = int(
        float(str(request.form.get('first_funding_year') or 0)))
    last_funding_year = int(
        float(str(request.form.get('last_funding_year') or 0)))
    first_funding_gap = int(
        float(str(request.form.get('first_funding_gap') or 0)))
    last_funding_gap = int(
        float(str(request.form.get('last_funding_gap') or 0)))
    first_last_funding_gap = int(
        float(str(request.form.get('first_last_funding_gap') or 0)))
    # int_features = [int(float(str(x) or 0)) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # print(final_features)
    prediction = run_model([[market, funding_total_usd, region, city, funding_rounds, founding_year,
                           first_funding_year, last_funding_year, first_funding_gap, last_funding_gap, first_last_funding_gap]])
    # prediction=run_model([[8, 50000, 104, 449, 2, 2014, 2015, 2017, 2, 1, 1]])
    response = ""
    if int(prediction) == 1:
        response = "Company is likely to be Acquired/Merged"
    else:
        response = "Company is NOT likely to be Acquired/Merged"

    return render_template('index.html', prediction=response)


if __name__ == "__main__":
    app.run(debug=True)

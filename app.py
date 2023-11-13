from flask import Flask, render_template, request
import joblib
from joblib import dump, load
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

filename = 'file_WineQuality.pkl'
model = joblib.load('file_WineQuality.pkl')
dump(model, 'filename.joblib')      # save the model
model = load('filename.joblib')     # load the model

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    Alcohol = request.form['alcohol']
    Volatile_Acidity = request.form['volatile_acidity']
    Sulphates = request.form['sulphates']
    Citric_Acid = request.form['citric_acid']
    pred = model.predict(np.array([[float(Alcohol), float(Volatile_Acidity), float(Sulphates), float(Citric_Acid) ]])) #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

import keras
import datetime
from sklearn.feature_selection import mutual_info_regression

app = Flask(__name__)

@app.route('/', methods=['GET'])
def test():
    return jsonify({"status": "OK"})

# zapolnimo missing data 
def zapolni_podatke(df):
    imp = IterativeImputer()
    imp.fit(df)
    temp_df = imp.transform(df)
    df = pd.DataFrame(temp_df, columns=df.columns)
    return df

def izbira_znacilnic(X,Y,st_izbranih_znacilnic):
    # izracunamo katere znacilnice vzeti
    mutual_info = mutual_info_regression(X, Y)
    izbrane = X.columns[np.argsort(mutual_info)[-st_izbranih_znacilnic:]]
    return izbrane

def pripravi_timeseries(X, Y, st_zaporednih, st_napovedi):
    dolzina = len(X) # x_train ali x_test

    temp_X = []
    temp_Y = []

    for i in range(st_zaporednih, dolzina - st_napovedi + 1):
        temp_X.append(X[i - st_zaporednih:i, :])
        temp_Y.append(Y[i + st_napovedi - 1:i + st_napovedi])

    return np.array(temp_X), np.array(temp_Y)

@app.route('/predict', methods=['POST'])
def predict():
    # spremenljivke za settings
    st_izbranih_znacilnic = 10
    st_zaporednih = 504
    st_napovedi = 1

    data = request.get_json() # pridobimo json file
    df = pd.DataFrame.from_dict(data) # shranimo json file v df

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    df = df.drop(['time'], axis=1)

    # zapolnimo missing data
    df = zapolni_podatke(df)

    df = df.tail(505)
    X = df.drop(['global energy'], axis=1)
    Y = df['global energy'].to_numpy()

    # izberemo znacilnice
    izbira = izbira_znacilnic(X,Y,st_izbranih_znacilnic)
    X = X[izbira]

    # standardizacija podatkov
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # naredimo timeseries
    X, Y = pripravi_timeseries(X, Y, st_zaporednih, st_napovedi)

    # nalozimo nauceni model
    model = keras.models.load_model('model.h5')

    # napovemo vrednost
    predictions = model.predict(X)

    # vrnemo vrednost
    return jsonify({"predict": float(predictions[0][0])})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000)

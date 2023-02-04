# Projektna naloga RIRSU
# Datoteka za spletno storitev
# Rok Rozman

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() # pridobimo json file
    df = pd.DataFrame.from_dict(data) # shranimo json file v df
    df_avg = df['global energy'].mean()
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)

    df = df.drop(['time'], axis=1)

    imputer = IterativeImputer()
    imputer.fit(df)
    df_imputed = imputer.transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns)

    df = df.tail(505)

    X = df.drop(['global energy'], axis=1)
    Y = df['global energy'].to_numpy()

    mi = mutual_info_regression(X, Y)
    selected_features = X.columns[np.argsort(mi)[-10:]]
    X = X[selected_features]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_future = 1
    n_past = 504

    X_train_temp = []
    Y_train_temp = []
    X_test_temp = []
    Y_test_temp = []

    for i in range(n_past, len(X) - n_future + 1):
        X_train_temp.append(X[i - n_past:i, :])
        Y_train_temp.append(Y[i + n_future - 1:i + n_future])

    X = np.array(X_train_temp)
    Y = np.array(Y_train_temp)

    model = keras.models.load_model('model2.h5')

    predictions = model.predict(X)

    print("avg")
    print(df_avg)
    return jsonify({"predict": float(predictions[0][0])})

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5000)

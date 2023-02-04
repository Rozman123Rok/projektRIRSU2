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
    df_avg = df['rel. hum.'].mean()
    print("df ko preberemo iz json")
    print(df.head())
    print(df.isnull().sum())
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    
    print("df ko sort by time")
    print(df.head())
    df = df.drop(['time'], axis=1)
    print("df drop time")
    print(df.head())

    imputer = IterativeImputer()
    imputer.fit(df)
    df_imputed = imputer.transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns)
    print("df fill missing")
    print(df.head())
    print(df.isnull().sum())
    #print("num of rows")
    #print(df.shape[0])
    df = df.tail(505)
    print("num of rows")
    print(df.shape)

    X = df.drop(['rel. hum.'], axis=1)
    Y = df['rel. hum.'].to_numpy()
    #print("X")
    #print(X)
    #print("Y")
    #print(Y)

    mi = mutual_info_regression(X, Y)
    selected_features = X.columns[np.argsort(mi)[-5:]]
    X = X[selected_features]
    print("izberemo top 5")
    print(selected_features)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=4032, shuffle=False)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #X_test = scaler.transform(X_test)
    #print("X po standard scaler")
    #print(X)

    print("shape")
    print(X.shape)
    print(Y.shape)

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
    print("shape")
    print(X.shape)
    print(Y.shape)

    print("nalozimo model")
    #model = pickle.load(open('pc_lstm_test_train.h5', 'rb')) # nalozimo narejen model
    model = keras.models.load_model('pc_lstm_test_train.h5')

    predictions = model.predict(X)
    print("Predictions")
    print(predictions)

    print("avg")
    print(df_avg)
    return jsonify({"predict": float(predictions[0][0])})

@app.route('/predict1', methods=['POST'])
def predict1():
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
    selected_features = X.columns[np.argsort(mi)[-5:]]
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

    model = keras.models.load_model('model.h5')

    predictions = model.predict(X)

    print("avg")
    print(df_avg)
    return jsonify({"predict": float(predictions[0][0])})

    #return #jsonify({"predict1": float(predictions[0][0])})

@app.route('/predict2', methods=['POST'])
def predict2():
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

@app.route('/predict3', methods=['POST'])
def predict3():
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

    df = df.tail(1009)

    X = df.drop(['global energy'], axis=1)
    Y = df['global energy'].to_numpy()

    mi = mutual_info_regression(X, Y)
    selected_features = X.columns[np.argsort(mi)[-5:]]
    X = X[selected_features]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_future = 1
    n_past = 1008

    X_train_temp = []
    Y_train_temp = []
    X_test_temp = []
    Y_test_temp = []

    for i in range(n_past, len(X) - n_future + 1):
        X_train_temp.append(X[i - n_past:i, :])
        Y_train_temp.append(Y[i + n_future - 1:i + n_future])

    X = np.array(X_train_temp)
    Y = np.array(Y_train_temp)

    model = keras.models.load_model('model3.h5')

    predictions = model.predict(X)

    print("avg")
    print(df_avg)
    return jsonify({"predict": float(predictions[0][0])})

if __name__ == '__main__': 
    timestamp = 1667433600000
    dt_object = datetime.datetime.fromtimestamp(timestamp / 1000.0)
    print(dt_object)
    app.run(host='0.0.0.0', port=5000)





"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=4032, shuffle=False)
print("x train")
print(X_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("x train po standard scaler")
print(X_train)


print("nalozimo model")
model = pickle.load(open('pc_lstm_test_train.h5', 'rb')) # nalozimo narejen model

predictions = model.predict(X_test)
print("Predictions")
print(predictions)
"""
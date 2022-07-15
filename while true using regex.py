# imports
from cgi import print_exception
import numpy as np
import pandas_datareader as web
import pandas as pd
import datetime as dt
import math
import win10toast
import re


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from plyer import notification


# Notification received of the Bitcoin predition price for today
def notification(price):
    title = 'BTC Prediction Tracker',
    message = 'Todays prediction is!', price
    app_icon = None,
    timeout = 10
    ticker = 'Todays prediction is!'

# Load & analyze training data
# Year: YYYY-M-D
def time():
    start = dt.datetime(2013,1,1)
    end = dt.datetime.now()
    return start, end
def btc_predictor(start,end):
    crypto_currency = 'BTC'
    dollar_currency = 'USD'

# /// Retrieve data from yahoo finance///

    data = web.DataReader(name= "BTC-USD",data_source="yahoo", start=start)
# print(data)
# Prepare data

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
# Input should return prediction days with no more than 3 maximum digits, otherwise raise error
    while True:
        try:
           prediction_days = int(input("How many prediction days?"))
           if not re.match(r'[\d]{1,3}',str(prediction_days)):
                raise TypeError
           else:
                break
        except TypeError:
            print("Only a digit with 3 digits max is accepted, try again")
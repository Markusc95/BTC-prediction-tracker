# imports
import numpy as np
import pandas_datareader as web
import pandas as pd
import datetime as dt
import math

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from plyer import notification


# Load & analyze training data
def BTC_Predictor():
    crypto_currency = 'BTC'
    dollar_currency = 'USD'
# Year: YYYY-M-D
    start = dt.datetime(2013,1,1)
    end = dt.datetime.now()
# /// Retrieve data from yahoo finance with conversion of BTC to USD///
    data = web.DataReader(name= "BTC-USD", data_source="yahoo", start=start)
    print(data)

    

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
# Iterate through a known sequence of numbers from the Yahoo finance data Of BTC
    x_train= []
    y_train= []

    for x in range(prediction_days, len(scaled_data)):
      x_train.append(scaled_data[x-prediction_days:x, 0])
      y_train.append(scaled_data[x, 0])
 
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer = 'adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32)
  
    test_start= dt.datetime(2022,6,11)
    test_end= dt.datetime.now()
    
    crypto_currency= 'BTC'
    dollar_currency= 'USD'

    test_data = web.DataReader(f'{crypto_currency}-{dollar_currency}', 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)


    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform (prediction_prices)
    return prediction_prices

def toast(price):
    toaster = win10toast.ToastNotifier()
    toaster.show_toast("BTC Prediction Tracker", f"Today's prediction is {price}!", duration=10) 

# Result in the next prediction day
def main():
    start,end = time()
    price = btc_predictor(start, end)
    price = round(float(price[-1][0]), 2)
    print(price)
    toast(price)

if __name__ == '__main__':
    main()
    

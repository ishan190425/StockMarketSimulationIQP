from datetime import timedelta
from datetime import datetime
import random
import pandas as pd
import numpy as np
import talib as ta
import sys
import os
import matplotlib.pyplot as plt
from os import walk
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import dense
from tensorflow.keras.layers import lstm
from tensorflow.keras.layers import dropout
from tensorflow.keras.layers import flatten
from tensorflow.keras.optimizers import adam

rsi_period = 14
adx_period = 14
leftshift = 33
window = 60
padding = "--------------------------------"
data_path = "best/graphs"
data_path = "/users/ishan/coding/wpi/stock_market_simulation_iqp/datasets/30y_stock_csvs"

class hidden_prints():
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_dates():
    start = datetime(2019, 1, 1)
    start_date = datetime(1981, 1, 1)

    while start_date < start:
        year = random.randint(0, 22)
        leapyear = False

        if (year % 400 == 0) and (year % 100 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
            leapyear = True

        if year < 10:
            year = year = "200{}".format(year)

        else:
            year = "20{}".format(year)

        month = str(random.randint(1, 12))
        possible_days = {'1': 31, '2': 28, '3': 31, '4': 30, '5': 31,
                        '6': 30, '7': 31, '8': 31, '9': 30, '10': 31, '11': 30, '12': 31}
        day = str(random.randint(1, possible_days[month]))

        if leapyear and month is '2':
            day = str(random.randint(1, 29))

        start_date = '{}-{}-{}'.format(year, month, day)

        start_date = datetime.strptime(start_date, "%y-%m-%d")

    days = timedelta(76)
    end_date = start_date + days
    start_date = str(start_date)[:10]
    end_date = str(end_date)[:10]
    return start_date, end_date

def mean_se(yhat, ypred):
    return np.mean((ypred-yhat) ** 2)

def normalize(value, sum, length):

    return (value / sum)

def negative_mse(value, max):
    return (-1 * value)

def get_data(stock, start_date, end_date,variables_to_include,training=False):
    number_of_features = len(variables_to_include)
    data_path = "/users/ishan/coding/wpi/stock_market_simulation_iqp/datasets/30y_stock_csvs"
    data_train = pd.read_csv("{}/{}.csv".format(data_path, stock))  # import csv
    if training:
        training_data_points = round(len(data_train) * .4)
        data_train = data_train[training_data_points + 1:]
    else:
        data_train['datetime'] = pd.to_datetime(data_train['datetime'])
        if start_date < data_train.iloc[0, :]['datetime']:
            start_date = data_train.iloc[0, :]['datetime']
        days = timedelta(93)
        start_date = start_date - days
        if start_date < data_train.iloc[0, :]['datetime']:
            start_date = data_train.iloc[0, :]['datetime']
        data_train = data_train[~(data_train['datetime'] < start_date)]
        data_train = data_train[~(data_train['datetime'] > end_date)]
    variables_to_include = variables_to_include
    number_of_features = len(variables_to_include)
    training_data_points = round(len(data_train) * .4)
    data_train.rename(columns={'close': 'close'}, inplace=True)
    data_train.rename(columns={'high': 'high'}, inplace=True)
    data_train.rename(columns={'low': 'low'}, inplace=True)
    data_train.rename(columns={'volume': 'volume'}, inplace=True)

    data_train['close'] = data_train['close'].astype(float).fillna(0)
    data = data_train['close']
    data_train["rsi"] = ta.rsi(data_train['close'], rsi_period).fillna(0)
    data_train["adx"] = ta.adx(
        data_train['high'], data_train['low'], data_train['close'], adx_period).fillna(0)
    fastk, fastd = ta.stochf(
        data_train['high'], data_train['low'], data_train['close'])
    data_train['fastd'] = fastd
    data_train['fastk'] = fastk
    macd, macdsignal, macdhist = ta.macd(data_train['close'])
    data_train['macd'] = macd
    data_train['macdsignal'] = macdsignal
    data_train['macdhist'] = macdhist
    upper, middle, lower = ta.bbands(data_train['close'])
    data_train['bb_lowerband'] = lower
    data_train['bb_middleband'] = middle
    data_train['bb_upperband'] = upper
    new_train = pd.data_frame()
    new_train = data_train[variables_to_include]
    real_stock_price = data_train['close'].values  # convert to numpy to train rnn
    new_train["close"] = new_train['close'].shift(1)
    return real_stock_price, data, new_train, number_of_features

def convert_to_numpy(new_test,number_of_features,sc,y_sc_set = none, training=False):
    # convert to numpy to train rnn
    training_data_points = round(len(new_test) * .4)
    training_set = new_test.iloc[:, 0:number_of_features].values
    inputs = training_set
    inputs = sc.transform(inputs)
    x_test = []
    if not training:
        for i in range(window+leftshift, len(inputs)):
            x_test.append(inputs[i-window:i])
        x_test = np.array(x_test)
        # batchsize, input_size, number_of_features
        try:
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_of_features))
        except:
            return []
    else:
        x_train = []
        y_train = []
        for i in range(window+leftshift, training_data_points):
            x_train.append(inputs[i-window:i])
            y_train.append(y_sc_set[i])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # ## reshaping the data

        # batchsize, input_size, number_of_features
        x_train = np.reshape(
            x_train, (x_train.shape[0], x_train.shape[1], number_of_features))
        return x_train,y_train

    return x_test 

def get_predictions(regressor, x_test,real_stock_price, y_sc, shift):
    predicted_price = regressor.predict(x_test)
    predicted_price = y_sc.inverse_transform(predicted_price)
    diffrence = (len(real_stock_price)) - (len(predicted_price))
    real_stock_price = real_stock_price[diffrence-shift:]
    return predicted_price, real_stock_price

def calculate_profit(starting_capital, predicted_price, real_stock_price):
    stock = 5
    stocks_owned = {}
    liquid_value = starting_capital
    daily_returns = []
    sold = 0

    loss = pd.data_frame(columns=['i', 'price'])
    profit = pd.data_frame(columns=['i', 'price'])
    buy = pd.data_frame(columns=['i', 'price'])
    return_daily = 0

    for i in range(len(predicted_price)):
        if real_stock_price[i] < predicted_price[i] and stock not in stocks_owned:
            stocks_owned[stock] = (real_stock_price[i], liquid_value/real_stock_price[i])
            liquid_value -= liquid_value/real_stock_price[i] * real_stock_price[i]
            buy.loc[len(buy)] = [i, real_stock_price[i]]
        elif stock in stocks_owned and stocks_owned[stock][0] < real_stock_price[i]:
            if stocks_owned[stock][1] * real_stock_price[i] > stocks_owned[stock][1] * stocks_owned[stock][0]:
                profit.loc[len(profit)] = [i, real_stock_price[i]]
            else:
                loss.loc[len(loss)] = [i, real_stock_price[i]]
            liquid_value += stocks_owned[stock][1] * real_stock_price[i]
            percent_gain = (
                real_stock_price[i] - stocks_owned[stock][0]) / stocks_owned[stock][0]
            sold += 1
            return_daily = percent_gain
            stocks_owned.pop(stock)
        if stock in stocks_owned and i == len(predicted_price) - 1:
            if stocks_owned[stock][1] * real_stock_price[i] > stocks_owned[stock][1] * stocks_owned[stock][0]:
                profit.loc[len(profit)] = [i, real_stock_price[i]]
            else:
                loss.loc[len(loss)] = [i, real_stock_price[i]]
            liquid_value += stocks_owned[stock][1] * real_stock_price[i]
            percent_gain = (
                real_stock_price[i] - stocks_owned[stock][0]) / stocks_owned[stock][0]
            return_daily = percent_gain
            stocks_owned.pop(stock)
            sold += 1
        daily_returns.append(return_daily)

    daily_returns = pd.series(daily_returns, dtype='float')
    profit_value = (liquid_value - starting_capital)
    return profit, daily_returns ,loss, profit_value, buy, sold

def plot_data(stock, real_stock_price, predicted_price, buy, profit, loss,testing = False,layer1 = 0, layer2 = 0):
    plt.plot(real_stock_price, color='red', label="real stock price")
    plt.plot(predicted_price, color='blue', label="predicted stock price")
    if not testing:
        plt.scatter(buy['i'], buy['price'], marker="o", label = 'buy', s=50)
        plt.scatter(profit['i'], profit['price'], marker="^",
                    color='green', label="sell - profit", s=50)
        plt.scatter(loss['i'], loss['price'], marker="v",
                    color='red', label='sell - loss ', s=50)
    plt.title("{} stock price".format(stock))
    plt.xlabel('time')
    plt.ylabel("price")
    plt.legend()
    if testing:
        plt.savefig('graphs-testing/{}-{}-{}.png'.format(stock, layer1, layer2))

    else:
        plt.savefig('graphs/{}.png'.format(stock))

    plt.figure()

def calculate_ratios(daily_returns, start, end_date):
    tradingdays = (end_date-start).days
    if len(daily_returns) and daily_returns.std():
        sharpe_ratio = daily_returns.mean() / daily_returns.std()
        sharperatio = (tradingdays ** 0.5) * sharpe_ratio
    else:
        sharperatio = 0
    negs = daily_returns[daily_returns < 0]
    if len(negs) and negs.std():
        sortino_ratio = (daily_returns.mean() / negs.std()) * \
            (tradingdays ** 0.5)
    else:
        sortino_ratio = 0

    # calculate the max drawdown in the past window days for each day in the series.
    # use min_periods=1 if you want to let the first 252 days data have an expanding window
    roll_max = daily_returns.cummax()
    daily_drawdown = (daily_returns/roll_max) - 1.0

    # next we calculate the minimum (negative) daily drawdown in that window.
    # again, use min_periods=1 if you want to allow the expanding window
    max_daily_drawdown = daily_drawdown.cummin()
    calm = daily_returns.mean()
    calmer_ratio = (calm/max_daily_drawdown) * (tradingdays ** 0.5)
    calmer_ratio = 0

    return sharperatio, sortino_ratio, calmer_ratio

def get_companies():
    companies = []
    for (dirpath, dirnames, filenames) in walk(data_path):
        companies.extend(filenames)
        break
    return companies

def get_companies_weighted():
    weights = pd.read_csv("output_weights.csv")
    df = weights[['company', 'weight']]
    companies_weighted = list(df['company'])
    weight = dict(df.values)
    return companies_weighted, weight

def build_model(stock,x_train, y_train,training = False,layer1 = 0, layer2 = 0):
        regressor = sequential()

        regressor.add(lstm(units = layer1, return_sequences = True, input_shape = (x_train.shape[1],number_of_features))) 
        regressor.add(dropout(rate = 0.2))

        regressor.add(lstm(units = layer2))
        regressor.add(dropout(rate = 0.2))
  
        
        # last layer
        regressor.add(dense(units=1))#output layer, default since this is regression not classfition 

        # %% [markdown]
        # ## adding output layer

        # %%
        optimizer = adam(learning_rate=lr)
        regressor.compile(optimizer=optimizer, loss='mean_squared_error',
                          metrics='accuracy')

        # %%
        regressor.fit(x_train, y_train, epochs=20, batch_size=32,verbose=0)
        if training:
            regressor.save(
                "models-testing/{}-{}-{}-model.json".format(stock,layer1, layer2))
        else:
            regressor.save(
                "best/models/{}-model.json".format(stock))
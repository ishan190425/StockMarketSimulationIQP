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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam

rsi_period = 14
adx_period = 14
leftshift = 33
window = 60
padding = "--------------------------------"
data_path = "/Users/ishan/Coding/Wpi/StockMarketSimulationIQP/Datasets/30y_stock_csvs"
variables_to_include = ['close', 'volume',
                      "rsi", "adx", "fastd", "fastk", "macd"]

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

        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    days = timedelta(76)
    end_date = start_date + days
    start_date = str(start_date)[:10]
    end_date = str(end_date)[:10]
    return start_date, end_date


def mean_se(yhat, ypred):
    return np.mean((ypred-yhat) ** 2)


def normalize(value, sum):

    return (value / sum)


def negative_mse(value):
    return (-1 * value)


def get_data(stock, start_date, end_date, variables_to_include, training=False):
    data_train = pd.read_csv(
        "{}/{}.csv".format(data_path, stock))  # import csv
    if training:
        training_data_points = round(len(data_train) * .4)
        data_train = data_train[training_data_points + 1:]
    else:
        data_train['Datetime'] = pd.to_datetime(data_train['Datetime'])
        if start_date < data_train.iloc[0, :]['Datetime']:
            start_date = data_train.iloc[0, :]['Datetime']
        temp = data_train
        temp = temp[~(temp['Datetime'] < start_date)]
        temp = temp[~(temp['Datetime'] > end_date)]
        trading_days = len(temp)
        days = timedelta(125+(trading_days-31)+1)
        start_date = start_date - days
        if start_date < data_train.iloc[0, :]['Datetime']:
            start_date = data_train.iloc[0, :]['Datetime']
        data_train = data_train[~(data_train['Datetime'] < start_date)]
        data_train = data_train[~(data_train['Datetime'] > end_date)]
    
    variables_to_include = variables_to_include
    number_of_features = len(variables_to_include)
    training_data_points = round(len(data_train) * .4)
    if len(data_train) == 0:
        return 0, 0, 0, 0
    data_train.rename(columns={'CLOSE': 'close'}, inplace=True)
    data_train.rename(columns={'HIGH': 'high'}, inplace=True)
    data_train.rename(columns={'LOW': 'low'}, inplace=True)
    data_train.rename(columns={'VOLUME': 'volume'}, inplace=True)

    data_train['close'] = data_train['close'].astype(float).fillna(0)
    prices = data_train['close']
    data_train["rsi"] = ta.RSI(data_train['close'], rsi_period).fillna(0)
    data_train["adx"] = ta.ADX(
        data_train['high'], data_train['low'], data_train['close'], adx_period).fillna(0)
    fastk, fastd = ta.STOCHF(
        data_train['high'], data_train['low'], data_train['close'])
    data_train['fastd'] = fastd
    data_train['fastk'] = fastk
    macd, macdsignal, macdhist = ta.MACD(data_train['close'])
    data_train['macd'] = macd
    data_train['macdsignal'] = macdsignal
    data_train['macdhist'] = macdhist
    upper, middle, lower = ta.BBANDS(data_train['close'])
    data_train['bb_lowerband'] = lower
    data_train['bb_middleband'] = middle
    data_train['bb_upperband'] = upper
    indicators = pd.DataFrame()
    indicators = data_train[variables_to_include]
    # convert to numpy to train rnn
    prices_numpy = data_train['close'].values
    indicators["close"] = indicators['close'].shift(1)
    return prices_numpy, prices, indicators, number_of_features


def convert_to_numpy(new_test, number_of_features, sc, y_sc_set=None, training=False):
    # convert to numpy to train rnn
    training_data_points = round(len(new_test) * .4)
    training_set = new_test.iloc[:, 0:number_of_features].values
    raw_data = training_set
    raw_data = sc.transform(raw_data)
    indicators = []
    if not training:
        for i in range(window+leftshift, len(raw_data)):
            indicators.append(raw_data[i-window:i])
        indicators = np.array(indicators)
        # batchsize, input_size, number_of_features
        try:
            indicators = np.reshape(
                indicators, (indicators.shape[0], indicators.shape[1], number_of_features))
        except:
            return []
    else:
        indicators = []
        prices = []
        for i in range(window+leftshift, training_data_points):
            indicators.append(raw_data[i-window:i])
            prices.append(y_sc_set[i])

        indicators, prices = np.array(indicators), np.array(prices)

        # ## reshaping the data

        # batchsize, input_size, number_of_features
        indicators = np.reshape(
            indicators, (indicators.shape[0], indicators.shape[1], number_of_features))
        return indicators, prices

    return indicators


def get_predictions(regressor, x_test, real_stock_price, y_sc, shift):
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
    amount_of_times_traded = 0

    loss_array = pd.DataFrame(columns=['i', 'price','profit'])
    profit_array = pd.DataFrame(columns=['i', 'price','profit'])
    buy_array = pd.DataFrame(columns=['i', 'price'])
    return_daily = 0

    for i in range(len(predicted_price)):
        if real_stock_price[i] < predicted_price[i] and stock not in stocks_owned:
            stocks_owned[stock] = (real_stock_price[i],
                                   liquid_value/real_stock_price[i])
            liquid_value -= liquid_value / \
                real_stock_price[i] * real_stock_price[i]
            buy_array.loc[len(buy_array)] = [i, real_stock_price[i]]
        elif stock in stocks_owned and stocks_owned[stock][0] < real_stock_price[i]:
            if stocks_owned[stock][1] * real_stock_price[i] > stocks_owned[stock][1] * stocks_owned[stock][0]:
                profit_array.loc[len(profit_array)] = [i, real_stock_price[i],real_stock_price[i]-stocks_owned[stock][0]]
            else:
                loss_array.loc[len(loss_array)] = [i, real_stock_price[i],real_stock_price[i]-stocks_owned[stock][0]]
            liquid_value += stocks_owned[stock][1] * real_stock_price[i]
            percent_gain = (
                real_stock_price[i] - stocks_owned[stock][0]) / stocks_owned[stock][0]
            amount_of_times_traded += 1
            return_daily = percent_gain
            stocks_owned.pop(stock)
        if stock in stocks_owned and i == len(predicted_price) - 1:
            if stocks_owned[stock][1] * real_stock_price[i] > stocks_owned[stock][1] * stocks_owned[stock][0]:
                profit_array.loc[len(profit_array)] = [i, real_stock_price[i],real_stock_price[i]-stocks_owned[stock][0]]
            else:
                loss_array.loc[len(loss_array)] = [i, real_stock_price[i],real_stock_price[i]-stocks_owned[stock][0]]
            liquid_value += stocks_owned[stock][1] * real_stock_price[i]
            percent_gain = (
                real_stock_price[i] - stocks_owned[stock][0]) / stocks_owned[stock][0]
            return_daily = percent_gain
            stocks_owned.pop(stock)
            amount_of_times_traded += 1
        if return_daily:
            daily_returns.append(return_daily)

    daily_returns = pd.Series(daily_returns, dtype='float')
    profit_value = (liquid_value - starting_capital)
    return profit_array, daily_returns, loss_array, profit_value, buy_array, amount_of_times_traded


def plot_data(stock, real_stock_price, predicted_price, buy, profit, loss, testing=False, layer1=0, layer2=0,save=False,final = False):
    if not final:
        plt.plot(real_stock_price[:-3], color='red', label="Real stock price")
        plt.plot(predicted_price, color='blue', label="Predicted stock price")
        if not testing:
            plt.scatter(buy['i'], buy['price'], marker="o", label='buy', s=50)
            plt.scatter(profit['i'], profit['price'], marker="^",
                        color='green', label="sell - profit", s=50)
            plt.scatter(loss['i'], loss['price'], marker="v",
                        color='red', label='sell - loss ', s=50)
        plt.title("{} Stock Price".format(stock))
        plt.xlabel('Time')
        plt.ylabel("Price")
        plt.legend()
        if testing:
            plt.savefig('Graphs-testing/{}-{}-{}.png'.format(stock, layer1, layer2))
        if save:
            plt.savefig('Final/{}.png'.format(stock))
        plt.figure()
    if final:
        weeks = round(len(real_stock_price) / 8)
        for week in range(8):
            real_stock_price_week = real_stock_price[week * weeks:weeks*(week+1)]
            predicted_price_week = predicted_price[week * weeks:weeks*(week+1)]
            buy_for_the_week = convert_to_weekly(buy, weeks, week)
            profit_for_the_week = convert_to_weekly(profit, weeks, week)
            loss_for_the_week = convert_to_weekly(loss, weeks, week)
            plt.plot(real_stock_price_week,
                     color='red', label="Real stock price")
            plt.plot(predicted_price_week, color='blue', label="Predicted stock price")
            plt.scatter(buy_for_the_week['i'], buy_for_the_week['price'], marker="o", label='Buy', s=50)
            plt.scatter(profit_for_the_week['i'], profit_for_the_week['price'], marker="^",
                        color='green', label="Sell - Profit", s=50)
            plt.scatter(loss_for_the_week['i'], loss_for_the_week['price'], marker="v",
                        color='red', label='Sell - Loss ', s=50)
            plt.title("{} Stock Price - Week {}".format(stock,week))
            plt.xlabel('Days')
            plt.ylabel("Price")
            plt.figtext(0, 0, "Profit - ${}".format(round(calculate_weekly_profit(profit_for_the_week,loss_for_the_week),2)), fontsize=14)
            plt.legend()
            plt.savefig('Final/Week{}/{}-Week{}.png'.format(week,stock,week), facecolor='w')
            plt.close()
            plt.figure()
            

def calculate_weekly_profit(profit, loss):
    profits = profit['profit'].sum()
    losses = loss['profit'].sum()
    return profits + losses
    
def convert_to_weekly(original, weeks, week,debug = False):
    new = original[original['i'] >= (weeks * week)]
    new = new[new['i'] < (weeks * (week + 1))]
    new_adjusted_dates = new
    new_adjusted_dates['i'] = new_adjusted_dates['i'] - (weeks * week)
    if debug:
        print(f'Week-{week}, Weeks-{weeks}')
        print("Orignal")
        print(original)
        print("New")
        print(new)
        print(new_adjusted_dates)
        print("___________")
    return new_adjusted_dates

    


def calculate_ratios(daily_returns, start, end_date):
    tradingdays = (end_date-start).days
    if len(daily_returns) and daily_returns.std():
        sharpe_ratio = daily_returns.mean() / daily_returns.std()
        sharperatio = (252 ** 0.5) * sharpe_ratio
    else:
        sharperatio = 0
    negs = daily_returns[daily_returns < 0]
    if len(negs) and negs.std():
        sortino_ratio = (daily_returns.mean() / negs.std()) * (252 ** 0.5)
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
    calmer_ratio = (calm/max_daily_drawdown) * (252 ** 0.5)
    calmer_ratio = 0

    return sharperatio, sortino_ratio, calmer_ratio


def get_companies():
    companies = []
    for (_, _, filenames) in walk(data_path):
        companies.extend(filenames)
        break
    return companies


def get_companies_weighted():
    weights = pd.read_csv("output_weights.csv")
    df = weights[['Company', 'Weight']]
    companies_weighted = list(df['Company'])
    weight = dict(df.values)
    return companies_weighted, weight


def build_model(stock, x_train, y_train,number_of_features, training=False, layer1=0, layer2=0):
    regressor = Sequential()

    regressor.add(LSTM(units=layer1, return_sequences=True,
                  input_shape=(x_train.shape[1], number_of_features)))
    regressor.add(Dropout(rate=0.2))

    regressor.add(LSTM(units=layer2))
    regressor.add(Dropout(rate=0.2))

    # last layer
    # output layer, default since this is regression not classfition
    regressor.add(Dense(units=1))

    # ## adding output layer

    optimizer = Adam(learning_rate=0.01)
    regressor.compile(optimizer=optimizer, loss='mean_squared_error',
                      metrics='accuracy')

    regressor.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0)
    if training:
        regressor.save(
            "models-testing/{}-{}-{}-model.json".format(stock, layer1, layer2))
    else:
        regressor.save(
            "best/models/{}-model.json".format(stock))

#!/usr/bin/env python
# coding: utf-8

# in[1]:


from sklearn.metrics import mean_squared_error as mse
import warnings
from pandas.core.common import setting_with_copy_warning
warnings.simplefilter(action="ignore", category=setting_with_copy_warning)
from os import walk
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.fatal)
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime,timedelta
from sklearn.preprocessing import min_max_scaler
from math import floor
import pandas as pd
from tqdm import tqdm
from helperFunctions import *
import helperFunctions




companies = get_companies()
companies_weighted, weight = get_companies_weighted()




def company(stock,start_date, end_date,starting_capital,plot = False, s = 1):
    model = "best/models/{}-model.json".format(stock)

    start = start_date
    
   
    variables_to_include = ['close', 'volume',"rsi", "adx", "fastd", "fastk", "macd"]
    
    _, data, new_train,number_of_features = get_data(stock,start_date,end_date,variables_to_include)
    
    training_set = new_train.iloc[:,0:number_of_features].values #convert to numpy to train rnn
    y_set = data.astype(float).values.reshape(-1, 1)

    # ## feature scaling

    # use normalization x - min(x) / max(min) - min(x)
    sc = min_max_scaler(feature_range=(0,1)) # all values between 0 and 1
    y_scaler = min_max_scaler(feature_range=(0,1))
    _ = y_scaler.fit_transform(y_set)
    _ = sc.fit_transform(training_set)

    regressor=tf.keras.models.load_model(model)
    
    real_stock_price,_,new_test,number_of_features = get_data(stock,start_date,end_date,variables_to_include) #convert to numpy to train rnn
    
    x_test = convert_to_numpy(new_test, number_of_features, sc)
    
    if not len(x_test):
        return 0,0,0,0,0
    
    predicted_price, real_stock_price = get_predictions(regressor,x_test,real_stock_price, y_scaler, s)
    # %% [markdown]
    # ## predict price

    profit_series, daily_returns, loss, profit, buy, sold = calculate_profit(
        starting_capital, predicted_price, real_stock_price)
    
    
    if plot:
        plot_data(stock, real_stock_price, predicted_price, buy, profit_series, loss)

    
    sharpe_ratio, sortino_ratio, calmer_ratio = calculate_ratios(daily_returns, start, end_date)
    
    

    return profit,sharpe_ratio,sortino_ratio,calmer_ratio,sold


# in[4]:


def eval(start_date="1982-3-12", end_date="2022-02-1",weighted = false, companies = companies, s = 1):
    start_date = datetime.strptime(str(start_date), "%y-%m-%d")
    end_date = datetime.strptime(str(end_date), "%y-%m-%d")
    start_date_threshold = datetime.strptime("1982-3-12", "%y-%m-%d")
    spstart = start_date

    if start_date_threshold > start_date:
        spstart = start_date_threshold
    profit = 0

    with hidden_prints():
        sp500 = yf.download('^gspc',spstart,end_date)

    profit_sp500_percentage = ((sp500['close'][-1] - sp500['close'][0])/sp500['close'][0])*100
    profit_sp500 = (500_000) * (profit_sp500_percentage/100)

    profit = 0
    sharpe_ratio = []
    sortino_ratio = []
    calmer_ratio = []

    top_stock = ""
    max_profit = float('-inf')
    lowest_stock = ""
    min_profit = float('inf')

    index = 0
    sold = 0
    value = 500_000

    for i in tqdm(companies):
        #with hidden_prints():
        index += 1
    

        if weighted:    
            weight_c = weight[i]
            starting_capital = 500_000 * weight_c
            profit_c, sharperatio_c, sortino_ratio_c, calmer_ratio_c,sold_c = company(i, start_date, end_date, starting_capital, s = s)
            sharperatio_c *= weight_c
            sortino_ratio_c *= weight_c
            calmer_ratio_c *= weight_c

        else:
            i = i[:-4]
            print(i)
            profit_c, sharperatio_c, sortino_ratio_c, calmer_ratio_c, sold_c = company(
                i, start_date, end_date,s = s)

        profit += profit_c
        sharpe_ratio.append(sharperatio_c)
        sortino_ratio.append(sortino_ratio_c)
        calmer_ratio.append(calmer_ratio_c)
        sold += sold_c
        if profit_c < min_profit:
            min_profit = profit_c
            lowest_stock = i

        if profit_c > max_profit:
            max_profit = profit_c
            top_stock = i
    
    sharpe_ratio = pd.series(sharpe_ratio).dropna()
    sortino_ratio = pd.series(sortino_ratio).dropna()
    
    profit_percentage = (profit/(value))*100
    profit = round(profit,2)
    profit_percentage = round(profit_percentage,2)
    profit_sp500 = round(profit_sp500,2)
    profit_sp500_percentage = round(profit_sp500_percentage,2)
    sharpe_ratio = round((sharpe_ratio.mean()),2)
    sortino_ratio = round((sortino_ratio.mean()),2)
    calmer_ratio = 0
    min_profit = round(min_profit,2)
    max_profit = round(max_profit,2)

    if profit_sp500 < profit:
        beat = True
    else:
        beat = False
        

    print("start date: {}".format(str(start_date.date())))
    print("end date: {}".format(str(end_date.date())))
    print("profit: ${}".format(profit))
    print("roi: {}%".format(profit_percentage))
    print("most profitable stock: {}".format(top_stock))
    print("profit for {}: ${}".format(top_stock,max_profit))
    print("least profitable stock: {}".format(lowest_stock))
    print("profit for {}: ${}".format(lowest_stock,min_profit))
    print("profit s&p500: ${}".format(profit_sp500))
    print("s&p500 roi: {}%".format(profit_sp500_percentage))
    print("sharpe ratio: {}".format(sharpe_ratio))
    print("sortino ratio: {}".format(sortino_ratio))
    print("calmer ratio: {}".format(calmer_ratio))
    print("traded {} times".format(sold))
    return profit, beat




total_wins = 0
epochs = 10

dates_eval = []
with hidden_prints():
    for i in tqdm(range(epochs)):
        start_date, end_date = get_dates()
        dates_eval.append(start_date)
        _, win = eval(start_date = start_date, end_date = end_date, s = 3,weighted=True, companies=companies_weighted)
        if win:
            total_wins += 1

if total_wins > 5:
    print(f"you beat s&p500 {total_wins} times out of {epochs} times.")
else:
    print(f"s&p500 beat you {epochs-total_wins} times out of {epochs} times.")

print(f'evaluvated on dates {dates_eval}')





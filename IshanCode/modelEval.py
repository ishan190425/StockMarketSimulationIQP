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
from sklearn.preprocessing import MinMaxScaler
from math import floor
import pandas as pd
from tqdm import tqdm
from helperFunctions import *
import helperFunctions


companies = get_companies()
companies_weighted, weights_per_company = get_companies_weighted()


def evaluvate_company(stock, start_date, end_date, starting_capital, plot= False, shift = 1, final = False):
    model = "Best/Models/{}-model.json".format(stock)

    real_stock_price, _, technical_indicators, number_of_features = get_data(
        stock, start_date, end_date, variables_to_include)

    if not number_of_features:
        return 0, 0, 0, 0, 0
    # convert to numpy to train RNN
    indicators = technical_indicators.values
    prices = real_stock_price.reshape(-1, 1)

    # ## Feature Scaling

    # Use normalization x - min(x) / max(min) - min(x)
    indicator_scaler = MinMaxScaler(feature_range=(0, 1))  # all values between 0 and 1
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    _ = price_scaler.fit_transform(prices)
    _ = indicator_scaler.fit_transform(indicators)

    price_predictor = tf.keras.models.load_model(model)

    indciators_as_numpy = convert_to_numpy(technical_indicators, number_of_features, indicator_scaler)

    if not len(indciators_as_numpy):
        return 0, 0, 0, 0, 0

    predicted_price, real_stock_price = get_predictions(
        price_predictor, indciators_as_numpy, real_stock_price, price_scaler, shift)
    # %% [markdown]
    # ## Predict price

    profit_series, daily_returns, loss, profit, buy, sold = calculate_profit(
        starting_capital, predicted_price, real_stock_price)

    if plot:
        plot_data(stock, real_stock_price, predicted_price,
                  buy, profit_series, loss, final=final)

    sharpe_ratio, sortino_ratio, calmer_ratio = calculate_ratios(
        daily_returns, start_date, end_date)

    return profit, sharpe_ratio, sortino_ratio, calmer_ratio, sold


def evaluvate_portfolio_for_profit_and_ratios(start_date="1982-3-12", end_date="2022-02-1", weighted=False, companies=companies, shift=1):
    start_date = datetime.strptime(str(start_date), "%Y-%m-%d")
    end_date = datetime.strptime(str(end_date), "%Y-%m-%d")
    startDateThreshold = datetime.strptime("1982-3-12", "%Y-%m-%d")
    spstart = start_date

    if startDateThreshold > start_date:
        spstart = startDateThreshold
    profit = 0

    with hidden_prints():
        sp500 = yf.download('^GSPC', spstart, end_date)

    try:
        profitsp500percentage = (
            (sp500['Close'][-1] - sp500['Close'][0])/sp500['Close'][0])*100
        profitsp500 = (500_000) * (profitsp500percentage/100)
    except:
        profitsp500percentage = -1
        profitsp500 = -1
    profit = 0
    sharpe_ratio = []
    sortino_ratio = []
    calmer_ratio = []
    profits = {}

    top_stock = ""
    max_profit = float('-inf')
    worst_stock = ""
    min_profit = float('inf')

    index = 0
    amount_of_times_sold = 0
    value = 500_000

    for i in tqdm(companies):
        #with HiddenPrints():
        index += 1

        if weighted:
            weight_company = weights_per_company[i]
            starting_capital = 500_000 * weight_company
            profit_c, sharperatio_c, sortino_ratio_c, calmer_ratio_c, amount_of_times_sold_c = evaluvate_company(
                i, start_date, end_date, starting_capital, shift=shift, final=True, plot=False)

        else:
            i = i[:-4]
            print(i)
            profit_c, sharperatio_c, sortino_ratio_c, calmer_ratio_c, amount_of_times_sold_c = evaluvate_company(
                i, start_date, end_date, shift=shift)

        profit += profit_c
        profits[profit_c] = i
        sharpe_ratio.append(sharperatio_c)
        sortino_ratio.append(sortino_ratio_c)
        calmer_ratio.append(calmer_ratio_c)
        amount_of_times_sold += amount_of_times_sold_c
        if profit_c < min_profit:
            min_profit = profit_c
            worst_stock = i

        if profit_c > max_profit:
            max_profit = profit_c
            top_stock = i

    sharpe_ratio = pd.Series(sharpe_ratio).dropna()
    sortino_ratio = pd.Series(sortino_ratio).dropna()

    profit_percentage = (profit/(value))*100
    profit = round(profit, 2)
    profit_percentage = round(profit_percentage, 2)
    profitsp500 = round(profitsp500, 2)
    profitsp500percentage = round(profitsp500percentage, 2)
    sharpe_ratio = round((sharpe_ratio.mean()), 2)
    sortino_ratio = round((sortino_ratio.mean()), 2)
    calmer_ratio = 0
    min_profit = round(min_profit, 2)
    max_profit = round(max_profit, 2)

    if profitsp500 < profit:
        beat = True
    else:
        beat = False

    print("Start Date: {}".format(str(start_date.date())))
    print("End Date: {}".format(str(end_date.date())))
    print("Profit: ${}".format(profit))
    print("ROI: {}%".format(profit_percentage))
    print("Most Profitable Stock: {}".format(top_stock))
    print("Profit for {}: ${}".format(top_stock, max_profit))
    print("Least Profitable Stock: {}".format(worst_stock))
    print("Profit for {}: ${}".format(worst_stock, min_profit))
    print("Profit S&P500: ${}".format(profitsp500))
    print("S&P500 ROI: {}%".format(profitsp500percentage))
    print("Sharpe Ratio: {}".format(sharpe_ratio))
    print("Sortino Ratio: {}".format(sortino_ratio))
    print("Calmer Ratio: {}".format(calmer_ratio))
    print("Traded {} Times".format(amount_of_times_sold))
    return profit, beat








def main():

    companies = get_companies()
    
    companies_weighted, weights_per_company = get_companies_weighted()
    
    total_wins = 0

    epochs = 10

    dates_eval = []
    with hidden_prints():
        for i in tqdm(range(epochs)):
            start_date, end_date = get_dates()
            dates_eval.append(start_date)
            _, win = evaluvate_portfolio_for_profit_and_ratios(
                start_date=start_date, end_date=end_date, shift=3, weighted=True, companies=companies_weighted)
            if win:
                total_wins += 1

    if total_wins > 5:
        print(f"you beat s&p500 {total_wins} times out of {epochs} times.")
    else:
        print(f"s&p500 beat you {epochs-total_wins} times out of {epochs} times.")

    print(f'evaluvated on dates {dates_eval}')

#!/usr/bin/env python



from sklearn.metrics import mean_squared_error as MSE
from datetime import datetime, timedelta
import random
import shutil
from tqdm import tqdm
import sys
from math import floor
from sklearn.preprocessing import MinMaxScaler
import talib as ta
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import walk
import os
from scipy import stats
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
from helperFunctions import *
import modelEval



def company(stock, start_date, end_date, s=1):
    starting_capital = 500_000
    profit,sharperatio,sortino_ratio,calmer_ratio,sold = modelEval.company(stock,start_date, end_date,starting_capital,plot = False, s=s)
    profit_percent = (profit / starting_capital)
    return profit_percent


# in[63]:


def evaluvate_comapnies(debug = False):
    companies = get_companies()
    weights = {}
    for idx, i in enumerate(companies):
        count = 0
        profit_avg = 0
        i = i[:-4]
        while count < 5:
            profit = -1
            while profit == -1:
                start_date, end_date = get_dates()
                profit = company(i,start_date,end_date)
            profit_avg += profit
            count += 1
        profit_avg = profit_avg/count
        weights[i] = profit_avg
        if debug:
            print(f'number: {idx+1} company: {i} profit: {round(profit_avg * 100,2)}%')
        weights = pd.data_frame(list(weights.items()), columns=["company", "profit"])
    return weights


# in[64]:


weights = evaluvate_comapnies()


# in[69]:


def transform_weights(weights):
    return MinMaxScaler().fit_transform(weights['profit'].values.reshape(-1, 1))

weights['transformed profit'] = transform_weights(weights)


# in[70]:


weights.describe()


# in[72]:


def normalize_weights(weights):
    sums = weights['transformed mse'].sum()
    return weights.apply(lambda row: normalize(
        row['transformed profit'], sums, len(weights)), axis=1)
        
weights['weight'] = normalize_weights(weights)
weights.to_csv("output.csv", index=False)
weights['weight'].sum()


# in[73]:


new_weights = weights
new_weights = new_weights[new_weights['profit'] > 0.05] 

new_weights['transformed profit'] = transform_weights(new_weights)

new_weights['weight'] = normalize_weights(new_weights)


# in[74]:


new_weights.reset_index().plot.scatter(x='index', y='profit')


# in[75]:


new_weights['profit'].plot.density()


# in[76]:


new_weights[new_weights['weight'] == new_weights['weight'].max()]


# in[52]:


new_weights.to_csv("output_weights.csv", index=False)


# in[ ]:





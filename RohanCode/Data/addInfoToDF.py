import pandas as pd
from datetime import datetime, timedelta
import csv
from config import *
from os import walk
import talib as ta


def main():
    f = []
    for (dirpath, dirnames, filenames) in walk("C:/Users/rohan/Desktop/SchoolWork/StockMarketSimulationIQP/RohanCode/Data/clean_30y_stock_csvs/"):
        f.extend(filenames)
        break


    for stock in f:
        stock = stock[:-4]
        df = pd.read_csv("C:/Users/rohan/Desktop/SchoolWork/StockMarketSimulationIQP/RohanCode/Data/clean_30y_stock_csvs/{}.csv".format(stock))
        df = df.rename(columns={"CLOSE": "Adj Close", "VOLUME": "Volume"})
        df['Price_Moving_Avg'] = df['Adj Close'].rolling(window=30).mean()
        df['Vol_Moving_Avg'] = df['Volume'].rolling(window=30).mean()
        df = df[df['Price_Moving_Avg'].notna()]
        df['ADX'] = ta.ADX(df['HIGH'],df['LOW'],df['Adj Close'], timeperiod=14)
        df['RSI'] = ta.RSI(df['Adj Close'], timeperiod=14)
        df['Price Lower than MAVG'] = df['Price_Moving_Avg'].gt(df['Adj Close'])
        df['Volume Higher than MAVG'] = df['Vol_Moving_Avg'].gt(df['Volume'])
        #print(df)
        df.to_csv("C:/Users/rohan/Desktop/SchoolWork/StockMarketSimulationIQP/RohanCode/Data/filter_30y_stock_csvs/{}.csv".format(stock), index = False, header=True)

main()

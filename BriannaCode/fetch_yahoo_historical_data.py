#!/usr/bin/env python 
"""
Retrieve intraday stock data from Yahoo Finance.
"""

import requests
import pandas as pd
import arrow
import datetime
import argparse
import shutil
from config import *

failed_fetch_file = "failed_fetch.csv"
retry_fetch_file = "retry_fetch.csv"

## Using a data range of 30 years gets a sufficient amount of data per stock for neural network training.
## Function based on: https://gist.github.com/lebedov/f09030b865c4cb142af1#gistcomment-2674318
def get_historical_quote_data(symbol, data_range='30y', data_interval='1d'):
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals())
    #print(url) # Uncomment this if you would like to check the url used.
    headers = {'User-Agent': User_Agent}
    res = requests.get(url,
                       headers = headers)
    data = res.json()
    body = data['chart']['result'][0]
    dt = datetime.datetime
    #print(body) # Uncomment this if you would like to inspect the response body
    # to check the data was successfully retrieved.
    dt = pd.Series(map(lambda x: arrow.get(x).to('Asia/Calcutta').datetime.replace(tzinfo=None), body['timestamp']), name='Datetime')
    df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
    dg = pd.DataFrame(body['timestamp']) #TODO: Check if this is necessary.
    df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
    df.dropna(inplace=True) # Remove NaN rows.
    df.columns = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] # Rename columns in pandas dataframe.
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry', default = False)
    args = parser.parse_args()
    if args.retry:
        shutil.copyfile(failed_fetch_file, retry_fetch_file)
        csv_file = retry_fetch_file
    else:
        csv_file = stock_symbols_csv_file
    stocks_to_fetch = pd.read_csv(csv_file)
    not_finished_file = open(failed_fetch_file, "w")
    headers = ",".join(stocks_to_fetch.columns.values)
    #print(headers) # Uncomment this to see what the headers will look like.
    not_finished_file.write(headers + "\n")
    for index, row in stocks_to_fetch.iterrows():
        print(index)
        symbol = row["Symbol"]
        try:
            data = get_historical_quote_data(symbol)
            #print(data) # Uncomment this to see what the stored data will look like.
            data.to_csv(raw_directory + '/' + symbol + '.csv')
        except Exception as exception:
            print(exception)
            row_string = ",".join(row.values)
            print("Could not fetch " + row_string)
            not_finished_file.write(row_string + "\n")
    not_finished_file.close()

main()

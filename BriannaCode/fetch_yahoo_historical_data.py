#!/usr/bin/env python 
"""
Retrieve intraday stock data from Yahoo Finance.
"""

import requests
import pandas as pd
import arrow
import datetime

# Replace the following with your own User Agent
User_Agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
# Directory to store fetch data in csv files:
#stock_csvs_directory = '30y_stock_csvs'
stock_csvs_directory = 'trying_without_indicies_30y_stock_csvs'

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
    dg = pd.DataFrame(body['timestamp'])    
    df = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
    df.dropna(inplace=True) # Remove NaN rows.
    df.columns = ['OPEN','HIGH','LOW','CLOSE','VOLUME'] # Rename columns in pandas dataframe.
    
    return df

def main(csv_file="500_Stocks.csv"):
    stocks_to_fetch = pd.read_csv(csv_file)
    not_finished_file = open("failed_fetch.csv", "w")
    headers = ",".join(stocks_to_fetch.columns.values)
    print(headers)
    not_finished_file.write(headers + "\n")
    for _, row in stocks_to_fetch.iterrows():
        symbol = row["Symbol"]
        print(row)
        print(symbol)
        try:
            data = get_historical_quote_data(symbol)
            #print(data) # Uncomment this to see what the stored data will look like.
            data.to_csv(stock_csvs_directory + '/' + symbol + '.csv')
        except:
            row_string = ",".join(row.values)
            print("Could not fetch " + row_string)
            not_finished_file.write(row_string + "\n")
    not_finished_file.close()

#main()
main("retry_fetch.csv")
# mv failed_fetch.csv to retry_fetch.csv and uncomment above to retry any failed fetches

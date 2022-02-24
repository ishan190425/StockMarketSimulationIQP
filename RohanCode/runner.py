import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
import datetime
from datetime import date, timedelta
import math
import matplotlib as mpl
from os import walk

def trade(symbol): 
    StartingAmount = (500_000/505)

    # %% [markdown]
    # import libaries

    # %%
    stock = symbol
    print(stock)
    df = pd.read_csv("C:/Users/rohan/Desktop/SchoolWork/StockMarketSimulationIQP/RohanCode/Data/30y_stock_csvs/{}".format(stock))
    df = df.rename(columns={"CLOSE": "Adj Close", "VOLUME": "Volume"})
    df

    # %%
    df['Price_Moving_Avg'] = df['Adj Close'].rolling(window=30).mean()
    df['Vol_Moving_Avg'] = df['Volume'].rolling(window=30).mean()


    # %%
    daysHistory = len(df['Adj Close'])
    df = df[df['Price_Moving_Avg'].notna()]

    df

    # %%
    close_price = df["Adj Close"]
    mavgplot = df["Price_Moving_Avg"]
    vmagplot = df["Vol_Moving_Avg"]
    volumePlot = df['Volume']
    mpl.rc('figure',figsize=(15,10))
    #mpl.style.use('ggplot')

    #close_price.plot(label=(stock+" Price"),legend=True,color='blue')
    #mavgplot.plot(label = 'Moving Avg of Price',legend=True,color='orange')
    #vmagplot.plot(secondary_y=True,label='Volume Avg',legend = True,color='red')
    #volumePlot.plot(label ='Volume',secondary_y=True,legend =False,color = 'green')
    #mpl.pyplot.legend(loc="upper right")
    #mpl.pyplot.show()
    #AddPriceMoving Avg

    # %%
   #vmagplot.plot(secondary_y=False,label='Volume Avg',legend = True,color='red')
    #volumePlot.plot(label ='Volume',secondary_y=False,legend =False,color = 'green')
    #mpl.pyplot.legend()
    #mpl.pyplot.show()

    # %%
    #close_price.plot(label=(stock+" Price"),legend=True,color='blue')
    #mavgplot.plot(label = 'Moving Avg of Price',legend=True,color='orange')
    #mpl.pyplot.legend()
    #mpl.pyplot.show()

    # %%
    df['Price Lower than MAVG'] = df['Price_Moving_Avg'].gt(df['Adj Close'])
    # adj close . lt - Price moving

    df['Volume Higher than MAVG'] = df['Vol_Moving_Avg'].gt(df['Volume'])

    df

    # %%
    z=1
    PL=0.00
    Total_Gain =0
    starting_price =1
    #PG -> Per_Gain

    Start_Price = (df['Adj Close'].head(1))
    Start_Price = float(Start_Price)
    #print("Start Price:", Start_Price)

    End_Price = (df['Adj Close'].tail(1))
    End_Price = float(End_Price)
    #print("End Price:", End_Price)

    Return = (PL/Start_Price)
    Return_Per = "{:.2%}".format(Return)



    benchRe = End_Price - Start_Price
    benchREP = (benchRe/Start_Price) 
    TotalPro = (benchREP) * StartingAmount
    #print(TotalPro)

    # %%
    index =0
    for date_var,row in df.iterrows():
        maxValue = df.iloc[index-30:index]['Adj Close'].max()
        if(math.isnan(maxValue)):
            maxValue = 0
        if row['Volume Higher than MAVG']==1:
            if row['Price Lower than MAVG']==1:
                if z==1:
                    #print(date_var,row['Adj Close'], '- BUY')
                    close_adj = row['Adj Close']
                    starting_price = close_adj
                    z -=1
                    
        elif(row['Adj Close']>=(maxValue) and maxValue !=0):
            if z==1:
                    #print(date_var,row['Adj Close'], '- BUYING BC OF TREND INCREASE')
                    close_adj = row['Adj Close']
                    starting_price = close_adj
                    z -=1
        else:
            if row['Volume Higher than MAVG']==0 and row['Price Lower than MAVG']==0 :
                    if z==0:
                        #print(date_var,row['Adj Close'],'- SELL')
                        close_adj = row['Adj Close']
                        single_trade_percent_gain = ((close_adj - starting_price) / starting_price) * 100
                        Total_Gain += single_trade_percent_gain

                        #print("This trade gain/loss results: "+str(round(single_trade_percent_gain,2))+"%")
                        #print()
                        z+=1
            else:
                if (((row['Adj Close']- starting_price)/starting_price) * 100) > .5:
                    if z==0:
                        #print(date_var,row['Adj Close'],'- SELL DUE TO PRICE INCREASE')
                        close_adj = row['Adj Close']
                        single_trade_percent_gain = ((close_adj - starting_price) / starting_price) * 100
                        Total_Gain += single_trade_percent_gain
                        #print("This trade gain/loss results: "+str(round(single_trade_percent_gain,2))+"%")
                        #print()
                        z+=1
        index +=1

    if(z==0):
        #print(date_var,row['Adj Close'],'- SELL DUE TO LAST DAY')
        close_adj = row['Adj Close']
        single_trade_percent_gain = ((close_adj - starting_price) / starting_price) * 100
        Total_Gain += single_trade_percent_gain
        #print("This trade gain/loss results: "+str(round(single_trade_percent_gain,2))+"%")

    Hold_Return = (End_Price - Start_Price)
    Hold_Return_Per = "{:.2%}".format((End_Price-Start_Price)/Start_Price)
    TotalReturn = (Total_Gain/100) * StartingAmount
    #print()
    #print("The return for holding start to end was: "+str(Hold_Return_Per)+".")
    #print()
    #print("Return percentage from all trades: " + str(round(Total_Gain,2))+"% based on "+str(daysHistory)+" days of data")
    #print("With a starting amount of: $"+str(round(StartingAmount,3))+" it ended with a profit of: $"+str(round(TotalReturn,3)))
    #print("This algoritm has gained you: $"+str(round(StartingAmount+TotalReturn,3)))
    print(round(TotalReturn,2))
    print()
    return round(TotalReturn,2)

def main():
    f = []
    money = {}
    for (dirpath, dirnames, filenames) in walk("C:/Users/rohan/Desktop/SchoolWork/StockMarketSimulationIQP/RohanCode/Data/30y_stock_csvs/"):
        f.extend(filenames)
        break
    
    for stock in f:
        money[stock] = trade(stock)

    sum = 0
    for s in money:
        sum+= money[s]
        
    print(sum)

if __name__ =='__main__':
    main()

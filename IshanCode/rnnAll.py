# %% [markdown]
# # Part1 - Preprocessing

# %% [markdown]



# %%
from datetime import datetime, timedelta
import imp
import absl.logging
import warnings
import csv
import os
from os import walk
from eth_utils import combine_argument_formatters
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
import talib as ta
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
absl.logging.set_verbosity(absl.logging.ERROR)
import shutil
import os
from tqdm import tqdm


padding = "--------------------------------"
# %% [markdown]
# ## Import training set


def rnn(stock, startDate="1982-3-12", endDate="2022-02-1", lr=0.01, layer1=50, layer2=50, layer3=50, layer4=50):  # %%
    startDate = datetime.strptime(str(startDate), "%Y-%m-%d")
    endDate = datetime.strptime(str(endDate), "%Y-%m-%d")
    rsiPeriod = 14
    adxPeriod = 14
    bollingerBandWindow = 20
    shift = 1
    leftshift = 33
    window = 60

    # %%
    variablesToInclude = ['Close','Volume',"RSI","ADX","fastd","fastk","macd"]
    numberOfFeatures = len(variablesToInclude)
    numberOfFeatures

    dataPath = "/Users/ishan/Coding/Wpi/StockMarketSimulationIQP/Datasets/30y_stock_csvs"

    # %%
    dataTrain = pd.read_csv("{}/{}.csv".format(dataPath,stock)) #import csv
    temp = dataTrain
    trainingDataPoints = round(len(dataTrain) * .4)
    if trainingDataPoints < 93:
        return 0
    dataTrain.rename(columns = {'CLOSE':'Close'}, inplace = True)
    dataTrain.rename(columns = {'HIGH':'High'}, inplace = True)
    dataTrain.rename(columns = {'LOW':'Low'}, inplace = True)
    dataTrain.rename(columns = {'VOLUME':'Volume'}, inplace = True)


    # %%

    dataTrain['Close'] = dataTrain['Close'].astype(float).fillna(0)
    data = dataTrain['Close']

    dataTrain["RSI"] = ta.RSI(dataTrain['Close'],rsiPeriod).fillna(0)
    dataTrain["ADX"] = ta.ADX(dataTrain['High'],dataTrain['Low'],dataTrain['Close'],adxPeriod).fillna(0)
    fastk, fastd = ta.STOCHF(dataTrain['High'],dataTrain['Low'],dataTrain['Close'])
    dataTrain['fastd'] = fastd
    dataTrain['fastk'] = fastk
    macd, macdsignal, macdhist = ta.MACD(dataTrain['Close'])
    dataTrain['macd'] = macd
    dataTrain['macdsignal'] = macdsignal
    dataTrain['macdhist'] = macdhist
    upper,middle,lower = ta.BBANDS(dataTrain['Close'])
    dataTrain['bb_lowerband'] = lower
    dataTrain['bb_middleband'] = middle
    dataTrain['bb_upperband'] = upper

    # %%
    newTrain = pd.DataFrame()
    newTrain = dataTrain[variablesToInclude]
    newTrain["Close"] = newTrain['Close'].shift(1)

    # %%

    trainingSet = newTrain.iloc[:,0:numberOfFeatures].values #convert to numpy to train RNN
    ySet = data.astype(float).values.reshape(-1, 1)

    # %% [markdown]
    # ## Feature Scaling

    # %%
    # Use normalization x - min(x) / max(min) - min(x)
    sc = MinMaxScaler(feature_range=(0,1)) # all values between 0 and 1
    ySC = MinMaxScaler(feature_range=(0,1))
    ySCSet = ySC.fit_transform(ySet)
    scaleTrainingSet = sc.fit_transform(trainingSet)

    # %% [markdown]
    # ## Create a data structure woth 60 timesteps and 1 output

    # %%
    # Look at the 60 previous timesteps to predict this timestep
    xTrain = []
    yTrain = []
    for i in range(window+leftshift,trainingDataPoints):
        xTrain.append(scaleTrainingSet[i-window:i])
        yTrain.append(ySCSet[i])

    # %%
    # convert xtrain and yTrain to numpy for RNN
    xTrain, yTrain = np.array(xTrain), np.array(yTrain)

    # %% [markdown]
    # ## Reshaping the data

    # %%

    xTrain = np.reshape(xTrain, (xTrain.shape[0],xTrain.shape[1],numberOfFeatures)) #batchsize, inputSize, numberOfFeatures

    # %% [markdown]
    # # Part 2 - Build RNN

    # %% [markdown]HHhH
    model = "Best/Models/{}-model.json".format(stock)
    if os.path.exists(model):
        regressor=tf.keras.models.load_model(model)
        print("loaded")
   
    else:
        # %%
        regressor = Sequential()

        regressor.add(LSTM(units = layer1, return_sequences = True, input_shape = (xTrain.shape[1],numberOfFeatures))) 
        regressor.add(Dropout(rate = 0.2))
        #regressor.add(Dense(units=16,activation = 'relu',input_shape = (xTrain.shape[1],numberOfFeatures)))

        regressor.add(LSTM(units = layer2))
        regressor.add(Dropout(rate = 0.2))
  
        
        # Last Layer
        regressor.add(Dense(units=1))#output layer, default since this is regression not classfition 

        # %% [markdown]
        # ## Adding output layer

        # %%
        optimizer = Adam(learning_rate=lr)
        regressor.compile(optimizer=optimizer, loss='mean_squared_error',
                          metrics='accuracy')

        # %%
        regressor.fit(xTrain, yTrain, epochs=20, batch_size=32,verbose=0)
        
        regressor.save(
            "Models-Testing/{}-{}-{}-model.json".format(stock,layer1, layer2))
    # %% [markdown]
    # ## Part 3 - Predictions and visualing the results

    # %%
    dataTest = temp
    dataTest.rename(columns = {'CLOSE':'Close'}, inplace = True)
    dataTest.rename(columns = {'HIGH':'High'}, inplace = True)
    dataTest.rename(columns = {'LOW':'Low'}, inplace = True)
    dataTest.rename(columns = {'VOLUME':'Volume'}, inplace = True)

    # %%

    dataTest = dataTest[trainingDataPoints + 1:]
    dataTest['Close'] = dataTest['Close'].astype(float)
    dataTest["RSI"] = ta.RSI(dataTest['Close'],rsiPeriod)
    dataTest["ADX"] = ta.ADX(dataTest['High'],dataTest['Low'],dataTest['Close'],adxPeriod)
    fastk, fastd = ta.STOCHF(dataTest['High'],dataTest['Low'],dataTest['Close'])
    dataTest['fastd'] = fastd
    dataTest['fastk'] = fastk
    macd, macdsignal, macdhist = ta.MACD(dataTest['Close'])
    dataTest['macd'] = macd
    dataTest['macdsignal'] = macdsignal
    dataTest['macdhist'] = macdhist
    upper,middle,lower = ta.BBANDS(dataTest['Close'])
    dataTest['bb_lowerband'] = lower
    dataTest['bb_middleband'] = middle
    dataTest['bb_upperband'] = upper

    
    # %%
    newTest = pd.DataFrame()
    newTest = dataTest[variablesToInclude]
    

    # %%

    realStockPrice = dataTest['Close'].values #convert to numpy to train RNN
    newTest["Close"] = newTest['Close'].shift(1)
    trainingSet = newTest.iloc[:,0:numberOfFeatures].values #convert to numpy to train RNN
    realStockPrice = realStockPrice[window+leftshift:]

    # %% [markdown]
    # ## Predict price

    # %%

    inputs = trainingSet
    inputs = sc.transform(inputs)

    # %%
    xTest = []
    for i in range(window+leftshift,len(inputs)):
        xTest.append(inputs[i-window:i])
    xTest = np.array(xTest)
    xTest = np.reshape(xTest, (xTest.shape[0],xTest.shape[1],numberOfFeatures)) #batchsize, inputSize, numberOfFeatures

    # %%
    predictedPrice = regressor.predict(xTest)
    predictedPrice = ySC.inverse_transform(predictedPrice)

    
    # %% [markdown]
    # # Visualsing the data

    # %%
    plt.plot(realStockPrice, color = 'red', label = "Real Stock Price")
    plt.plot(predictedPrice, color = 'blue', label = "Predicted Stock Price")
    plt.title("{} Stock Price".format(stock))
    plt.xlabel('Time')
    plt.ylabel("Price")
    plt.legend()
    plt.savefig('Graphs-Testing/{}-{}-{}.png'.format(stock,layer1,layer2))
    plt.figure()

    MSE = mean_squared_error(realStockPrice,predictedPrice)

    stocksOwned = {}
    liquidValue = 500000/505
    startingValue = liquidValue
    sold = 0

    for i in range(len(predictedPrice)):
        if realStockPrice[i] < predictedPrice[i] and stock not in stocksOwned:
            stocksOwned[stock] = (
                realStockPrice[i], liquidValue/realStockPrice[i])
            liquidValue -= liquidValue / \
                realStockPrice[i] * realStockPrice[i]
        elif stock in stocksOwned and stocksOwned[stock][0] < realStockPrice[i]:
            liquidValue += stocksOwned[stock][1] * realStockPrice[i]
            percentGain = (
                realStockPrice[i] - stocksOwned[stock][0]) / stocksOwned[stock][0]
            sold += 1

            stocksOwned.pop(stock)
        if stock in stocksOwned and i == len(predictedPrice) - 1:
            liquidValue += stocksOwned[stock][1] * realStockPrice[i]
            percentGain = (
                realStockPrice[i] - stocksOwned[stock][0]) / stocksOwned[stock][0]
            stocksOwned.pop(stock)
            sold += 1
    profit = liquidValue - startingValue
    return profit,mse




companies = []
stocks = {}
profit = 0
data_path = "/Users/ishan/Coding/Wpi/StockMarketSimulationIQP/Datasets/30y_stock_csvs/"
for (dirpath, dirnames, filenames) in walk(data_path):
    companies.extend(filenames)
    break 
companyLayers = {}
for company in tqdm(companies):
    company = company[:-4]
    if os.path.exists("Best/Graphs/{}.png".format(company)):
        continue
    print(padding)
    print(company)
    mse = float('inf')
    layers = ()
    for layer1 in range(1, 21): 
        _, tempMse = rnn(company, layer1 = layer1, layer2 = 1)
        if tempMse == 0:
            continue
        if tempMse < mse:
            layers = (layer1, 1)
            mse = tempMse

    if not layers:
        continue
    layer1 = layers[0]
    for layer2 in range(1, 21):
        _, tempMse = rnn(company, layer1=layer1, layer2=layer2)
        if tempMse == 0:
            continue
        if tempMse < mse:
            layers = (layer1, layer2)
            mse = tempMse
        print(tempMse)
    layer1, layer2 = layers
    if mse == float('inf'):
        continue

    os.rename("Models-Testing/{}-{}-{}-model.json".format(company, layer1, layer2), "Best/Models/{}-model.json".format(company))
    os.rename('Graphs-Testing/{}-{}-{}.png'.format(company, layer1, layer2),"Best/Graphs/{}.png".format(company))
    dir = "/Users/ishan/Coding/Wpi/StockMarketSimulationIQP/IshanCode/Models-Testing"
    
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    dir = "/Users/ishan/Coding/Wpi/StockMarketSimulationIQP/IshanCode/Graphs-Testing"
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

# %%

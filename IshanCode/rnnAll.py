from datetime import datetime, timedelta
import os
from os import walk
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.preprocessing import MinMaxScaler
import shutil
import os
from tqdm import tqdm
from helperFunctions import *

# ## Import training set


def rnn(stock, start_date="1982-3-12", end_date="2022-02-1", lr=0.01, layer1=50, layer2=50, layer3=50, layer4=50):  # %%
    start_date = datetime.strptime(str(start_date), "%y-%m-%d")
    end_date = datetime.strptime(str(end_date), "%y-%m-%d")
    

    variables_to_include = ['close','volume',"rsi","adx","fastd","fastk","macd"]
    
   

    real_stock_price, data, new_train, number_of_features = get_data(
        stock, start_date, end_date, variables_to_include)

    y_set = data.astype(float).values.reshape(-1, 1)

    # ## feature scaling
    # use normalization x - min(x) / max(min) - min(x)
    sc = MinMaxScaler(feature_range=(0,1)) # all values between 0 and 1
    y_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaled = y_scaler.fit_transform(y_set)
    
    x_train, y_train = convert_to_numpy(
        new_train, number_of_features, sc, y_scaled, training=True)

    model = "best/models/{}-model.json".format(stock)
    if os.path.exists(model):
        regressor=tf.keras.models.load_model(model)
        print("loaded")
   
    else:
        build_model(stock,x_train,y_train,True,layer1,layer2)
    # ## part 3 - predictions and visualing the results


    real_stock_price, data, new_test, number_of_features = get_data(stock,None,None,variables_to_include,True)

    
    # ## predict price

    x_test = convert_to_numpy(new_test, number_of_features, sc)

    predicted_price, real_stock_price = get_predictions(
        regressor, x_test, real_stock_price, y_scaler, shift = 0)

    # # visualsing the data

   
    plot_data(stock,real_stock_price,predicted_price,None,None,None,True,layer1,layer2)

    mse = mean_squared_error(real_stock_price,predicted_price)
    liquid_value = 500000/505
    starting_value = liquid_value
    profit, _, _, _, _, _ = calculate_profit(starting_value,predicted_price,real_stock_price)
    
    return profit,mse




companies = []
stocks = {}
profit = 0
data_path = "/users/ishan/coding/wpi/stock_market_simulation_iqp/datasets/30y_stock_csvs/"
for (dirpath, dirnames, filenames) in walk(data_path):
    companies.extend(filenames)
    break 
company_layers = {}
for company in tqdm(companies):
    company = company[:-4]

    if os.path.exists("best/graphs/{}.png".format(company)):
        continue

    print(padding)
    print(company)

    mse = float('inf')
    layers = ()
    for layer1 in range(1, 21): 
        _, temp_mse = rnn(company, layer1 = layer1, layer2 = 1)
        if temp_mse == 0:
            continue
        if temp_mse < mse:
            layers = (layer1, 1)
            mse = temp_mse

    if not layers:
        continue
    layer1 = layers[0]
    for layer2 in range(1, 21):
        _, temp_mse = rnn(company, layer1=layer1, layer2=layer2)
        if temp_mse == 0:
            continue
        if temp_mse < mse:
            layers = (layer1, layer2)
            mse = temp_mse
    layer1, layer2 = layers
    if mse == float('inf'):
        continue

    os.rename("models-testing/{}-{}-{}-model.json".format(company, layer1, layer2), "best/models/{}-model.json".format(company))
    os.rename('graphs-testing/{}-{}-{}.png'.format(company, layer1, layer2),"best/graphs/{}.png".format(company))
    dir = "/users/ishan/coding/wpi/stock_market_simulation_iqp/ishan_code/models-testing"

    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    dir = "/users/ishan/coding/wpi/stock_market_simulation_iqp/ishan_code/graphs-testing"
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

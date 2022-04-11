# By: Brianna Roskind

import tensorflow
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import numpy
from numpy import genfromtxt
import argparse
from tensorflow.keras.optimizers import SGD
from config import *
import seaborn
import matplotlib.pyplot as plot
import pandas

training_accuracy_file = "training_accuracy_1.csv"
testing_accuracy_file = "testing_accuracy_1.csv"
batch_size = 128
sampling_rate = 1
parameters_per_day = 7
sequence_stride = 1
training_start_index = 0
number_of_testing_weeks = 8#16
number_of_testing_days = 5 * number_of_testing_weeks #+ 1
trading_days_per_year = 252

def split_training_and_testing_data():
    print("number_of_testing_days = ", number_of_testing_days)
    global num_stocks
    global one_day_data_size
    print("Retrieving dataset from mother file")
    full_dataset = genfromtxt(mother_file, delimiter=',')
    print("Full dataset length = " + str(len(full_dataset)))
    one_day_data_size = len(full_dataset[0])
    print("one_day_data_size = " + str(one_day_data_size))
    num_stocks = one_day_data_size//parameters_per_day
    print("num_stocks = " + str(num_stocks))
    labels = genfromtxt(labels_file, delimiter=',').flatten()
    labels_one_hot = tensorflow.keras.utils.to_categorical(labels, num_classes=num_stocks)
    print("About to assemble data set generators")
    testing_end_index = len(full_dataset) - 1
    training_end_index = testing_end_index - number_of_testing_days
    testing_start_index = len(full_dataset) - number_of_testing_days - days_to_train_on
    print("training_start_index = " + str(training_start_index))
    print("training_end_index = " + str(training_end_index))
    print("testing_start_index = " + str(testing_start_index))
    print("testing_end_index = " + str(testing_end_index))
    training_generator = timeseries_dataset_from_array(data=full_dataset,
                                                       targets=labels_one_hot,
                                                       sequence_length=days_to_train_on,
                                                       sequence_stride=sequence_stride,
                                                       sampling_rate=sampling_rate, # Default is 1
                                                       batch_size=batch_size, # Default is 128
                                                       end_index=training_end_index,
                                                       start_index=training_start_index)
    print("Finish building training dataset generator")
    testing_generator = timeseries_dataset_from_array(data=full_dataset,
                                                      targets=labels_one_hot,
                                                      sequence_length=days_to_train_on,
                                                      sequence_stride=sequence_stride,
                                                      sampling_rate=sampling_rate, # Default is 1
                                                      batch_size=batch_size, # Default is 128
                                                      end_index=testing_end_index,
                                                      start_index=testing_start_index)
    labels_one_hot += labels_one_hot[0]
    profit_testing_generator = timeseries_dataset_from_array(data=full_dataset,
                                                      targets=labels_one_hot,
                                                      sequence_length=days_to_train_on,
                                                      sequence_stride=sequence_stride,
                                                      sampling_rate=sampling_rate, # Default is 1
                                                      batch_size=1, # Default is 128
                                                      #end_index=testing_end_index + 1,
                                                      start_index=testing_start_index)
    print("Finish building testing dataset generator")
    print("training_generator length = " + str(len(training_generator)))
    print("testing_generator length = " + str(len(testing_generator)))
    #print("Here is the testing_generator")
    #for row in testing_generator:
    #    print(row)
    #print("Here is the training_generator")
    #for row in training_generator:
    #    print(row)
    return training_generator, testing_generator, profit_testing_generator

def make_new_nn_model_1_hidden_layer():
    print("About to build model")
    nn_model = tensorflow.keras.models.Sequential()
    nn_model.add(tensorflow.keras.layers.Flatten())
    #nn_model.add(tensorflow.keras.Input(shape=(sequence_length, one_day_data_size)))
    nn_model.add(tensorflow.keras.layers.Dense(8000, activation='relu'))
    nn_model.add(tensorflow.keras.layers.Dense(num_stocks, activation='softmax'))
    print("Finished building model")
    print("About to compile")
    nn_model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'], run_eagerly=True)
    print("Finished compiling")
    return nn_model

def make_new_cnn_model(convolution_parameters_per_day, convolution_parameters_per_30_days, learning_rate):
    print("About to build model")
    nn_model = tensorflow.keras.models.Sequential()
    nn_model.add(tensorflow.keras.Input(shape=(days_to_train_on, one_day_data_size, 1)))
    nn_model.add(tensorflow.keras.layers.Conv2D(filters=convolution_parameters_per_day, kernel_size=(1, parameters_per_day), strides=(1, parameters_per_day), activation='relu'))
    nn_model.add(tensorflow.keras.layers.Reshape((days_to_train_on, convolution_parameters_per_day * num_stocks, 1)))
    nn_model.add(tensorflow.keras.layers.Conv2D(filters=convolution_parameters_per_30_days, kernel_size=(days_to_train_on, convolution_parameters_per_day), strides=(1, convolution_parameters_per_day), activation='relu'))
    nn_model.add(tensorflow.keras.layers.Flatten())
    nn_model.add(tensorflow.keras.layers.Dense(num_stocks, activation='softmax'))
    print("Finished building model")
    print(nn_model.summary())
    print("About to compile")
    print("learning_rate = ", str(learning_rate))
    optimizer = SGD(learning_rate=learning_rate)
    nn_model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'], run_eagerly=True)
    print("Finished compiling")
    return nn_model

def train_and_predict(epochs, nn_model, training_generator, testing_generator, profit_testing_generator):
    with open(training_accuracy_file, 'w') as training_handle:
        with open(testing_accuracy_file, 'w') as testing_handle:
            for i in range (1, epochs + 1):
                nn_model.fit(training_generator, epochs=1, verbose=0)
                _, training_accuracy = nn_model.evaluate(training_generator, verbose=0)
                training_handle.write(str(training_accuracy * 100.0) + ",")
                training_handle.flush()
                _, testing_accuracy = nn_model.evaluate(testing_generator, verbose=0)
                testing_handle.write(str(testing_accuracy * 100.0) + ",")
                testing_handle.flush()
                print(".", end="", flush=True)
    _, testing_accuracy = nn_model.evaluate(testing_generator, verbose=1)
    print('For Epochs = {}: Accuracy = {}'.format(epochs, testing_accuracy * 100.0))
    starting_money = 500_000#100.0
    money = starting_money
    profit_testing_iterator = iter(profit_testing_generator)
    next_day_set = profit_testing_iterator.get_next()
    money_per_day_array = []
    money_per_day_array.append(money)
    percent_profit_per_day_array = []
    list_of_symbols = pandas.read_csv('500_Stocks.csv')["Symbol"].to_numpy()
    stock_bought_each_day_array = []
    for index in range(1, len(profit_testing_generator)-1):
        previous_days_set = next_day_set
        next_day_set = profit_testing_iterator.get_next()
        
        one_hot_predictions_array = nn_model.predict(previous_days_set[0], verbose=0)[0]
        #print("One hot predictions array:")
        #print(one_hot_predictions_array)
        #print(len(one_hot_predictions_array))
        best_stock_index = numpy.argmax(one_hot_predictions_array, axis=0)
        print("Best stock index: " + str(best_stock_index))

        next_day_data = next_day_set[0].numpy()[0][0]
        #print(next_day_data)
        #print(next_day_data[0])
        #print(len(next_day_data))
        #print(best_stock_index * parameters_per_day)
        open_price = next_day_data[best_stock_index * parameters_per_day]
        close_price = next_day_data[(best_stock_index * parameters_per_day) + 1]
        money = (money / open_price) * close_price
        if 6 == index: # SPECIFICALLY FOR THE 2/14 - 3/8 TESTING RANGE
            # Add in an empty entry for president's day (stock market was closed).
            money_per_day_array.append(money_per_day_array[-1])
            percent_profit_per_day_array.append(0)
            stock_bought_each_day_array.append("NO STOCK TRADING")            
        money_per_day_array.append(money)
        percent_profit_per_day_array.append(100 * ((close_price / open_price) - 1))
        stock_bought_each_day_array.append(list_of_symbols[best_stock_index])
    print("Starting Money: $" + str(starting_money))
    print("Ending Money: $" + str(money))
    print("Percent Profit: " + str(100 * ((money - starting_money) / starting_money)) + "%")
    graph_money_per_day(money_per_day_array)
    #graph_bar_chart_percent_profit_per_day(percent_profit_per_day_array, stock_bought_each_day_array)
    #graph_percent_profit_per_day(percent_profit_per_day_array, stock_bought_each_day_array)

    sharpe_ratio = ((trading_days_per_year ** 0.5)
                    * numpy.mean(percent_profit_per_day_array) / numpy.std(percent_profit_per_day_array))
    print("Sharpe Ratio: " + str(sharpe_ratio))
    numpy_percent_profit_per_day_array = numpy.array(percent_profit_per_day_array)
    std_dev_of_negative_performers = numpy_percent_profit_per_day_array[numpy_percent_profit_per_day_array < 0]
    if 0 == len(std_dev_of_negative_performers):
        print("The Sortino Ratio is not applicable since there were no negative performance days.")
    else:
        sortino_ratio = ((trading_days_per_year ** 0.5)
                         * numpy.mean(percent_profit_per_day_array) / numpy.std(std_dev_of_negative_performers))
        print("Sortino Ratio: " + str(sortino_ratio))

def graph_money_per_day(money_per_day_array):
    money_per_day_line_graph(money_per_day_array)
    original_index_array = [1, 2, 3, 4, 5]
    index_array = original_index_array
    for i in range (1, number_of_testing_weeks + 1):
        for j in range(1, len(index_array) + 1):
            index_array[j - 1]= j + 5 * (i - 1)
        array_portion = money_per_day_array[1:][5 * (i - 1):5 * i]
        money_per_day_line_graph(array_portion, "Week " + str(i) + " ", index_array)

def money_per_day_line_graph(money_per_day_array, title_prefix="", index_array=None):
    if None == index_array:
        seaborn.lineplot(data=money_per_day_array, marker="o")
    else:
        seaborn.lineplot(x=index_array, y=money_per_day_array, marker="o")
        plot.xticks(index_array)
    plot.suptitle(title_prefix + "Dollars Per Day", size = 24);
    plot.ylabel("Dollars", size = 24)
    plot.xlabel("Days", size = 24)
    plot.show()

def graph_bar_chart_percent_profit_per_day(percent_profit_per_day_array, stock_bought_each_day_array):
    percent_profit_stock_per_day_dataframe = pandas.DataFrame({
        "percent_profit_per_day":percent_profit_per_day_array,
        "stock_bought_each_day":stock_bought_each_day_array
    })
    percent_profit_stock_per_day_dataframe.index += 1
    percent_profit_per_day_bar_chart(percent_profit_stock_per_day_dataframe)
    for i in range (1, number_of_testing_weeks + 1):
        dataframe_portion = percent_profit_stock_per_day_dataframe[5 * (i - 1):5 * i]
        print(dataframe_portion.to_string())
        percent_profit_per_day_bar_chart(dataframe_portion, "Week " + str(i) + " ")

def percent_profit_per_day_bar_chart(percent_profit_stock_per_day_dataframe, title_prefix=""):
    seaborn.barplot(
        data=percent_profit_stock_per_day_dataframe,
        x=percent_profit_stock_per_day_dataframe.index,
        y="percent_profit_per_day",
        hue="stock_bought_each_day",
        dodge=False
    )
    plot.suptitle(title_prefix + "Percent Profit Per Day", size = 24);
    plot.ylabel("Percent Profit", size = 24)
    plot.xlabel("Days", size = 24)
    plot.legend(fontsize=17)
    plot.show()

def graph_percent_profit_per_day(percent_profit_per_day_array, stock_bought_each_day_array):
    percent_profit_stock_per_day_dataframe = pandas.DataFrame({
        "percent_profit_per_day":percent_profit_per_day_array,
        "stock_bought_each_day":stock_bought_each_day_array
    })
    percent_profit_stock_per_day_dataframe.index += 1
    seaborn.lineplot(
        data=percent_profit_stock_per_day_dataframe,
        x=percent_profit_stock_per_day_dataframe.index,
        y="percent_profit_per_day",
    )
    seaborn.scatterplot(
        data=percent_profit_stock_per_day_dataframe,
        x=percent_profit_stock_per_day_dataframe.index,
        y="percent_profit_per_day",
        hue="stock_bought_each_day",
        marker="o"
    )
    plot.suptitle("Percent Profit Per Day", size = 24);
    plot.ylabel("Percent Profit", size = 24)
    plot.xlabel("Days", size = 24)
    plot.xlim(0, number_of_testing_days)
    plot.legend(fontsize=17)
    plot.show()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 0)
    parser.add_argument('--model_file', default = None)
    parser.add_argument('--save_to_file', default = None)
    parser.add_argument('--convolution_parameters_per_day', default = 5)
    parser.add_argument('--convolution_parameters_per_30_days', default = 8)
    parser.add_argument('--learning_rate', default = 0.02)
    parser.add_argument('--accuracy_files_suffix', default = None)
    args = parser.parse_args()
    epochs = int(args.epochs)
    model_file = args.model_file
    save_to_file = args.save_to_file
    accuracy_files_suffix = args.accuracy_files_suffix
    if None != accuracy_files_suffix:
        global training_accuracy_file
        global testing_accuracy_file
        training_accuracy_file = "training_accuracy_" + accuracy_files_suffix + ".csv"
        testing_accuracy_file = "testing_accuracy_" + accuracy_files_suffix + ".csv"
    training_generator, testing_generator, profit_testing_generator = split_training_and_testing_data()
    if None == model_file:
        print("Making new model")
        convolution_parameters_per_day = int(args.convolution_parameters_per_day)
        convolution_parameters_per_30_days = int(args.convolution_parameters_per_30_days)
        learning_rate = float(args.learning_rate)
        nn_model = make_new_cnn_model(convolution_parameters_per_day, convolution_parameters_per_30_days, learning_rate)
    else:
        print("Making model from file: " + model_file)
        nn_model = tensorflow.keras.models.load_model(model_file)
    train_and_predict(epochs, nn_model, training_generator, testing_generator, profit_testing_generator)
    if None != save_to_file:
        print("Saving model to file: " + save_to_file)
        nn_model.save(save_to_file)

main()

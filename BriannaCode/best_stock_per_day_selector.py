# By: Brianna Roskind

import tensorflow
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from numpy import genfromtxt
import argparse
from config import *

training_1_hidden_layer_accuracy_file = "training_2_stocks_1_hidden_layer_accuracy.csv"
testing_1_hidden_layer_accuracy_file = "testing_2_stocks_1_hidden_layer_accuracy.csv"
batch_size = 128
parameters_per_day = 5
sequence_stride = 2 # Use every other dataset(Shift 2 days over)
training_start_index = 0
testing_start_index = 1 # Start on alternate than training.

def split_training_and_testing_data():
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
    end_index = len(full_dataset) - 1
    training_generator = timeseries_dataset_from_array(data=full_dataset,
                                                       targets=labels_one_hot,
                                                       sequence_length=days_to_train_on,
                                                       sequence_stride=sequence_stride,
                                                       #sampling_rate=1, # Default is already 1
                                                       batch_size=batch_size, # Default is 128
                                                       end_index=end_index,
                                                       start_index=training_start_index)
    print("Finish building training dataset generator")
    testing_generator = timeseries_dataset_from_array(data=full_dataset,
                                                      targets=labels_one_hot,
                                                      sequence_length=days_to_train_on,
                                                      sequence_stride=sequence_stride,
                                                      #sampling_rate=1, # Default is already 1
                                                      batch_size=batch_size, # Default is 128
                                                      end_index=end_index,
                                                      start_index=testing_start_index)
    print("Finish building testing dataset generator")
    print("training_generator length = " + str(len(training_generator)))
    print("testing_generator length = " + str(len(testing_generator)))
    return training_generator, testing_generator

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

#2 hidden layer idea:
#75750 #Original dimensions
#7600 #Dense
#800 #Dense
#505 #Softmax 1 hot

def train_and_predict(epochs, nn_model, training_generator, testing_generator):
    with open(training_1_hidden_layer_accuracy_file, 'w') as training_1_hidden_layer_handle:
        with open(testing_1_hidden_layer_accuracy_file, 'w') as testing_1_hidden_layer_handle:
            for i in range (1, epochs + 1):
                nn_model.fit(training_generator, epochs=1, verbose=0)
                _, training_accuracy = nn_model.evaluate(training_generator, verbose=0)
                training_1_hidden_layer_handle.write(str(training_accuracy) + ",")
                training_1_hidden_layer_handle.flush()
                _, testing_accuracy = nn_model.evaluate(testing_generator, verbose=0)
                testing_1_hidden_layer_handle.write(str(testing_accuracy) + ",")
                testing_1_hidden_layer_handle.flush()
                print(".", end="", flush=True)
    _, testing_accuracy = nn_model.evaluate(testing_generator, verbose=1)
    print('For Epochs = {}: Accuracy = {}'.format(epochs, testing_accuracy * 100.0))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default = 100)
    parser.add_argument('--model_file', default = None)
    parser.add_argument('--save_to_file', default = None)
    args = parser.parse_args()
    epochs = int(args.epochs)
    model_file = args.model_file
    save_to_file = args.save_to_file
    training_generator, testing_generator = split_training_and_testing_data()
    if None == model_file:
        print("Making new model")
        nn_model = make_new_nn_model_1_hidden_layer()
    else:
        print("Making model from file: " + model_file)
        nn_model = tensorflow.keras.models.load_model(model_file)
    train_and_predict(epochs, nn_model, training_generator, testing_generator)
    if None != save_to_file:
        print("Saving model to file: " + save_to_file)
        nn_model.save(save_to_file)

main()

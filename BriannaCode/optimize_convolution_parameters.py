import os
import pandas
import seaborn
import matplotlib.pyplot as plot
from numpy import genfromtxt

range_start = 1#5 # Use 1 for the first conv. layer, and 5 for the second conv. layer.
range_end_plus_one = 7#16 # Use 7 for the first conv. layer, and 5 for the second conv. layer.
learning_rates_array = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]

def optimize_parameter(parameter_name):
    command_base = "python3 best_stock_per_day_selector.py --epochs=20 --" + parameter_name  + "="
    command_save_to_file_addon = " --save_to_file=model_learning_rate_0_02_" + parameter_name  + "_"
    command_accuracy_files_suffix_addon = " --accuracy_files_suffix=learning_rate_0_02_" + parameter_name  + "_"
    for i in range(range_start, range_end_plus_one):
        os.system(command_base + str(i) +
                  command_save_to_file_addon + str(i) +
                  command_accuracy_files_suffix_addon + str(i))

def plot_per_epoch(parameter_name):
    training_accuracy_dataframe = pandas.DataFrame()
    testing_accuracy_dataframe = pandas.DataFrame()
    for i in range(range_start, range_end_plus_one):
        training_accuracy_array = genfromtxt("training_accuracy_learning_rate_0_02_" + parameter_name  + "_" + str(i) + ".csv", delimiter=',')[:-1] # There is an extra comma at the end of the file.
        testing_accuracy_array = genfromtxt("testing_accuracy_learning_rate_0_02_" + parameter_name  + "_" + str(i) + ".csv", delimiter=',')[:-1] # There is an extra comma at the end of the file.
        training_accuracy_dataframe[parameter_name  + " = " + str(i)] = training_accuracy_array
        testing_accuracy_dataframe[parameter_name  + " = " + str(i)] = testing_accuracy_array
    # Change dataframes to have 1-based indexing
    training_accuracy_dataframe.index += 1
    testing_accuracy_dataframe.index += 1

    make_epoch_accuracy_plot(training_accuracy_dataframe, "Training", parameter_name)
    make_epoch_accuracy_plot(testing_accuracy_dataframe, "Testing", parameter_name)

def optimize_learning_rate():
    command_base = "python3 best_stock_per_day_selector.py --epochs=20 --learning_rate="
    command_save_to_file_addon = " --save_to_file=model_2_learning_rate_"
    command_accuracy_files_suffix_addon = " --accuracy_files_suffix=2_learning_rate_"
    for learning_rate in learning_rates_array:
        os.system(command_base + str(learning_rate) +
                  command_save_to_file_addon + (str(learning_rate)).replace(".", "_") +
                  command_accuracy_files_suffix_addon + (str(learning_rate)).replace(".", "_"))

def plot_file_suffixes(parameter_name, file_suffixes):
    training_accuracy_dataframe = pandas.DataFrame()
    testing_accuracy_dataframe = pandas.DataFrame()
    for i in range(0, len(file_suffixes)):
        training_accuracy_array = genfromtxt("training_accuracy_2_" + file_suffixes[i] + ".csv", delimiter=',')[:-1] # There is an extra comma at the end of the file.
        testing_accuracy_array = genfromtxt("testing_accuracy_2_" + file_suffixes[i] + ".csv", delimiter=',')[:-1] # There is an extra comma at the end of the file.
        training_accuracy_dataframe[file_suffixes[i]] = training_accuracy_array
        testing_accuracy_dataframe[file_suffixes[i]] = testing_accuracy_array
    # Change dataframes to have 1-based indexing
    training_accuracy_dataframe.index += 1
    testing_accuracy_dataframe.index += 1

    make_epoch_accuracy_plot(training_accuracy_dataframe, "Training", parameter_name)
    make_epoch_accuracy_plot(testing_accuracy_dataframe, "Testing", parameter_name)

def make_epoch_accuracy_plot(dataframe, data_name, parameter_name):
    seaborn.lineplot(data=dataframe,marker="o")
    plot.suptitle("Accuracy Per Epoch of " + data_name + " for each " + parameter_name + " variable", size = 24);
    plot.ylabel("Accuracy", size = 24)
    plot.xlabel("Epochs of training", size = 24)
    plot.xticks(ticks=dataframe.index)
    plot.legend(fontsize=17)
    plot.show()
    

#optimize_parameter("convolution_parameters_per_30_days")
#plot_per_epoch("convolution_parameters_per_30_days")
#optimize_parameter("convolution_parameters_per_day")
#plot_per_epoch("convolution_parameters_per_day")
optimize_learning_rate()
#plot_file_suffixes("learning rate", ["learning_rate_0_01", "learning_rate_0_02", "learning_rate_0_03", "learning_rate_0_04", "learning_rate_0_05", "learning_rate_0_06", "learning_rate_0_07", "learning_rate_0_08", "learning_rate_0_09", "learning_rate_0_1", "learning_rate_0_2"])

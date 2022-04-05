# By Brianna Roskind

import pandas as pd
import csv
from config import *

human_readable_labels_file = "human_readable_" + labels_file

class ProblematicStockDataInfo:
    # The only known problem thus far is that the opening price is zero.
    
    def __init__(self, file_name, problematic_date):
        self.file_name = file_name
        self.first_problematic_date = problematic_date
        self.last_problematic_date = problematic_date
        self.problematic_count = 1

    def print_problematic_info(self):
        print("\n" + self.file_name + " had problematic data in " + str(self.problematic_count) + " entries.")
        print("\t The first occured at: " + self.first_problematic_date)
        print("\t The last occured at: " + self.last_problematic_date)

    def add_problematic_date(self, problematic_date):
        self.last_problematic_date = problematic_date
        self.problematic_count += 1
        
def fill_labels_file():
    stocks = pd.read_csv(stock_symbols_csv_file)
    clean_csv_handle_array = []
    clean_csv_reader_array = []
    for symbol in stocks.Symbol:
        path_to_clean_csv_file = clean_directory + "/" + symbol + ".csv"
        clean_csv_handle = open(path_to_clean_csv_file, 'r')
        clean_csv_reader = csv.reader(clean_csv_handle, delimiter=',')
        try:
            headers = next(clean_csv_reader)
            for i in range(0, days_to_train_on):
                next(clean_csv_reader)
        except csv.Error:
            print("CSV Error in " + csv_file)
            continue
        except StopIteration:
            print("Empty file not used: " + symbol)
            continue
        if (headers[0] != "Datetime") or (headers[1] != "OPEN") or (headers[-2] != "CLOSE"):
            print(csv_file + " does not have OPEN and CLOSE headings in the expected placement.")
            continue
        clean_csv_handle_array.append(clean_csv_handle)
        clean_csv_reader_array.append(clean_csv_reader)
    with open(labels_file, 'w') as labels_file_handle:
        with open(human_readable_labels_file, 'w') as human_readable_labels_file_handle:
            human_readable_labels_file_handle.write("Datetime,Index,Symbol,Percent Gain\n")
            done_writing_labels_file = False
            rows_read = 0
            problematic_dictionary = {}
            while not done_writing_labels_file:
                highest_percentage_increase = None
                best_stock_code = None
                best_stock_symbol = None
                handle_index = 0
                current_date = None
                for clean_csv_reader in clean_csv_reader_array:
                    clean_csv_file_name = clean_csv_handle_array[handle_index].name
                    handle_index += 1
                    try:
                        current_row = next(clean_csv_reader) # Read the header line of the csv to set up to read data.
                    except csv.Error:
                        print("Quit due to: CSV Error in " + clean_csv_file_name)
                        done_writing_labels_file = True # Quit due to error.
                        break
                    except StopIteration:
                        done_writing_labels_file = True # Finished reading file.
                        break
                    if None == current_date:
                        current_date = current_row[0]
                    open_price = float(current_row[1])
                    if (0 == open_price):
                        problematic_stock_data_info = problematic_dictionary.get(clean_csv_file_name)
                        if problematic_stock_data_info:
                            problematic_stock_data_info.add_problematic_date(current_date)
                        else:
                            problematic_stock_data_info = ProblematicStockDataInfo(clean_csv_file_name, current_date)
                            problematic_dictionary[clean_csv_file_name] = problematic_stock_data_info
                        continue # We do not consider problematic stocks to be evaluated as the best stock.
                    if 0 == float(current_row[-1]): # Check if the VOLUME is 0.
                        # DO NOT INCLUDE STOCKS THAT WERE NOT TRADING / WERE NOT PUBLIC YET THAT DAY.
                        continue
                    close_price = float(current_row[-2])
                    current_percentage_increase = 100 * ((close_price - open_price) / open_price)
                    if (None == highest_percentage_increase) or (current_percentage_increase > highest_percentage_increase):
                        highest_percentage_increase = current_percentage_increase
                        best_stock_code = str(handle_index-1) # We already incremented the handle index.
                        best_stock_symbol = (((clean_csv_file_name.split("/"))[-1]).split(".csv"))[0]
                if not done_writing_labels_file:
                    labels_file_handle.write(best_stock_code + "\n")
                    human_readable_labels_file_handle.write(current_date +
                                                           "," + best_stock_code +
                                                           "," + best_stock_symbol +
                                                           "," + str(highest_percentage_increase) +
                                                           "\n")
                    print(".", end="", flush=True)
                    rows_read += 1
                    labels_file_handle.flush()
    for clean_csv_handle in clean_csv_handle_array:
        assert not clean_csv_handle.readline()
        clean_csv_handle.close()
    for _, problematic_stock_data_info in problematic_dictionary.items():
        problematic_stock_data_info.print_problematic_info()
    print("Rows read for labels file: " + str(rows_read) + " (The first " + str(days_to_train_on) + " days do not get labels)")

def fill_mother_file():
    stocks = pd.read_csv(stock_symbols_csv_file)
    CNN_formatted_csv_handle_array = []
    for symbol in stocks.Symbol:
        path_to_CNN_formatted_csv_file = CNN_formatted_directory + "/" + symbol + ".csv"
        CNN_formatted_csv_handle = open(path_to_CNN_formatted_csv_file, 'r')
        current_row = CNN_formatted_csv_handle.readline()
        if not current_row:
            print("Empty file not used: " + symbol)
            continue
        CNN_formatted_csv_handle_array.append(CNN_formatted_csv_handle)
    with open(mother_file, 'w') as mother_file_handle:
        done_writing_mother_file = False
        first_comma_index = None
        rows_read = 0
        while not done_writing_mother_file:
            first_file_read = False
            for CNN_formatted_csv_handle in CNN_formatted_csv_handle_array:
                current_row = CNN_formatted_csv_handle.readline()
                if not current_row:
                    done_writing_mother_file = True
                    break # Finished reading file.
                if None == first_comma_index:
                    first_comma_index = current_row.find(",")
                if first_file_read:
                    current_row_data = current_row[first_comma_index:-1]
                else:
                    current_row_data = current_row[(first_comma_index + 1):-1]
                    first_file_read = True
                mother_file_handle.write(current_row_data)
            if not done_writing_mother_file:
                mother_file_handle.write("\n")
                print(".", end="", flush=True)
                rows_read += 1
            mother_file_handle.flush()
    for CNN_formatted_csv_handle in CNN_formatted_csv_handle_array:
        assert not CNN_formatted_csv_handle.readline()
        CNN_formatted_csv_handle.close()
    print("\nRows read for mother file: " + str(rows_read))
        
def main():
    print("Filling in the mother file")
    fill_mother_file()
    #print("Filling in the labels file")
    #fill_labels_file()

main()

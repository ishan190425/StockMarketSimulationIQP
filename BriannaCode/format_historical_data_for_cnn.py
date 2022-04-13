# By: Brianna Roskind

import pandas as pd
from math import log
from datetime import datetime
import csv
from config import *

date_format_string = "%Y-%m-%d %H:%M:%S"
extra_parameters = ["Month", "DayOfWeek"]

def get_month_and_day_of_week(date):
    datetime_date = datetime.strptime(date, date_format_string)
    day_of_week = datetime_date.weekday()
    month = datetime_date.month
    return [month, day_of_week]

def format_for_CNN():
    stocks = pd.read_csv(stock_symbols_csv_file)
    for symbol in stocks.Symbol:
        print(".", end="", flush=True)
        symbol_csv_base = "/" + symbol + ".csv"
        path_to_clean_csv_file = clean_directory + symbol_csv_base
        with open(path_to_clean_csv_file, 'r') as clean_csv_handle:
            clean_csv_reader = csv.reader(clean_csv_handle, delimiter=',')
            headers = next(clean_csv_reader)
            if (headers[0] != "Datetime") or (headers[-1] != "VOLUME"):
                print("Error in file format " + csv_file)
                continue
            path_to_CNN_formatted_csv_file = CNN_formatted_directory + symbol_csv_base
            #reformat headers:
            reformatted_headers = [None] * (len(headers) + len(extra_parameters))
            for index in range(0, len(headers)):
                if headers[index] != "Datetime":
                    reformatted_headers[index] = "log(" + headers[index] + ")"
                else:
                    reformatted_headers[index] = headers[index]
            for negative_index in range(-1, -(len(extra_parameters) + 1), -1):
                reformatted_headers[negative_index] = extra_parameters[negative_index]
            with open(path_to_CNN_formatted_csv_file, 'w') as CNN_formatted_csv_handle:
                CNN_formatted_csv_writer = csv.writer(CNN_formatted_csv_handle, delimiter=',')
                CNN_formatted_csv_writer.writerow(reformatted_headers)
                while True:
                    try:
                        current_row = next(clean_csv_reader)
                    except csv.Error:
                        print("CSV Error in " + csv_file)
                        continue
                    except StopIteration:
                        break # Finished reading file.
                    new_row = [None] * len(reformatted_headers)
                    new_row[0] = current_row[0]
                    for index in range(1, len(headers)):
                        new_row[index] = log(float(current_row[index]) + 1)
                    month_and_day_of_week = get_month_and_day_of_week(current_row[0])
                    for negative_index in range(-1, -(len(extra_parameters) + 1), -1):
                        new_row[negative_index] = month_and_day_of_week[negative_index]
                    CNN_formatted_csv_writer.writerow(new_row)

def main():
    print("Converting values to logs and adding extra fields")
    format_for_CNN()

main()

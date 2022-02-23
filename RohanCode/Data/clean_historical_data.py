# By: Brianna Roskind

import pandas as pd
from datetime import datetime, timedelta
import csv
from config import *

date_format_string = "%Y-%m-%d %H:%M:%S"
one_day_duration = timedelta(1)

def check_and_adjust_time(date):
    if date[11:] != "20:00:00":
        # All records are recorded at 8 pm except the last one.
        if date[11:] < "20:00:00":
            datetime_date = datetime.strptime(date, date_format_string)
            datetime_date -= one_day_duration
            date = datetime_date.strftime(date_format_string)
        date = date[:11] + "20:00:00"
    return date

def get_list_of_all_dates():
    dates_dict = {}
    stocks = pd.read_csv(stock_symbols_csv_file)
    for symbol in stocks.Symbol:
        print(".", end="", flush=True)
        path_to_raw_csv_file = raw_directory + "/" + symbol + ".csv"
        stock_records = pd.read_csv(path_to_raw_csv_file)
        for date in stock_records.Datetime:
            date = check_and_adjust_time(date)
            date_seen_count = dates_dict.get(date)
            if None == date_seen_count:
                # The date is not in the dictionary.
                dates_dict[date] = 1
            else:
                dates_dict[date] = date_seen_count + 1
    complete_date_list = sorted(dates_dict)
    #print(complete_date_list) # Uncomment this if you would like to see all of the dates in order.
    return complete_date_list

def add_dummy_values_for_missing_dates(complete_date_list):
    stocks = pd.read_csv(stock_symbols_csv_file)
    for symbol in stocks.Symbol:
        print(".", end="", flush=True)
        symbol_csv_base = "/" + symbol + ".csv"
        path_to_raw_csv_file = raw_directory + symbol_csv_base
        with open(path_to_raw_csv_file, 'r') as raw_csv_handle:
            raw_csv_reader = csv.reader(raw_csv_handle, delimiter=',')
            headers = next(raw_csv_reader)
            if (headers[0] != "Datetime") or (headers[-1] != "VOLUME"):
                print("Error in file format " + csv_file)
                continue
            path_to_clean_csv_file = clean_directory + symbol_csv_base
            with open(path_to_clean_csv_file, 'w') as clean_csv_handle:
                clean_csv_writer = csv.writer(clean_csv_handle, delimiter=',')
                clean_csv_writer.writerow(headers)
                required_iterator = iter(complete_date_list)
                while True:
                    try:
                        current_row = next(raw_csv_reader)
                    except csv.Error:
                        print("CSV Error in " + csv_file)
                        continue
                    except StopIteration:
                        break # Finished reading file.
                    current_date = current_row[0]
                    current_date = check_and_adjust_time(current_date)
                    current_row[0] = current_date
                    required_date = next(required_iterator) # There is a current row so there must be a required date.
                    while required_date < current_date:
                        create_and_write_dummy_row(current_row, required_date, clean_csv_writer)
                        required_date = next(required_iterator)
                    # Write the current row as is.
                    clean_csv_writer.writerow(current_row)
                # Finish adding any leftover required dates.
                while True:
                    try:
                        required_date = next(required_iterator)
                    except StopIteration:
                        break # Finished filling out required dates.
                    create_and_write_dummy_row(current_row, required_date, clean_csv_writer)
                        
def create_and_write_dummy_row(current_row, required_date, clean_csv_writer):
    # Create a dummy row for data.
    dummy_row = current_row.copy()
    dummy_row[0] = required_date # Create record with required date and otherwise preserved values.
    dummy_row[-1] = "0" # Set volume to 0
    clean_csv_writer.writerow(dummy_row)
    
def main():
    print("Accumulating complete date list")
    complete_date_list = get_list_of_all_dates()
    print("\nAdding dummy variables for missing dates")
    add_dummy_values_for_missing_dates(complete_date_list)

main()

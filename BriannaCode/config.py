# Replace the following with your own User Agent
User_Agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"

# File containing Name, Symbol data for all stocks you would like to fetch.
stock_symbols_csv_file = "500_Stocks.csv"

# File where raw fetched data for stocks will be stored.
raw_directory = "30y_stock_csvs"

# File where cleaned stocks data will be stored.
clean_directory = "clean_30y_stock_csvs"

# File where CNN formatted data will be stored:
CNN_formatted_directory = "CNN_formatted_stock_csvs"

# Data for training/testing file:
mother_file = "mother_file.csv"

# Labels for training/testing file:
labels_file = "best_stock_per_day_labels.csv"

# Training window size
days_to_train_on = 30

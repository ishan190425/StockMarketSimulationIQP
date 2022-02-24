# Replace the following with your own User Agent
User_Agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15"

# File containing Name, Symbol data for all stocks you would like to fetch.
stock_symbols_csv_file = "500_Stocks.csv"

# File where raw fetched data for stocks will be stored.
raw_directory = "30y_stock_csvs"

# File where cleaned stocks data will be stored.
clean_directory = "clean_30y_stock_csvs"

# Data for training/testing file:
mother_file = "mother_file.csv"

# Labels for training/testing file:
labels_file = "best_stock_per_day_labels.csv"

# Training window size
days_to_train_on = 30

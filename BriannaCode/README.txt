# By: Brianna Roskind
# Below is the order of commands to fetch, clean, format, and train on stock data:



# First, fetch the data from Yahoo finance

python3 fetch_yahoo_historical_data.py

    # If failed_fetch.csv is not empty, you must then use the following command

    python3 fetch_yahoo_hitorical_data.py --retry=True



# Once the failed_fetch.csv file is empty, you may continue, otherwise, repeat the previous step.

python3 clean_historical_data.py



# To create the mother and labels file for training, use the following command

python3 make_mother_and_labels_file.py

	# Note that now the human_readable labels file has been made, and you can check that to confirm
	# that the labels are correct.



# To train a model on your data, use the following command

python3 best_stock_per_day_selector.py

    # Note that the default number of epochs is 100.
    # If you would like to train on a different number of epochs
    # (for example 200), use the following command

    python3 best_stock_per_day_selector.py --epochs=200

    # Note that by default the script will make and train on a new model
    # If you would like to continue to train an existing model
    # (for example my_existing_model.h5), use the following command

    python3 best_stock_per_day_selector.py --model_file=my_existing_model.h5

    # Note that by default the script will not save the model you trained
    # If you would like to save your trained model to a file
    # (for example my_trained_model.h5), use the following command

    python3 best_stock_per_day_selector.py --save_to_file=my_trained_model.h5



import os
from datasets import dataset_handler
from src.prepare_training_set import prepare_and_start_training
from src.utils import export_to_csv


def normalize_data():
    dataset_handler.import_datasets()
    export_to_csv(dataset_handler.csv_df, "datasets/normalized_output.csv")


def __init__():
    # Normalize the data if it is not already normalized.
    if not os.path.exists("datasets/normalized_output.csv"):
        print("Normalized data not found. Normalizing")
        normalize_data()
    #Once it is normalized, prepare the the training data and start the training
    prepare_and_start_training()



if __name__ == "__main__":
    __init__()
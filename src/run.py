

import os
import sys
from datasets import dataset_handler
import datasets.normalize_inference_data as normalize_inference_data
from src.prepare_training_set import prepare_and_start_inference, prepare_and_start_training
from src.utils import export_to_csv


def normalize_data():
    dataset_handler.import_datasets()
    export_to_csv(dataset_handler.csv_df, "datasets/normalized_output.csv")

def normalize_inference():
    normalize_inference_data.import_inference_dataset()
    print(normalize_inference_data.csv_df)
    export_to_csv(normalize_inference_data.csv_df, "datasets/normalized_inference_output.csv")


def __run_inference__():
    if not os.path.exists("datasets/normalized_inference_output.csv"):
        print("Normalized data not found. Normalizing")
        normalize_inference()
    #Once it is normalized, prepare the the training data and start the training
    prepare_and_start_inference()


def __run_training__():
    # Normalize the data if it is not already normalized.
    if not os.path.exists("datasets/normalized_output.csv"):
        print("Normalized data not found. Normalizing")
        normalize_data()
    #Once it is normalized, prepare the the training data and start the training
    prepare_and_start_training()



# if __name__ == "__main__":
#     __init__()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a mode [training/inference]")

    input_model = None
    if sys.argv[1] == 'training':
        __run_training__()
    if sys.argv[1] == 'inference':
        __run_inference__()


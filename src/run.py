

import os
import sys
from datasets import dataset_handler
import datasets.normalize_inference_data as normalize_inference_data
from src.prepare_training_set import binary_quick_inference, prepare_and_start_inference, prepare_and_start_training, quick_inference, quick_train, sanity_check
from src.utils import export_to_csv


def normalize_data():
    dataset_handler.import_datasets()
    export_to_csv(dataset_handler.csv_df, "datasets/csvs/training/normalized_output.csv")

def normalize_inference():
    normalize_inference_data.import_inference_dataset()
    export_to_csv(normalize_inference_data.csv_df, "datasets/csvs/inference/normalized_output.csv")


def __run_inference__():
    if not os.path.exists("datasets/csvs/inference/normalized_output.csv"):
        print("Normalized data not found. Normalizing")
        normalize_inference()
    #Once it is normalized, prepare the the training data and start the training
    prepare_and_start_inference()


def __run_training__():
    # Normalize the data if it is not already normalized.
    if not os.path.exists("datasets/csvs/training/normalized_output.csv"):
        print("Normalized data not found. Normalizing")
        normalize_data()
    #Once it is normalized, prepare the the training data and start the training
    prepare_and_start_training()


def __run_quick_inference__():
    quick_inference()


def __run_quick_training__():
    quick_train()

def __run_sanity__():
    sanity_check()

def __run_quick_binary_inference__():
    binary_quick_inference()

def print_help():
    print("Usage:")
    print("  python -m src.run inference")
    print("  python -m src.run training")
    print("  python -m src.run quick-inference")
    print("  python -m src.run quick-training")
    print("  python -m src.run sanity")
    print("  python -m src.run binary-quick-inference")


# if __name__ == "__main__":
#     __init__()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide a mode [training/inference]")

    input_model = None
    if sys.argv[1] == 'training':
        __run_training__()
    elif sys.argv[1] == 'inference':
        __run_inference__()
    elif sys.argv[1] == 'quick-training':
        __run_quick_training__()
    elif sys.argv[1] == 'quick-inference':
        __run_quick_inference__()
    elif sys.argv[1] == 'binary-quick-inference':
        __run_quick_binary_inference__()
    elif sys.argv[1] == 'sanity':
        __run_sanity__()
    else:
        print_help()


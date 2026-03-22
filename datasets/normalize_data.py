from urllib.parse import urlparse

import pandas

from datasets import  dataset_handler
from src.utils import export_to_csv


def __init__():
    dataset_handler.import_datasets()
    export_to_csv(dataset_handler.csv_df, "datasets/normalized_output.csv")

if __name__ == "__main__":
    __init__()
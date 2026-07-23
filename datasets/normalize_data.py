from urllib.parse import urlparse

import pandas

from datasets import  dataset_handler
from src.utils import export_to_csv


def __init__():
    """Normalize the training knowledge-base CSVs and export the merged result.

    This entry point loads the museum datasets through the shared dataset handler,
    performs the normalization pipeline, and writes the consolidated training data to
    datasets/csvs/training/normalized_output.csv.
    """
    dataset_handler.import_datasets()
    export_to_csv(dataset_handler.csv_df, "datasets/csvs/training/normalized_output.csv")


if __name__ == "__main__":
    __init__()
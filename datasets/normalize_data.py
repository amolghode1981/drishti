from urllib.parse import urlparse

import pandas

import dataset_handler as dataset_handler


    #print(dataframe)

def __init__():
    dataset_handler.import_datasets()
    dataset_handler.export_to_csv(dataset_handler.csv_df, "datasets/normalized_output.csv")

if __name__ == "__main__":
    __init__()
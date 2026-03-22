import os

import pandas
from cma_connector import CMAConnector
from met_connector import MetConnector
from utils import get_instrument_labels, get_directory_name_for_instrument, get_last_url_component
from sklearn.model_selection import train_test_split
# idea is that we want to use 20% of the images for validation.

def stratified_split(df, label, val_ratio=0.2):
    # Filter images that contain this label
    label_df = df[df["instruments"].apply(lambda x: label in x)]
    if label_df.empty:
        print(f"No data for instrument {label}")
        return pandas.DataFrame(), pandas.DataFrame()
    train, val = train_test_split(label_df, test_size=val_ratio, random_state=42)
    return train, val

# Following structure needs to be prepared
# clip_instruments/
# ├── venv/
# ├── setup_dirs.py
# └── data/
#     ├── train/
#     │   ├── guitar/
#     │   ├── violin/
#     │   └── non_instrument/
#     └── val/
#         ├── guitar/
#         ├── violin/
#         └── non_instrument/

def create_directory_structure(csvHeaders):
    instruments = get_instrument_labels(csvHeaders)
    #preparing training folder structure
    folder_name = "data"
    os.makedirs(folder_name, exist_ok=True)
    folder_name = "data/train"
    os.makedirs(folder_name, exist_ok=True)
    for instrument in instruments:
        os.makedirs(f"data/train/{get_directory_name_for_instrument(instrument)}", exist_ok=True)
    folder_name = "data/train/non_instrument"
    os.makedirs(folder_name, exist_ok=True)

    #preparing validation folder structure
    folder_name = "data/val"
    os.makedirs(folder_name, exist_ok=True)
    for instrument in instruments:
        os.makedirs(f"data/val/{get_directory_name_for_instrument(instrument)}", exist_ok=True)
    folder_name = "data/val/non_instrument"
    os.makedirs(folder_name, exist_ok=True)
    get_instrument_labels(csvHeaders)

def download_training_data(instrument, train):
    instrument_directory = f"data/train/{get_directory_name_for_instrument(instrument)}"
    metConnector = MetConnector("met")
    cmaConnector = CMAConnector("cma")
    i = 0
    for index, training_object in train.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        if training_object["museum_index"] == "met":
            metConnector.download_image(training_object["image_url"], f"{instrument_directory}/{file_name}")
        if training_object["museum_index"] == "cma":
            cmaConnector.download_image(training_object["image_url"], f"{instrument_directory}/{file_name}")
        # #Pseudo test limit for testing
        # if i > 3:
        #     print("breaking")
        #     break
        # i = i + 1

def download_validation_data(instrument, train):
    instrument_directory = f"data/val/{get_directory_name_for_instrument(instrument)}"
    metConnector = MetConnector("met")
    cmaConnector = CMAConnector("cma")
    i = 0
    for index, training_object in train.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        if training_object["museum_index"] == "met":
            metConnector.download_image(training_object["image_url"], f"{instrument_directory}/{file_name}")
        if training_object["museum_index"] == "cma":
            cmaConnector.download_image(training_object["image_url"], f"{instrument_directory}/{file_name}")
        # #Pseudo test limit for testing
        # if i > 3:
        #     print("breaking")
        #     break
        # i = i + 1

def prepare_trainingset(instruments, training_dataframe):
    #print(training_dataframe)
    for instrument in instruments:
        train, val = stratified_split(training_dataframe, instrument)
        if not train.empty and not val.empty:
            download_training_data(instrument, train)
            download_validation_data(instrument, val)


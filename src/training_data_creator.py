import os

import pandas
from src.cma_connector import CMAConnector
from src.met_connector import MetConnector
from src.utils import get_instrument_labels, get_directory_name_for_instrument, get_last_url_component
from sklearn.model_selection import train_test_split
'''
there are some errors in the data partitioning. Code does not consider additional instruments.
it only partitions on known instruments. This might bite us back later. Lets add completely different
set of images for non_instruments
'''
# idea is that we want to use 20% of the images for validation.
training_images = []
validation_images = []
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

def download_training_data(instrument, dataset):
    instrument_directory = f"data/train/{get_directory_name_for_instrument(instrument)}"
    i = 0
    for index, training_object in dataset.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        if training_images.count(file_name) == 0:
            training_images.append(file_name)

        download_data_shorthand(training_object, instrument_directory)
        # #Pseudo test limit for testing
        # #Pseudo test limit for testing
        # if i > 3:
        #     print("breaking")
        #     break
        # i = i + 1

def download_validation_data(instrument, dataset):
    instrument_directory = f"data/val/{get_directory_name_for_instrument(instrument)}"
    i = 0
    for index, training_object in dataset.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        if validation_images.count(file_name) == 0:
            validation_images.append(file_name)
        download_data_shorthand(training_object, instrument_directory)
        # #Pseudo test limit for testing
        # if i > 3:
        #     print("breaking")
        #     break
        # i = i + 1
def download_data_shorthand(row, local_parent_dir):
        metConnector = MetConnector("met")
        cmaConnector = CMAConnector("cma")
        file_name = get_last_url_component(row["image_url"])
        if row["museum_index"] == "met":
            metConnector.download_image(row["image_url"], f"{local_parent_dir}/{file_name}")
        if row["museum_index"] == "cma":
            cmaConnector.download_image(row["image_url"], f"{local_parent_dir}/{file_name}")



def download_non_instrument_data(dataset):
    no_instrument_training_directory = f"data/train/non_instrument"
    no_instrument_validation_directory = f"data/val/non_instrument"
    i = 0
    current_parent = no_instrument_training_directory
    for index, training_object in dataset.iterrows():
        # Put in alternate traing and validation folder
        download_data_shorthand(training_object, current_parent)
        if current_parent == no_instrument_training_directory:
            current_parent = no_instrument_validation_directory
        elif current_parent == no_instrument_validation_directory:
            current_parent = no_instrument_training_directory

def verify_exclusive(lt, lv, instrument):
    # print(lt)
    # print(lv)
    intersection = set(lt["id"]) & set(lv["id"])
    print(f"Instrument {instrument} Common rows: {len(intersection)}")
    # intersection = list(set(lt) & set(lv))
    # if len(intersection) != 0:
    #     print(f"Instrument = {instrument} intersection = {intersection} \n")


def prepare_trainingset(instruments, training_dataframe):
    for instrument in instruments:
        train, val = stratified_split(training_dataframe, instrument)
        if not train.empty:
            download_training_data(instrument, train)
        if not val.empty:
            download_validation_data(instrument, val)

    intersection = list(set(training_images) & set(validation_images))
    print(intersection)
    # empty_df = training_dataframe[training_dataframe["instruments"].apply(lambda x: len(x) == 0)]
    # download_non_instrument_data(empty_df)

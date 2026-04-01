import os

import pandas
from src.utils import export_to_csv, get_instrument_labels, get_directory_name_for_instrument, get_last_url_component, getConnectorForMuseumId
from sklearn.model_selection import train_test_split

from src.training import run_training_loop, init
'''
there are some errors in the data partitioning. Code does not consider additional instruments.
it only partitions on known instruments. This might bite us back later. Lets add completely different
set of images for non_instruments
'''
# idea is that we want to use 20% of the images for validation.
training_images = []
validation_images = []

trainingdata = pandas.DataFrame()
validationdata = pandas.DataFrame()

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

def create_directory_structure(csvHeaders, mode = "training"):
    instruments = get_instrument_labels(csvHeaders)
    #preparing training folder structure
    folder_name = "data"
    os.makedirs(folder_name, exist_ok=True)
    folder_name = f"data/{'train' if mode == 'training' else 'inference'}"
    os.makedirs(folder_name, exist_ok=True)
    for instrument in instruments:
        os.makedirs(f"data/{'train' if mode == 'training' else 'inference'}/{get_directory_name_for_instrument(instrument)}", exist_ok=True)
    folder_name = f"data/{'train' if mode == 'training' else 'inference'}/non_instrument"
    os.makedirs(folder_name, exist_ok=True)

    #Validation folder is applicable only for the training mode
    if mode == 'training':
        #preparing validation folder structure
        folder_name = "data/val"
        os.makedirs(folder_name, exist_ok=True)
        for instrument in instruments:
            os.makedirs(f"data/val/{get_directory_name_for_instrument(instrument)}", exist_ok=True)
        folder_name = "data/val/non_instrument"
        os.makedirs(folder_name, exist_ok=True)
        get_instrument_labels(csvHeaders)

def download_training_data(instrument, dataset):
    global trainingdata
    instrument_directory = f"data/train/{get_directory_name_for_instrument(instrument)}"
    i = 0
    for _, training_object in dataset.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        local_path = f"{instrument_directory}/{file_name}"
        download_data_shorthand(training_object, local_path)
        if training_images.count(file_name) == 0:
            training_images.append(file_name)
        row = {
            "id": training_object["id"],
            "instrument": instrument,
            "museum_index": training_object["museum_index"],
            "local_path": local_path
        }
        trainingdata = pandas.concat([trainingdata, pandas.DataFrame([row])], ignore_index=True)

def download_inference_data(instrument, dataset):
    global trainingdata
    instrument_directory = f"data/inference/{get_directory_name_for_instrument(instrument)}"
    i = 0
    for _, training_object in dataset.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        local_path = f"{instrument_directory}/{file_name}"
        download_data_shorthand(training_object, local_path)
        if training_images.count(file_name) == 0:
            training_images.append(file_name)
        row = {
            "id": training_object["id"],
            "instrument": instrument,
            "museum_index": training_object["museum_index"],
            "local_path": local_path
        }
        trainingdata = pandas.concat([trainingdata, pandas.DataFrame([row])], ignore_index=True)


def download_validation_data(instrument, dataset):
    global validationdata
    instrument_directory = f"data/val/{get_directory_name_for_instrument(instrument)}"
    i = 0
    for _, training_object in dataset.iterrows():
        file_name = get_last_url_component(training_object["image_url"])
        local_path = f"{instrument_directory}/{file_name}"
        download_data_shorthand(training_object, local_path)
        if validation_images.count(file_name) == 0:
            validation_images.append(file_name)
        row = {
            "id": training_object["id"],
            "instrument": instrument,
            "museum_index": training_object["museum_index"],
            "local_path": local_path
        }
        validationdata = pandas.concat([validationdata, pandas.DataFrame([row])], ignore_index=True)

def download_data_shorthand(row, local_file):
        connector = getConnectorForMuseumId(row["museum_index"])
        connector.download_image(row["image_url"], local_file)

def prepare_trainingset(instruments, training_dataframe):
    for instrument in instruments:
        train, val = stratified_split(training_dataframe, instrument)
        if not train.empty:
            download_training_data(instrument, train)
        if not val.empty:
            download_validation_data(instrument, val)

    export_to_csv(trainingdata, "datasets/training_data_records.csv")
    export_to_csv(validationdata, "datasets/validation_data_records.csv")
    #init()
    #run_training_loop()

def prepare_inferenceset(instruments, inference_dataframe):
    for instrument in instruments:
        download_inference_data(instrument, inference_dataframe)
    export_to_csv(trainingdata, "datasets/inference_data_records.csv")

# def download_non_instrument_data(dataset):
#     no_instrument_training_directory = f"data/train/non_instrument"
#     no_instrument_validation_directory = f"data/val/non_instrument"
#     i = 0
#     current_parent = no_instrument_training_directory
#     for index, training_object in dataset.iterrows():
#         # Put in alternate traing and validation folder
#         download_data_shorthand(training_object, current_parent)
#         if current_parent == no_instrument_training_directory:
#             current_parent = no_instrument_validation_directory
#         elif current_parent == no_instrument_validation_directory:
#             current_parent = no_instrument_training_directory

import ast
import csv
import os

import pandas
import src.training_data_creator
import src.utils as utils
normalized_dataframe = None


def get_headers(file_path):
    with open(file_path) as f:
        headers = next(csv.reader(f))
    return headers

def get_depicted_instruments(headers, row):
    instrument_labels = utils.get_instrument_labels(headers)
    depicted_instruments = []
    for instrument in instrument_labels:
        # Either the instrument definitely exist or there is something similar with some uncertainty.
        # ignore the rest.
        if row[instrument] == 1 or row[instrument] == 3:
            depicted_instruments.append(instrument)
        elif instrument == "Additional instruments" and pandas.notna(row["Additional instruments"]):
            depicted_instruments.append(instrument)
    return depicted_instruments

def prepare_training_objects(file_path):
    csv_df = pandas.read_csv(file_path, dtype={"Museum Object Id": str})
    headers = get_headers(file_path)
    training_objects = []
    '''
    Parse row by row. Fetch the museum object to fetch the image url
    Once the image url is available, parse each row to get a structure such as
    image_url, title, [instruments]

    Then we can keep 20% images in the validation.
    '''
    for index, row in csv_df.iterrows():
        connector = utils.getConnectorForMuseumId(row["museum_index"])
        if row["downloaded"] == False:
            result = connector.fetchObject(row["Museum Object Id"])
        dict = {
            "image_url": result["image_url"],
            "title": result["title"],
            "museum_index": row["museum_index"],
            "id": row["Museum Object Id"]
        }
        depicted_intruments = get_depicted_instruments(headers, row)
        dict["instruments"] = depicted_intruments
        training_objects.append(dict)

    print(training_objects)
    with_instruments = pandas.DataFrame(training_objects)
    utils.export_to_csv(with_instruments, "datasets/training.csv")

def prepare_inference_objects(file_path):
    csv_df = pandas.read_csv(file_path, dtype={"Museum Object Id": str})
    # metConnector = MetConnector("met")
    # cmaConnector = CMAConnector("cma")
    # wamConnector = WAMConnector("wam")
    # mlaConnector = MLAConnector("mla")
    headers = get_headers(file_path)
    training_objects = []
    '''
    Parse row by row. Fetch the museum object to fetch the image url
    Once the image url is available, parse each row to get a structure such as
    image_url, title, [instruments]

    Then we can keep 20% images in the validation.
    '''
    for index, row in csv_df.iterrows():
        print(row["museum_index"])
        connector = utils.getConnectorForMuseumId(row["museum_index"])
        if row["downloaded"] == False:
            result = connector.fetchObject(row["Museum Object Id"])
        dict = {
            "image_url": result["image_url"],
            "title": result["title"],
            "museum_index": row["museum_index"],
            "id": row["Museum Object Id"]
        }
        depicted_intruments = get_depicted_instruments(headers, row)
        dict["instruments"] = depicted_intruments
        training_objects.append(dict)

    print(training_objects)
    with_instruments = pandas.DataFrame(training_objects)
    utils.export_to_csv(with_instruments, "datasets/inference_objects.csv")

'''

'''
def prepare_and_start_training():
    global csv_df
    if not os.path.exists("datasets/training.csv"):
        print("File not found")
        prepare_training_objects("datasets/normalized_output.csv")
    with_instruments =  pandas.read_csv("datasets/training.csv")
    # instruments becomes a string array and has its issues while finding a instrument in this list
    # e.g Tabla is found in Tablak if it is string. So to avoid that convert it to python arrays.
    with_instruments["instruments"] = with_instruments["instruments"].apply(ast.literal_eval)
    #print(with_instruments)
    headers = get_headers("datasets/normalized_output.csv")
    available_instruments = utils.get_instrument_labels(headers)

    src.training_data_creator.create_directory_structure(headers)
    src.training_data_creator.prepare_trainingset(available_instruments, with_instruments)

def prepare_and_start_inference():
    global csv_df
    if not os.path.exists("datasets/inference_objects.csv"):
        print("File not found")
        prepare_inference_objects("datasets/normalized_inference_output.csv")
    with_instruments =  pandas.read_csv("datasets/inference_objects.csv")
    # instruments becomes a string array and has its issues while finding a instrument in this list
    # e.g Tabla is found in Tablak if it is string. So to avoid that convert it to python arrays.
    with_instruments["instruments"] = with_instruments["instruments"].apply(ast.literal_eval)
    #print(with_instruments)
    headers = get_headers("datasets/normalized_output.csv")
    available_instruments = utils.get_instrument_labels(headers)

    src.training_data_creator.create_directory_structure(headers, "inference")
    src.training_data_creator.prepare_inferenceset(available_instruments, with_instruments)




#prepare_and_start_training()
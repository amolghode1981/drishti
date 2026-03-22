import csv
from io import StringIO
import src.utils

import pandas
csv_df = None
def fix_trailing_commas(file_path):
    with open(file_path, "r") as f:
        return "".join(line.rstrip(",\n") + "\n" for line in f)

def read_csv(museumId, file_path):
    global csv_df
    cleaned_data = fix_trailing_commas(file_path)
    df = pandas.read_csv(StringIO(cleaned_data), dtype={"Museum Object Id": str})
    df["museum_index"] = museumId
    df["downloaded"] = False
    if csv_df is None:
        csv_df = df
    else:
        csv_df = pandas.concat([csv_df, df], ignore_index=True)

# Overwrite DF column mangling for same name columns
def adjust_column_names(expected_columns):
    expected_columns.extend(["museum_index", "downloaded"])
    csv_df.columns = expected_columns

# Wherever there is no gender information, add 3 signifying mixed gender
def fillup_empty_gender():
    global csv_df
    for col in csv_df.columns:
        if col == "Gender":
            csv_df[col] = csv_df[col].fillna(3).astype(int)


def normalize_object_id():
    for index, row in csv_df.iterrows():
        if pandas.isna(row["Museum Object Id"]):
            museum_object_id = src.utils.get_last_url_component(row["Source Url"])
            #row["Museum Object Id"] = museum_object_id
            csv_df.at[index, "Museum Object Id"] = museum_object_id

def normalize():
    fillup_empty_gender()
    normalize_object_id()
    float_cols = csv_df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        csv_df[col] = csv_df[col].fillna(0).astype(int)

def import_datasets():
    read_csv("met", "datasets/met.csv")
    read_csv("cma", "datasets/cleveland.csv")
    # Both csvs must have same columns. This assumes so.
    # Dataframes mangle the column names if there are duplicates. We do not want that.
    with open("datasets/cleveland.csv") as f:
        headers = next(csv.reader(f))
    adjust_column_names(headers)
    normalize()

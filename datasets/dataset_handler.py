import csv
from io import StringIO
from src.utils import get_last_url_component
import src.utils
import pandas

csv_df = None
def read_csv(file_path):
    global csv_df
    df = src.utils.read_csv(file_path)
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
            museum_object_id = get_last_url_component(row["Source Url"])
            csv_df.at[index, "Museum Object Id"] = museum_object_id

def normalize():
    fillup_empty_gender()
    normalize_object_id()
    float_cols = csv_df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        csv_df[col] = csv_df[col].fillna(0).astype(int)

def import_datasets():
    read_csv("datasets/csvs/training/knowledge_base/met.csv")
    read_csv("datasets/csvs/training/knowledge_base/cleveland.csv")
    # Both csvs must have same columns. This assumes so.
    # Dataframes mangle the column names if there are duplicates. We do not want that.
    with open("datasets/csvs/training/knowledge_base/cleveland.csv") as f:
        headers = next(csv.reader(f))
    #Drop any column headers.
    headers = [col for col in headers if col != '']
    adjust_column_names(headers)
    csv_df["Museum Object Id"] = csv_df["Museum Object Id"].str.strip()
    normalize()


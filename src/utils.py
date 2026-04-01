from io import StringIO
from urllib.parse import parse_qs, urlparse

import pandas

from src.connectors.cma_connector import CMAConnector
from src.connectors.met_connector import MetConnector
from src.connectors.mla_connector import MLAConnector
from src.connectors.wam_connector import WAMConnector


def get_instrument_labels(headers):
    instruments = []
    '''
    headers contain lot of other fields along with the gender of the performer. Along with the other fields
    such as objectId gender will be dropped.
    '''
    subarray = headers[5:28]
    for header in subarray:
        if header == "Gender":
            continue
        # once the instrument list is prepared, replace / in Tambora/tambur to tambora_or_tambur
        # and space with underscore.
        instruments.append(header)
    return instruments

def get_directory_name_for_instrument(instrument_name):
    return instrument_name.replace("/", " or ").replace(" ", "_").lower()

def export_to_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)

def get_last_url_component(url: str) -> str:
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    if "download" in query_params:
        return query_params["download"][0]  # e.g. "PS1_W.890_Fnt_DD_T10.jpg"

    # fallback to path component if no download param
    return parsed.path.rstrip('/').split('/')[-1]

def fix_trailing_commas(file_path):
    with open(file_path, "r") as f:
        return "".join(line.rstrip(",\n") + "\n" for line in f)

def getMusiumIdFromName(name):
    match name:
        case "The morgan library and collection" :
            return "mlc"
        case "The Walter's Art museum":
            return "wam"
        case "Metropolitan museum newyork":
            return "met"
        case "The cleveland museum of art":
            return "cma"

def read_csv(file_path):
    cleaned_data = fix_trailing_commas(file_path)
    df = pandas.read_csv(StringIO(cleaned_data), dtype={"Museum Object Id": str})
    for index, row in df.iterrows():
        df.at[index, "museum_index"] = getMusiumIdFromName(row["Collection"])
    df["downloaded"] = False
    return df

def getConnectorForMuseumId(id):
    match(id):
        case "met":
            return  MetConnector("met")
        case "cma":
            return CMAConnector("cma")
        case "wam":
            return WAMConnector("wam")
        case "mlc":
            return MLAConnector("mla")
        case default:
            return None
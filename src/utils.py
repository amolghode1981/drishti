from urllib.parse import urlparse


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
    path = urlparse(url).path
    last_component =  path.rstrip('/').split('/')[-1]
    return last_component

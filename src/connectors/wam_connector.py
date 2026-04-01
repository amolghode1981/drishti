from src.connectors.connector import Connector
import uuid
'''
Connector for The Walter's Art museum.
It does not have any APIs. Download URLs are directly explosed. So fetchObject hardcodes
the output.
'''
class WAMConnector(Connector):
    def __init__(self, museum):
        super().__init__(museum)
        self.apiBaseUrl = "https://art.thewalters.org/object"


    def fetchObject(self, objectId):
        print(f"WAMConnector : No fetching required. Returning required data for {objectId}");
        dict = {}
        dict["image_url"] = f"https://art.thewalters.org/object/{objectId}"
        dict["title"] = "Inference Image"
        dict["id"] = uuid.uuid1()
        return dict

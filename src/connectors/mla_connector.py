from src.connectors.connector import Connector
import uuid
'''
Connector for The morgan library and collection.
It does not have any APIs. Download URLs are directly explosed. So fetchObject hardcodes
the output.
'''
class MLAConnector(Connector):
    def __init__(self, museum):
        super().__init__(museum)
        self.apiBaseUrl = "https://www.themorgan.org/sites/default/files/facsimile"


    def fetchObject(self, objectId):
        print(f"WMAConnector : No fetching required. Returning required data for {objectId}");
        dict = {}
        dict["image_url"] = f"https://www.themorgan.org/sites/default/files/facsimile/{objectId}"
        dict["title"] = "Inference Image"
        dict["id"] = uuid.uuid1()
        return dict

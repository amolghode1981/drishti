from src.connectors.connector import Connector

class CMAConnector(Connector):
    def __init__(self, museum):
        super().__init__(museum)
        self.apiBaseUrl = "https://openaccess-api.clevelandart.org/api/artworks"

    def __getObjectAttributes(self, result):
        dict = {}
        dict["image_url"] = result["data"]["images"]["web"]["url"]
        dict["title"] = result["data"]["title"]
        dict["id"] = result["data"]["id"]
        return dict

    def fetchObject(self, objectId):
        print(f"CMAConnector fetching {objectId}");
        objectUrl = f"{self.apiBaseUrl}/{objectId}"
        result = super().fetchObject(objectUrl)
        return self.__getObjectAttributes(result)

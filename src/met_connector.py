from connector import Connector

class MetConnector(Connector):
    def __init__(self, museum):
        super().__init__(museum)
        self.apiBaseUrl = "https://collectionapi.metmuseum.org/public/collection/v1"

    def __getObjectAttributes(self, result):
        dict = {}
        dict["image_url"] = result["primaryImage"]
        dict["title"] = result["title"]
        dict["id"] = result["objectID"]
        return dict

    def fetchObject(self, objectId):
        objectUrl = f"{self.apiBaseUrl}/objects/{objectId}"
        result = super().fetchObject(objectUrl)
        return self.__getObjectAttributes(result)

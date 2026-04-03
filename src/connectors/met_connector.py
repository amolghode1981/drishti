import os
import time

import pandas as pd

from src.connectors.connector import Connector
import urllib

class RateLimiter:
    def __init__(self, calls_per_second=2):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_call = time.time()

class MetConnector(Connector):
    def __init__(self, museum):
        super().__init__(museum)
        self.apiBaseUrl = "https://collectionapi.metmuseum.org/public/collection/v1"
        # MET supports 80 calls per second. Nonetheless, lets be conservative and put 50
        self.rateLimiter = RateLimiter(calls_per_second=10)

    def __getObjectAttributes(self, result):
        dict = {}
        dict["image_url"] = result["primaryImage"]
        dict["title"] = result["title"]
        dict["id"] = result["objectID"]
        return dict

    def fetchObject(self, objectId):
        print(f"MetConnector fetching {objectId}");
        objectUrl = f"{self.apiBaseUrl}/objects/{objectId}"
        result = super().fetchObject(objectUrl)
        return self.__getObjectAttributes(result)

    def listObjects(self, query):
        queryString = urllib.parse.urlencode(query)
        listUrl = f"{self.apiBaseUrl}/objects?{queryString}"
        result = super().fetchObject(listUrl)
        print(result)

    def shouldIncludeObject(self, object):
        try:
            print(f"shouldIncludeObject object type = {object["classification"]}")
            if (object["classification"].lower() == "paintings" or\
                object["classification"].lower() == "page") and \
                "india" in object["culture"].lower() and\
                object["primaryImage"] != "":
                return True
        except Exception as e:
            print(f"Error: {e}")
        return False

    def __import_Object(self, object):
        from src.utils import get_last_url_component
        localPath = get_last_url_component(object["image_url"])
        localPath = f"data/bulk_met/{localPath}"
        print(localPath)
        object["local_path"] = localPath
        super().download_image(object["image_url"], object["local_path"])

    def __append_to_csv(self, dataframe, filepath):
        if os.path.exists(filepath):
            dataframe.to_csv(filepath, mode='a', header=False, index=False)  # append, no header
        else:
            dataframe.to_csv(filepath, mode='w', header=True, index=False)   # create with header

    def filterMiniatureObjects(self, query):
        miniatures = []
        #queryString = urllib.parse.urlencode(query)
        #listUrl = f"{self.apiBaseUrl}/objects?{queryString}"
        listUrl = f"{self.apiBaseUrl}/search?q=India&medium=Paintings"
        result = super().fetchObject(listUrl)
        objectIds = result["objectIDs"]
        for objectId in objectIds[1000:]:
            try:
            #self.rateLimiter.wait()
                time.sleep(0.5)
                objectUrl = objectUrl = f"{self.apiBaseUrl}/objects/{objectId}"
                print(objectUrl)
                result = super().fetchObject(objectUrl, max_retries=None, delay=2)
                if result:
                    if self.shouldIncludeObject(result):
                        object = self.__getObjectAttributes(result)
                        self.__import_Object(object)
                        miniatures.append(object)
            except Exception as e:
                print("Foobar")
                print(e)
                continue

        dataframe = pd.DataFrame(miniatures)
        print(len(miniatures))
        self.__append_to_csv(dataframe, "output/sample_data.csv")
        time.sleep(5)
        print("-----Done----")
if __name__ == "__main__":
    connector = MetConnector("met")
    params = {
        "departmentIds": 6
    }
    connector.filterMiniatureObjects(params)
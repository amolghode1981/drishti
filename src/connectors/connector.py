import os

import requests
class Connector:
    def __init__(self, museum):
        self.museum = museum

    def listObjects(self):
        pass

    def fetchObject(self, objectUrl):
        try:
            response = requests.get(objectUrl)
            response.raise_for_status()  # raises error for 4xx/5xx status codes
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
        except requests.exceptions.ConnectionError:
            print("Connection error")
        except requests.exceptions.JSONDecodeError:
            print("Response is not valid JSON")
        return None

    def download_image(self, image_url, local_path):
        if os.path.exists(local_path):
            #print(f"Skipping download {image_url}. local file {local_path} exists")
            return;
        try:
            response = requests.get(image_url)
            print(f"downloading {image_url} to {local_path}")
            with open(local_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
                print(f"Error: {e}")

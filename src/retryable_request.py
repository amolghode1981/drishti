import time
import requests

def retryable_request(url, params=None, headers=None, max_retries=None, delay=1):
    attempt = 0
    while True:
        try:
            response = requests.get(url, params=params, headers=headers)

            if response.status_code == 200:
                return response
            elif response.status_code in [403, 404]:
                raise Exception(f"Object not found {url}")
            elif response.status_code == 429:  # rate limited
                wait = int(response.headers.get("Retry-After", 60))
                print(f"Rate limited. Waiting {wait} seconds...")
                time.sleep(wait)
            elif response.status_code in [500, 502, 503, 504]:  # server errors
                print(f"Server error {response.status_code}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Failed with status {response.status_code}. Retrying in {delay}s...")
                time.sleep(delay)

        except requests.exceptions.ConnectionError:
            print(f"Connection error. Retrying in {delay}s...")
            time.sleep(delay)
        except requests.exceptions.Timeout:
            print(f"Timeout. Retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:
            if "Object not found" in str(e):
                raise e
            print(f"Unexpected error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
        attempt += 1
        if max_retries and attempt >= max_retries:
            print(f"Max retries ({max_retries}) reached.")
            return None
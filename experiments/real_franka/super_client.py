"""
The client only needs a Python environment and the requests library (pip install requests); 
no other dependencies need to be installed.

Client (Standalone) Usage (assuming a server running on 0.0.0.0:5500):

"""
import requests
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Define the API endpoint
url = 'http://127.0.0.1:5500/api/inference'
# Create a session to maintain persistent connections
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=retries))
# Optional: Set headers to ensure long connections
session.headers.update({"Connection": "keep-alive"})

# Define the parameters you want to send
while True:
    for image in ['./asset/teaser.png', './asset/teaser.png']:
        start_time_1 = time.time()
        data = {
            'task_description': "Pick up the red can.",
            'reset': False,
        }
        # Write the data to a json file
        json.dump(data, open("data.json", "w"))

        with open("data.json", "r") as query_file:
            with open(image, "rb") as image_file:
                file = [
                    ('images', (image, image_file, 'image/jpeg')),
                    ('json', ("data.json", query_file, 'application/json'))
                ]
                start_time_0 = time.time()
                print('In request!!!!!!!')
                
                try:
                    # Make a POST request using the session
                    response = session.post(url, files=file, timeout=0.25)
                    print('Communication cost:', time.time() - start_time_0)
                    
                    if response.status_code == 200:
                        print(response.text)
                    else:
                        print("Failed to get a response from the API")
                        print(response.text)
                
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")
                    continue
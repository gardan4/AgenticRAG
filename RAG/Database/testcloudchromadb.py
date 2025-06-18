
import requests

url = "http://35.158.55.28:8000/collections"

response = requests.get(url)
print(response.status_code)
print(response.json())

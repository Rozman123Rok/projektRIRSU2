import requests
import json 

url = "http://127.0.0.1:5000/predict"

headers = {
    "Content-Type": "application/json"
}

# iir_request
with open("iir_request.json", "r") as f:
    data = json.load(f)


response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())

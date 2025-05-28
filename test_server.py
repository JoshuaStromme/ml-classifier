import requests

url = "https://api.cortex.cerebrium.ai/v4/p-bf25a4ed/ml-classifier/run"
image_path = "n01667114_mud_turtle.JPEG"

with open(image_path, "rb") as f:
    files = {"file": (image_path, f, "image/jpeg")}
    response = requests.post(url, files=files)
    
print("Status Code:", response.status_code)
print("Response:", response.json())
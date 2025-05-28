import requests
import base64
import json

url = "https://api.cortex.cerebrium.ai/v4/p-bf25a4ed/ml-classifier/predict"
image_path = "n01667114_mud_turtle.JPEG"
api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJwcm9qZWN0SWQiOiJwLWJmMjVhNGVkIiwibmFtZSI6IiIsImRlc2NyaXB0aW9uIjoiIiwiZXhwIjoyMDYzOTgyNDExfQ.xL67qYuQN4WOLJPw0xCFDfNa0ZQGFKv6cJsU75HBp9wmHAhrMiw0KDao_EDtORhYmtD126ALwhVh8J3WVSveNSWaxOzh4ddKRWp6gsjhhAbKmwTy1gQkTZmk227-24OZKL_sPv-N2UWBHykCFO_MlOIJiqPENOIJ0z5-ys004Cr4zHtmdw0HXp5oTyIosgWAB9-R_kyrKICJpnByEb-5ahy188vCq3LmrCaKIX9itLkez8lSUhkf9GivB-42YkwIkQluU6ZSTUFTRII5t_tTMUipnkxjh6k2P5Vq3OlX83z0LxPooTy17GF-DE0Xg9TWaf880Ub9fjS9b_PMjlmodQ"

with open(image_path, "rb") as f:
    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "image": img_b64
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
    
print("Status Code:", response.status_code)
print("Response:", response.json())
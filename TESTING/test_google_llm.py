
import requests

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
headers = {
    "x-goog-api-key": "AIzaSyD2ObasyPUZRv47m6bv9bcTs6HDWxs7HA8",
    "Content-Type": "application/json",
}
payload = {"contents": [{"parts": [{"text": "Hello, AI world!"}]}]}

response = requests.post(url, headers=headers, json=payload, timeout=30)
print(response.status_code)
print(response.text)
response.raise_for_status()
print(response.json())

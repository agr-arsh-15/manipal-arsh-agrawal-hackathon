import requests
import json
import time

def probe_gdelt():
    # Simple query with polite user-agent and retry delay
    url = "https://api.gdeltproject.org/api/v2/doc/doc?query=Fed&mode=ArtList&maxrecords=5&format=json"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    print("Waiting 6 seconds before querying GDELT to respect rate-limit...")
    time.sleep(6)
    r = requests.get(url, headers=headers, timeout=20)
    print(f"HTTP Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        articles = data.get("articles", [])
        print(f"Received {len(articles)} articles.")
        if articles:
            sample = articles[0]
            print("Available Keys:", list(sample.keys()))
            print("Sample Title:", sample.get("title"))
            print("Sample URL:", sample.get("url"))
            print("Sample Seendate:", sample.get("seendate"))
    else:
        print("Response:", r.text[:200])

if __name__ == "__main__":
    probe_gdelt()

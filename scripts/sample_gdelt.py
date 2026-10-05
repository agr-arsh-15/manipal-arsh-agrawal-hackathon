import requests
import json
import time

def fetch_gdelt_sample():
    # Use standard GDELT 2.0 query format
    url = "https://api.gdeltproject.org/api/v2/doc/doc?query=Fed&mode=artlist&format=json&maxrecords=5"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    time.sleep(5)
    r = requests.get(url, headers=headers, timeout=15)
    print("Status code:", r.status_code)
    try:
        data = r.json()
        articles = data.get("articles", [])
        print(f"Retrieved {len(articles)} articles.")
        for a in articles[:3]:
            print(f"- Title: {a.get('title')}")
            print(f"  URL: {a.get('url')}")
            print(f"  Seendate: {a.get('seendate')}")
            print(f"  Domain: {a.get('domain')}")
            print(f"  Language: {a.get('language')}")
    except Exception as e:
        print("Raw response preview:", r.text[:300])

if __name__ == "__main__":
    fetch_gdelt_sample()

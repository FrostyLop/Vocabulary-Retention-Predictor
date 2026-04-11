import requests
import json
import time
import os
from dotenv import load_dotenv

# Load the API token from the .env file
load_dotenv()
API_TOKEN = os.getenv("WANIKANI_API_TOKEN")

# Base URL
BASE_URL = "https://api.wanikani.com/v2"

# Headers for authentication
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Function to fetch all paginated data from a WaniKani endpoint
def fetch_all_pages(endpoint):
    url = f"{BASE_URL}/{endpoint}"
    all_data = []
    single_object_data = None
    saw_list_data = False

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()

        page_data = data.get("data")

        # Handle both list-based (paginated) and object-based endpoints.
        if isinstance(page_data, list):
            saw_list_data = True
            all_data.extend(page_data)
        elif page_data is not None:
            single_object_data = page_data

        # Get next page URL
        url = (data.get("pages") or {}).get("next_url")

        # Respect rate limits
        if url:
            time.sleep(0.5)

    if saw_list_data:
        return all_data

    return single_object_data

# Function to save data to JSON file
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Starting WaniKani API pull...")

    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    endpoints = ["review_statistics","subjects", "assignments", "summary"]

    for endpoint in endpoints:
        endpoint_data = fetch_all_pages(endpoint)
        output_file = os.path.join(output_dir, f"{endpoint}.json")
        save_to_json(endpoint_data, output_file)
        print(f"Saved {endpoint} data to {output_file}")
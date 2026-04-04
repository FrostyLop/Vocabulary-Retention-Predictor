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

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()

        # Append current page data
        all_data.extend(data["data"])

        # Get next page URL
        url = data["pages"]["next_url"]

        # Respect rate limits
        time.sleep(0.5)

    return all_data

# Function to save data to JSON file
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("Starting WaniKani API pull...")

    # Pull summary data
    assignments = fetch_all_pages("assignments")
    assignments
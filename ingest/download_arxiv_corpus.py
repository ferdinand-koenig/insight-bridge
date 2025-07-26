"""
Downloads PDFs from arXiv for a given category and date, based on config.
"""

import time
import requests
import feedparser
from pathlib import Path
from datetime import datetime
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DOCS_PATH = Path(config["raw_docs_path"])
ARXIV_CONFIG = config["arxiv_download"]
CATEGORIES = ARXIV_CONFIG["categories"]
YEAR = ARXIV_CONFIG["year"]
MONTH = ARXIV_CONFIG["month"]

# Prepare target date range for filtering
start_date = datetime(YEAR, MONTH, 1)
if MONTH == 12:
    end_date = datetime(YEAR + 1, 1, 1)
else:
    end_date = datetime(YEAR, MONTH + 1, 1)

RAW_DOCS_PATH.mkdir(parents=True, exist_ok=True)


def fetch_entries(category):
    base_url = "http://export.arxiv.org/api/query"
    start = 0
    max_results = 1000
    all_entries = []

    while True:
        query = f"cat:{category}"
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        response = requests.get(base_url, params=params)
        feed = feedparser.parse(response.text)

        if not feed.entries:
            break

        for entry in feed.entries:
            published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
            if published < start_date:
                return all_entries
            if start_date <= published < end_date:
                all_entries.append(entry)

        start += max_results
        time.sleep(0.2)

    return all_entries


def download_pdf(entry):
    arxiv_id = entry.id.split('/abs/')[-1]
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    save_path = RAW_DOCS_PATH / filename

    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Saved {filename}")
    except Exception as e:
        print(f"❌ Failed to download {arxiv_id}: {e}")


def main():
    for category in CATEGORIES:
        print(f"Fetching papers for category: {category}")
        entries = fetch_entries(category)
        print(f"Found {len(entries)} papers.")

        for entry in entries:
            download_pdf(entry)
            time.sleep(0.2)  # Be polite to arXiv servers


if __name__ == "__main__":
    main()

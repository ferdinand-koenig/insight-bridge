"""
download_arxiv_corpus.py

This script downloads the Arxiv corpus from KaggleHub and saves it to `data/raw/`
for ingestion into the InsightBridge document processing pipeline.

Usage:
    poetry run python ingest/download_arxiv_corpus.py
"""

from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Define constants
ARXIV_DATASET_ID = "Cornell-University/arxiv"
SAVE_DIR = Path("data/raw")
FILENAME = "arxiv-metadata-oai-snapshot.json"


def download_arxiv():
    # Ensure the raw data directory exists
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Arxiv corpus from KaggleHub ({ARXIV_DATASET_ID})...")

    # Load the dataset via KaggleHub using pandas
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        ARXIV_DATASET_ID,
        file_path=FILENAME
    )

    # Save locally as a backup or to support offline access
    save_path = SAVE_DIR / FILENAME
    df.to_json(save_path, orient="records", lines=True)

    print(f"Downloaded and saved Arxiv corpus to: {save_path}")
    print("Sample records:\n", df.head())


if __name__ == "__main__":
    download_arxiv()

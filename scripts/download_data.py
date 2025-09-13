#!/usr/bin/env python3
"""
Script to download data from Zenodo record 10463156.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import json


def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """Download a file from URL with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as file,
            tqdm(
                desc=filepath.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def get_zenodo_files(record_id: str) -> list:
    """Get file information from Zenodo record."""
    api_url = f"https://zenodo.org/api/records/{record_id}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        files = []
        for file_info in data.get("files", []):
            files.append(
                {
                    "filename": file_info["key"],
                    "download_url": file_info["links"]["self"],
                    "size": file_info["size"],
                }
            )

        return files
    except Exception as e:
        print(f"Error fetching Zenodo record: {e}")
        return []


def main():
    """Main function to download data from Zenodo."""
    record_id = "10463156"
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"Fetching file information for Zenodo record {record_id}...")
    files = get_zenodo_files(record_id)

    if not files:
        print("No files found or error occurred.")
        return

    print(f"Found {len(files)} files:")
    for i, file_info in enumerate(files, 1):
        size_mb = file_info["size"] / (1024 * 1024)
        print(f"{i}. {file_info['filename']} ({size_mb:.2f} MB)")

    # Download all files
    for file_info in files:
        filepath = data_dir / file_info["filename"]

        if filepath.exists():
            print(f"File {filepath.name} already exists, skipping...")
            continue

        print(f"Downloading {file_info['filename']}...")
        success = download_file(file_info["download_url"], filepath)

        if success:
            print(f"Successfully downloaded {filepath.name}")
        else:
            print(f"Failed to download {filepath.name}")

    print("Download process completed.")


if __name__ == "__main__":
    main()

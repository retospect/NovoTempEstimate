#!/usr/bin/env python3
"""
Script to download the paper PDF from DOI.
"""

import requests
from pathlib import Path
import time


def download_paper_pdf(doi_url: str, output_path: Path) -> bool:
    """Download paper PDF from DOI URL."""
    try:
        # Try to get the PDF directly from the DOI
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/pdf,*/*",
        }

        response = requests.get(doi_url, headers=headers, allow_redirects=True)

        # Check if we got a PDF
        if response.headers.get("content-type", "").startswith("application/pdf"):
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True

        # If not PDF, try to find PDF link in the page
        if "biorxiv" in response.url.lower():
            # For bioRxiv, construct the PDF URL
            pdf_url = response.url.replace(
                "/content/", "/content/biorxiv/early/"
            ).replace(".short", ".full.pdf")
            if not pdf_url.endswith(".pdf"):
                pdf_url = pdf_url + ".pdf"

            pdf_response = requests.get(pdf_url, headers=headers)
            if pdf_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(pdf_response.content)
                return True

        print(
            f"Could not find PDF. Response content type: {response.headers.get('content-type')}"
        )
        return False

    except Exception as e:
        print(f"Error downloading paper: {e}")
        return False


def main():
    """Main function to download the paper."""
    doi_url = "https://doi.org/10.1101/2023.03.27.534365"
    papers_dir = Path(__file__).parent.parent / "papers"
    papers_dir.mkdir(exist_ok=True)

    output_path = papers_dir / "protein_temperature_estimation_2023.pdf"

    if output_path.exists():
        print(f"Paper already exists at {output_path}")
        return

    print(f"Downloading paper from {doi_url}...")
    success = download_paper_pdf(doi_url, output_path)

    if success:
        print(f"Successfully downloaded paper to {output_path}")
    else:
        print("Failed to download paper. You may need to download it manually.")
        print(f"Please visit {doi_url} and save the PDF to {output_path}")


if __name__ == "__main__":
    main()

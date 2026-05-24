"""
download_images.py - Download sample images for image analysis demos.

Description
-----------
Downloads sample images from Pexels (free stock photos) for use with
the image compression demos.

Images are licensed under the Pexels License (free for personal and
commercial use, no attribution required).

Source: https://www.pexels.com/

Requirements
------------
Install required packages:
  pip install requests

For analysis scripts (test_image_id.py), also install:
  pip install numpy scipy matplotlib pillow

Author: Adrianna Gillman, Zydrunas Gimbutas
SPDX-License-Identifier: MIT
Version: 1.0.1
Date: April 22, 2026
Assisted by: Claude Code (Anthropic)
"""

import os
import requests
from pathlib import Path

# Image metadata: (pexels_id, photographer, local_filename)
IMAGES = [
    (149387, "flickr", "pexels-flickr-149387.jpg"),
    (4793404, "anniroenkae", "pexels-anniroenkae-4793404.jpg"),
    (7824822, "andre-ulysses-de-salis-2100065", "pexels-andre-ulysses-de-salis-2100065-7824822.jpg"),
]

DATA_DIR = Path(__file__).parent


def get_pexels_url(photo_id):
    """Construct Pexels download URL for a photo ID."""
    # Pexels provides direct image URLs in this format
    return f"https://images.pexels.com/photos/{photo_id}/pexels-photo-{photo_id}.jpeg"


def download_image(photo_id, photographer, filename):
    """Download a single image from Pexels."""
    outpath = DATA_DIR / filename

    if outpath.exists():
        print(f"  {filename} already exists, skipping")
        return str(outpath)

    url = get_pexels_url(photo_id)
    print(f"  Downloading {filename}...", end="", flush=True)

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(outpath, 'wb') as f:
            f.write(response.content)
        print(" done")
        return str(outpath)
    except Exception as e:
        print(f" failed ({type(e).__name__}: {e})")
        return None


def download_all_images():
    """Download all sample images."""
    print("=" * 60)
    print("Image Analysis Sample Data Downloader")
    print("=" * 60)
    print("""
This script downloads sample images from Pexels for the
image compression demos.

Images are free to use under the Pexels License.
Source: https://www.pexels.com/
    """)

    downloaded = []
    for photo_id, photographer, filename in IMAGES:
        result = download_image(photo_id, photographer, filename)
        if result:
            downloaded.append(result)

    print("\n" + "=" * 60)
    print(f"Download complete: {len(downloaded)}/{len(IMAGES)} images")
    print("=" * 60)

    return downloaded


def list_local_images():
    """List all downloaded images."""
    extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in DATA_DIR.iterdir()
             if f.suffix.lower() in extensions and f.name.startswith('pexels-')]
    return sorted([str(f) for f in files])


if __name__ == "__main__":
    files = download_all_images()

    # Show what we have
    local_files = list_local_images()
    if local_files:
        print(f"\nLocal images: {len(local_files)}")
        for f in local_files:
            print(f"  {os.path.basename(f)}")

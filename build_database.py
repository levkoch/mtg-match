"""
Database builder for MTG cards from Scryfall.
Downloads cards, extracts pHash, color histogram features, saves to JSON.
"""

import cv2
import numpy as np
import json
import requests
from PIL import Image
from pathlib import Path
from io import BytesIO

from config import BIN_VERSIONS, SETS


def extract_card_features(image: np.ndarray) -> dict:
    """Extract perceptual hash and color histogram from card image."""

    # first try - used perceptual hash (used in reference implementation) + color histogram since colors are more meaningful for MTG cards

    # Preprocess the image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # CLAHE for handling varying lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    processed = cv2.GaussianBlur(processed, (3, 3), 0)

    # 1. Perceptual hash
    hash_size = 16
    small = cv2.resize(processed, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    avg = np.mean(gray)
    binary_hash = (gray > avg).flatten()
    hash_bytes = np.packbits(binary_hash)
    perceptual_hash = "".join(f"{byte:02x}" for byte in hash_bytes)

    # 2. Color histogram [32, 32, 32] creates an output vector of size 32768 (too much)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

    out = {"perceptual_hash": perceptual_hash}

    # 3. histogram bins (orders of magnitude smaller)
    for bin_name, bin_counts in BIN_VERSIONS:
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bin_counts, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
        out[bin_name] = hist.flatten().tolist()
        
    return out


def fetch_all_cards_from_sets(sets_list: list) -> list:
    """Fetch cards from specified sets using Scryfall API."""
    all_cards = []

    for set_code in sets_list:
        print(f"Fetching cards from set: {set_code}")

        # Scryfall API endpoint for cards in a set
        url = f"https://api.scryfall.com/cards/search"
        params = {"q": f"set:{set_code}", "format": "json", "page": 1}

        while True:
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if "data" in data:
                    all_cards.extend(data["data"])
                    print(
                        f"  Fetched {len(data['data'])} cards from page {params['page']}"
                    )

                # Check if there are more pages
                if data.get("has_more", False):
                    params["page"] += 1
                else:
                    break

            except Exception as e:
                print(f"  Error fetching page {params['page']}: {e}")
                break

    print(f"Total cards fetched: {len(all_cards)}")
    return all_cards


def build_database_with_limit(
    max_cards: int = 2000, output_path: str = "./data/card_database.json"
):
    """
    Build card database with perceptual hash features, limited to max_cards.
    Also saves card images to disk.
    """
    print(f"Building perceptual hash-based card database (max {max_cards} cards)...")

    # Create directories
    images_dir = Path("./data/card_images")
    images_dir.mkdir(parents=True, exist_ok=True)

    cards_data = fetch_all_cards_from_sets(SETS)

    if not cards_data:
        print("No cards found!")
        return

    database = {}
    processed = 0

    for card in cards_data:
        if processed >= max_cards:
            break

        try:
            # Get card info
            card_name = card.get("name", "unknown")
            set_code = card.get("set", "unknown")

            # Get image URL (normal size)
            image_uris = card.get("image_uris", {})
            if "normal" not in image_uris:
                print(f"\n  Skipping {card_name}: No image available")
                continue

            image_url = image_uris["normal"]

            # Download image
            print(f"\r  Processing {processed + 1}/{max_cards}: {card_name}", end=" " * 40, flush=True)
            response = requests.get(image_url)
            response.raise_for_status()

            # Load image
            pil_image = Image.open(BytesIO(response.content))
            image_array = np.array(pil_image)

            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                print(f"Skipping {card_name}: Invalid image format")
                continue

            # Create database key
            norm_name = (card_name.replace(' ', '_').replace(',', '')
                        ).replace(':', '').replace('/', '').replace("'", '').replace('"', '')
            card_key = f"{set_code}_{norm_name}"

            # Save image to disk
            image_filename = f"{card_key}.png"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), image_bgr)

            # Extract features
            features = extract_card_features(image_bgr)

            # Create database entry
            database[card_key] = {
                "name": card_name,
                "set": set_code,
                "image_url": image_url,
                "image_file": str(image_path),  # local file path
                **features,  # include all features
            }

            processed += 1

        except Exception as e:
            print(f"  Error processing card: {e}")
            continue

    print()  # for newline after progress

    # Save database
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    print(f"Database saved with {len(database)} cards to {output_path}")
    print(f"Card images saved to {images_dir}")


def main():
    """Main function to build database."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build MTG card database with pHash and color histogram features."
    )
    parser.add_argument(
        "--max-cards", type=int, default=2000, help="Maximum number of cards to process"
    )
    parser.add_argument(
        "--output", default="./data/card_database.json", help="Output database file"
    )

    args = parser.parse_args()

    build_database_with_limit(args.max_cards, args.output)


if __name__ == "__main__":
    main()

"""
MTG card database builder using Scryfall bulk data and parallel processing.
Downloads cards, extracts pHash + color histogram features with CLAHE, saves to JSON.
"""

import cv2
import numpy as np
import json
import requests
from PIL import Image
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse

from config import BIN_VERSIONS


def fetch_bulk_cards(max_cards: int = 10000) -> list:
    """Download bulk card data from Scryfall (much faster for large datasets)."""
    print("Fetching bulk data info from Scryfall...")

    try:
        # Get bulk data endpoints
        response = requests.get("https://api.scryfall.com/bulk-data", timeout=30)
        response.raise_for_status()
        bulk_data = response.json()

        # Find the default cards bulk data
        default_cards_url = None
        for bulk in bulk_data["data"]:
            if bulk["type"] == "default_cards":
                default_cards_url = bulk["download_uri"]
                file_size = bulk.get("size", 0) / (1024 * 1024)  # Convert to MB
                print(f"Found bulk data: {file_size:.1f} MB")
                break

        if not default_cards_url:
            raise Exception("Could not find bulk card data URL")

        # Download the bulk file
        print("Downloading bulk card data (this may take a few minutes)...")
        start_time = time.time()
        response = requests.get(default_cards_url, timeout=300)  # 5 min timeout
        response.raise_for_status()
        download_time = time.time() - start_time
        print(f"Download completed in {download_time:.1f} seconds")

        # Parse JSON
        print("Parsing card data...")
        cards_data = response.json()

        # Filter to cards with normal images and reasonable data
        print("Filtering cards with images...")
        cards_with_images = []
        for card in cards_data:
            # Skip if no image
            if not card.get("image_uris", {}).get("normal"):
                continue

            # Skip if missing essential data
            if not card.get("name") or not card.get("set"):
                continue

            # Skip digital-only or special formats we don't want
            if card.get("digital", False):
                continue

            cards_with_images.append(card)

            # Stop if we have enough
            if len(cards_with_images) >= max_cards:
                break

        print(f"Found {len(cards_with_images)} cards with images")
        return cards_with_images[:max_cards]

    except Exception as e:
        print(f"Error downloading bulk data: {e}")
        return []


def extract_card_features(image: np.ndarray) -> dict:
    """Extract perceptual hash and color histogram from card image - SAME AS MATCHING."""

    try:
        # Preprocess the image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # CLAHE for handling varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        # 1. Perceptual hash
        hash_size = 16
        small = cv2.resize(
            processed, (hash_size, hash_size), interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        avg = np.mean(gray)
        binary_hash = (gray > avg).flatten()
        hash_bytes = np.packbits(binary_hash)
        perceptual_hash = "".join(f"{byte:02x}" for byte in hash_bytes)

        # 2. Color histogram
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        features = {"perceptual_hash": perceptual_hash}

        # 3. Histogram bins (all versions for consistency)
        for bin_name, bin_counts in BIN_VERSIONS:
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None, bin_counts, [0, 180, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
            features[bin_name] = hist.flatten().tolist()

        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return {}


def process_single_card(args):
    """Process a single card - download image and extract features."""
    card_data, images_dir = args

    try:
        card_name = card_data.get("name", "unknown")
        set_code = card_data.get("set", "unknown")

        # Get image URL
        image_uris = card_data.get("image_uris", {})
        image_url = image_uris["normal"]

        # Download image with timeout
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        # Load and convert image
        pil_image = Image.open(BytesIO(response.content))
        image_array = np.array(pil_image)

        if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            return None  # Skip invalid images

        # Create normalized filename
        norm_name = (
            card_name.replace(" ", "_")
            .replace(",", "")
            .replace(":", "")
            .replace("/", "")
            .replace("'", "")
            .replace('"', "")
        )
        card_key = f"{set_code}_{norm_name}"

        # Save image
        image_filename = f"{card_key}.png"
        image_path = images_dir / image_filename
        success = cv2.imwrite(str(image_path), image_bgr)
        if not success:
            return None

        # Extract features
        features = extract_card_features(image_bgr)
        if not features:
            return None

        return {
            "key": card_key,
            "data": {
                "name": card_name,
                "set": set_code,
                "image_url": image_url,
                "image_file": str(image_path),
                **features,
            },
        }

    except Exception as e:
        return None


def process_existing_card(args):
    """Process a card from existing image files."""
    image_path = args

    try:
        # Extract card info from filename
        filename = image_path.stem

        # Try to parse filename format (assuming something like "setcode_cardname.extension")
        if "_" in filename:
            parts = filename.split("_", 1)
            set_code = parts[0]
            card_name = parts[1].replace("_", " ")
        else:
            # Fallback: use filename as card name
            set_code = "unknown"
            card_name = filename.replace("_", " ")

        # Load and process image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Could not load image: {image_path}")
            return None

        # Extract features using your existing function
        features = extract_card_features(img)
        if not features:
            return None

        return {
            "key": filename,
            "data": {
                "name": card_name,
                "set": set_code,
                "image_url": "",  # No URL for local files
                "image_file": str(image_path),
                **features,
            },
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def build_database(
    max_cards: int = 10000,
    output_path: str = "./data/card_database_phash.json",
    max_workers: int = 8,
    card_images_dir: str = "./data/card_images",
    use_existing: bool = False,  # Add this parameter
):
    """Build card database using bulk data and parallel processing."""

    if use_existing:
        # Process existing images
        print(f"Building database from existing images in {card_images_dir}...")

        images_dir = Path(card_images_dir)
        if not images_dir.exists():
            print(f"Images directory not found: {card_images_dir}")
            return

        # Find all image files
        image_extensions = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))

        print(f"Found {len(image_files)} existing images")

        if not image_files:
            print("No image files found!")
            return

        # Process existing images in parallel
        print(f"Processing {len(image_files)} images with {max_workers} workers...")

        database = {}
        processed_count = 0
        error_count = 0

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use process_existing_card instead of process_single_card
            futures = [
                executor.submit(process_existing_card, img_path)
                for img_path in image_files
            ]

            # Collect results with progress tracking
            for i, future in enumerate(as_completed(futures)):
                result = future.result()

                if result:
                    database[result["key"]] = result["data"]
                    processed_count += 1
                else:
                    error_count += 1

                # Print progress every 100 images
                if (i + 1) % 100 == 0:
                    progress = (i + 1) / len(image_files) * 100
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(image_files) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"Progress: {i + 1}/{len(image_files)} ({progress:.1f}%) - "
                        f"{processed_count} successful, {error_count} failed - "
                        f"Rate: {rate:.1f} images/sec - ETA: {eta:.0f}s"
                    )

        total_time = time.time() - start_time

    else:
        # Original download logic (your existing code)
        print(
            f"Building card database with CLAHE (max {max_cards} cards, {max_workers} workers)..."
        )

        # Create directories
        images_dir = Path("./data/card_images")
        images_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Get bulk card data
        start_time = time.time()
        cards_data = fetch_bulk_cards(max_cards)

        if not cards_data:
            print("Failed to get bulk data, exiting...")
            return

        fetch_time = time.time() - start_time
        print(f"Data fetch completed in {fetch_time:.1f} seconds")

        # Process cards in parallel
        print(f"Processing {len(cards_data)} cards with {max_workers} workers...")

        # Prepare arguments
        args_list = [(card, images_dir) for card in cards_data]

        database = {}
        processed_count = 0
        error_count = 0

        process_start = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = [executor.submit(process_single_card, args) for args in args_list]

            # Collect results with progress tracking
            for i, future in enumerate(as_completed(futures)):
                result = future.result()

                if result:
                    database[result["key"]] = result["data"]
                    processed_count += 1
                else:
                    error_count += 1

                # Print progress every 50 cards
                if (i + 1) % 50 == 0:
                    progress = (i + 1) / len(cards_data) * 100
                    elapsed = time.time() - process_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(cards_data) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"Progress: {i + 1}/{len(cards_data)} ({progress:.1f}%) - "
                        f"{processed_count} successful, {error_count} failed - "
                        f"Rate: {rate:.1f} cards/sec - ETA: {eta:.0f}s"
                    )

        total_time = time.time() - start_time

    # Step 3: Save database (common for both paths)
    print(f"\nSaving database with {len(database)} cards...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    # Print final summary
    print(f"\n=== BUILD COMPLETE ===")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successfully processed: {processed_count} cards")
    print(f"Processing rate: {processed_count/total_time:.1f} cards/second")
    print(f"Database saved to: {output_path}")
    if use_existing:
        print(f"Processed existing images from: {card_images_dir}")
    else:
        print(f"Images saved to: {images_dir}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build MTG card database using bulk data and parallel processing"
    )
    parser.add_argument(
        "--existing",
        action="store_true",
        help="Process existing images instead of downloading",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=10000,
        help="Maximum number of cards to process (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default="./data/card_database_phash.json",
        help="Output database file (default: ./data/card_database_phash.json)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker threads (default: 8)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: process only 100 cards"
    )

    args = parser.parse_args()

    max_cards = args.max_cards
    max_workers = args.workers

    build_database(
        max_cards=max_cards,
        output_path=args.output,
        max_workers=args.workers,
        use_existing=args.existing,
    )


if __name__ == "__main__":
    main()

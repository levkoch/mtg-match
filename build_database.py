"""
MTG card database builder using Scryfall bulk data and parallel processing.
Downloads cards, extracts pHash + color histogram features with CLAHE, saves to JSON.
Optimized for processing 85K+ images (65GB+).
"""

from typing import Optional
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
from pydantic.dataclasses import dataclass
from functools import cached_property
import logging
import sys
from tqdm import tqdm

from config import BIN_VERSIONS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_database.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Card():
    name: str
    collector_num: str
    set_code: str
    image_url: str
    scryfall_url: str

    @cached_property
    def card_key(self) -> str:
        'normalized card key for filenames'
        card_key = self.scryfall_url.split("/card/")[1]
        card_key = card_key.split("?")[0]
        return card_key.replace("/", "_")
    
    def to_dict(self) -> dict[str, str]:
        return {
            'name': self.name,
            'collector_num': self.collector_num,
            'set_code': self.set_code,
            'image_url': self.image_url,
            'scryfall_url': self.scryfall_url,
            'card_key': self.card_key
        }

def fetch_bulk_cards(max_cards: Optional[int] = None) -> list[Card]:
    """Download bulk card data from Scryfall (much faster for large datasets)."""
    logger.info("Fetching bulk data info from Scryfall...")

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
                logger.info(f"Found bulk data: {file_size:.1f} MB")
                break

        if not default_cards_url:
            raise Exception("Could not find bulk card data URL")

        # Download the bulk file
        logger.info("Downloading bulk card data (this may take a few minutes)...")
        start_time = time.time()
        response = requests.get(default_cards_url, timeout=300)  # 5 min timeout
        response.raise_for_status()
        download_time = time.time() - start_time
        logger.info(f"Download completed in {download_time:.1f} seconds")

        # Parse JSON
        logger.info("Parsing card data...")
        cards_data = response.json()

        # Filter to cards with normal images and reasonable data
        logger.info("Filtering cards with images...")
        cards_with_images = []
        for card_dict in cards_data:
            try:
                # paper-only cards
                if "paper" not in card_dict.get('games', []): continue
                if card_dict.get("set_type") == "memorabilia": continue
                if card_dict.get('promo', False): continue
                if card_dict.get('lang') != 'en': continue
                if card_dict.get('image_status') == 'missing': continue
                if card_dict.get('set', "") == 'unk': continue

                if 'image_uris' not in card_dict:
                    # this is a double-faced card, so we will save the front side
                    if 'card_faces' not in card_dict or not card_dict['card_faces']:
                        continue
                    image_url = card_dict['card_faces'][0]['image_uris']['normal']
                else:
                    image_url = card_dict['image_uris']['normal']

                card = Card(
                    name = card_dict["name"],
                    collector_num = card_dict["collector_number"],
                    set_code = card_dict['set'],
                    image_url = image_url,
                    scryfall_url = card_dict["scryfall_uri"],
                )
                cards_with_images.append(card)
            except (KeyError, IndexError) as e:
                logger.debug(f'Issue with {card_dict.get("scryfall_uri", "unknown")}: {e}')

        logger.info(f"Found {len(cards_with_images)} cards with images")
        if max_cards:
            return cards_with_images[:max_cards]
        return cards_with_images

    except Exception as e:
        logger.error(f"Error downloading bulk data: {e}")
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
        for bin_name, bin_counts in BIN_VERSIONS[6:]: # only use bin_G
            hist = cv2.calcHist(
                [hsv], [0, 1, 2], None, bin_counts, [0, 180, 0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
            features[bin_name] = hist.flatten().tolist()

        return features

    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return {}


def process_single_card(args):
    """Process a single card - download image and extract features."""
    card: Card
    images_dir: Path
    card, images_dir = args

    try:
        # Check if image already exists
        image_filename = f"{card.card_key}.png"
        image_path = images_dir / image_filename
        
        if image_path.exists():
            # Load existing image
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                logger.warning(f"Could not load existing image: {image_path}")
                return None
        else:
            # Download and save image
            response = requests.get(card.image_url, timeout=15)
            response.raise_for_status()

            # Load and convert image
            pil_image = Image.open(BytesIO(response.content))
            image_array = np.array(pil_image)

            if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                logger.warning(f"Invalid image format for {card.name}")
                return None  # Skip invalid images

            # Save image
            success = cv2.imwrite(str(image_path), image_bgr)
            if not success:
                logger.warning(f"Failed to save image: {image_path}")
                return None

        # Extract features
        features = extract_card_features(image_bgr)
        if not features:
            return None

        return {
            "key": card.card_key,
            "data": {**card.__dict__, **features}
        }

    except Exception as e:
        logger.debug(f"Error processing card {card.name}: {e}")
        return None


def save_checkpoint(database: dict, checkpoint_path: Path, batch_num: int):
    """Save intermediate checkpoint of database."""
    checkpoint_file = checkpoint_path / f"checkpoint_{batch_num}.json"
    try:
        with open(checkpoint_file, "w") as f:
            json.dump(database, f, indent=2)
        logger.info(f"Checkpoint saved: {checkpoint_file} ({len(database)} cards)")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load the most recent checkpoint."""
    checkpoints = sorted(checkpoint_path.glob("checkpoint_*.json"))
    if not checkpoints:
        return {}
    
    latest_checkpoint = checkpoints[-1]
    logger.info(f"Loading checkpoint: {latest_checkpoint}")
    try:
        with open(latest_checkpoint, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return {}


def build_database(
    max_cards: Optional[int] = None,
    output_path: str = "./data/card_database_phash.json",
    max_workers: int = 16,
    card_images_dir: str = "./data/card_images",
    batch_size: int = 1000,
    resume: bool = False,
):
    """Build card database using bulk data and parallel processing."""

    checkpoint_path = Path("./data/checkpoints")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint if resuming
    database = {}
    if resume:
        database = load_checkpoint(checkpoint_path)
        logger.info(f"Resumed with {len(database)} cards already processed")

    # Original download logic
    logger.info(f"Building card database with CLAHE (max {max_cards or 'all'} cards, "
                f"{max_workers} workers)...")

    # Create directories
    images_dir = Path("./data/card_images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get bulk card data
    start_time = time.time()
    cards_data: list[Card] = fetch_bulk_cards(max_cards)

    if not cards_data:
        logger.error("Failed to get bulk data, exiting...")
        return

    fetch_time = time.time() - start_time
    logger.info(f"Data fetch completed in {fetch_time:.1f} seconds")

    # Filter out already processed cards if resuming
    if resume:
        processed_keys = set(database.keys())
        cards_data = [card for card in cards_data if card.card_key not in processed_keys]
        logger.info(f"Remaining to process: {len(cards_data)} cards")

    # Process cards in batches
    logger.info(f"Processing {len(cards_data)} cards with {max_workers} workers...")

    processed_count = len(database)
    error_count = 0
    process_start = time.time()

    # Process in batches
    for batch_num, batch_start in enumerate(range(0, len(cards_data), batch_size)):
        batch_end = min(batch_start + batch_size, len(cards_data))
        batch = cards_data[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_num + 1} ({batch_start} to {batch_end})...")

        args_list = [(card, images_dir) for card in batch]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_card, args) for args in args_list]

            for future in tqdm(as_completed(futures), total=len(futures),
                                desc=f"Batch {batch_num + 1}"):
                result = future.result()

                if result:
                    database[result["key"]] = result["data"]
                    processed_count += 1
                else:
                    error_count += 1

        # Save checkpoint after each batch
        save_checkpoint(database, checkpoint_path, batch_num)
        
        # Log progress
        elapsed = time.time() - process_start
        rate = processed_count / elapsed if elapsed > 0 else 0
        logger.info(f"Batch {batch_num + 1} complete: {processed_count} total processed, "
                    f"{error_count} failed, {rate:.1f} cards/sec")

    total_time = time.time() - start_time

    # Save final database
    logger.info(f"\nSaving final database with {len(database)} cards...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    # Print final summary
    logger.info(f"\n=== BUILD COMPLETE ===")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Successfully processed: {processed_count} cards")
    logger.info(f"Failed: {error_count} cards")
    logger.info(f"Processing rate: {processed_count/total_time:.1f} cards/second")
    logger.info(f"Database saved to: {output_path}")
    logger.info(f"Images saved to: {images_dir}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build MTG card database using bulk data and parallel processing"
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Maximum number of cards to process (default: all)",
    )
    parser.add_argument(
        "--output",
        default="./data/card_database_phash.json",
        help="Output database file (default: ./data/card_database_phash.json)",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of worker threads (default: 16)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000, 
        help="Batch size for checkpointing (default: 5000)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )

    args = parser.parse_args()

    build_database(
        max_cards=args.max_cards,
        output_path=args.output,
        max_workers=args.workers,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
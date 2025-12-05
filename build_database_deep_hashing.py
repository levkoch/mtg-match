"""
MTG card database builder using deep learning embeddings.
Downloads cards, extracts deep hash features using ResNet, saves to JSON.
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
import torch
import torchvision.models as models
import torchvision.transforms as transforms


class DeepHashExtractor:
    """Extract deep learning embeddings for card images."""

    def __init__(self, model_name="resnet50"):
        """
        Initialize the deep hash extractor.
        Uses a pre-trained ResNet model and extracts features from the penultimate layer.
        """
        print(f"Loading {model_name} model...")

        # Load pre-trained ResNet
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove the final classification layer to get embeddings
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        # Image preprocessing pipeline (standard ImageNet normalization)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep embedding from card image.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Normalized embedding vector
        """
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocess image
            input_tensor = self.transform(image_rgb)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                embedding = self.model(input_batch)

            # Flatten and convert to numpy
            embedding = embedding.squeeze().cpu().numpy()

            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None


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


def process_single_card(args):
    """Process a single card - download image and extract deep hash."""
    card_data, images_dir, extractor = args

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

        # Extract deep hash embedding
        embedding = extractor.extract_embedding(image_bgr)
        if embedding is None:
            return None

        return {
            "key": card_key,
            "data": {
                "name": card_name,
                "set": set_code,
                "image_url": image_url,
                "image_file": str(image_path),
                "deep_hash": embedding.tolist(),  # Convert numpy array to list for JSON
            },
        }

    except Exception as e:
        return None


def build_database(
    max_cards: int = 10000,
    output_path: str = "./data/card_database_deep_hash.json",
    max_workers: int = 4,  # Lower default for GPU processing
    model_name: str = "resnet50",
):
    """Build card database using deep learning embeddings."""

    print(f"Building card database with deep hashing (max {max_cards} cards)...")
    print(f"Model: {model_name}, Workers: {max_workers}")

    # Create directories
    images_dir = Path("./data/card_images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize deep hash extractor
    extractor = DeepHashExtractor(model_name=model_name)

    # Step 1: Get bulk card data
    start_time = time.time()
    cards_data = fetch_bulk_cards(max_cards)

    if not cards_data:
        print("Failed to get bulk data, exiting...")
        return

    fetch_time = time.time() - start_time
    print(f"Data fetch completed in {fetch_time:.1f} seconds")

    # Step 2: Process cards in parallel
    print(f"Processing {len(cards_data)} cards with {max_workers} workers...")
    print("Extracting deep embeddings using pre-trained ResNet...")

    # Prepare arguments
    args_list = [(card, images_dir, extractor) for card in cards_data]

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

            # Print progress every 25 cards (more frequent for slower processing)
            if (i + 1) % 25 == 0:
                progress = (i + 1) / len(cards_data) * 100
                elapsed = time.time() - process_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(cards_data) - i - 1) / rate if rate > 0 else 0
                print(
                    f"Progress: {i + 1}/{len(cards_data)} ({progress:.1f}%) - "
                    f"{processed_count} successful, {error_count} failed - "
                    f"Rate: {rate:.1f} cards/sec - ETA: {eta:.0f}s"
                )

    process_time = time.time() - process_start

    # Step 3: Save database
    print(f"\nSaving database with {len(database)} cards...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    # Print final summary
    total_time = time.time() - start_time
    print(f"\n=== BUILD COMPLETE ===")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Data fetch: {fetch_time:.1f}s, Processing: {process_time:.1f}s")
    print(f"Successfully processed: {processed_count}/{len(cards_data)} cards")
    print(
        f"Error rate: {error_count}/{len(cards_data)} ({error_count/len(cards_data)*100:.1f}%)"
    )
    print(f"Processing rate: {processed_count/process_time:.1f} cards/second")
    print(f"Database saved to: {output_path}")
    print(f"Images saved to: {images_dir}")
    print(f"Model: {model_name}")
    embedding_size = len(list(database.values())[0]["deep_hash"]) if database else 0
    print(f"Embedding size: {embedding_size} dimensions")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build MTG card database using deep learning embeddings"
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=10000,
        help="Maximum number of cards to process (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default="./data/card_database_deep_hash.json",
        help="Output database file (default: ./data/card_database_deep_hash.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4, lower for GPU)",
    )
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet50", "resnet18"],
        help="Model to use for embeddings (default: resnet50)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: process only 100 cards"
    )

    args = parser.parse_args()

    max_cards = 100 if args.test else args.max_cards

    build_database(max_cards, args.output, args.workers, args.model)


if __name__ == "__main__":
    main()

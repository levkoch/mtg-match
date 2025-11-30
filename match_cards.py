import cv2
import numpy as np
import json
from pathlib import Path
from typing import Any, Optional, Tuple
import csv

from extract import detect_card_edges_with_border, detect_card_edges_with_sides
from config import BIN_VERSIONS
from test_extract import compute_polygon_loss

from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_image(args):
    """Worker function for processing a single image."""
    img_path, database_data, bin_version, metadata, corners_dict = args

    # Create matcher instance in worker process
    matcher = CardMatcher(bin_version=bin_version)
    matcher.database = database_data

    try:
        # Process the image
        card, normalized, corners, top_matches = matcher.process_image(
            str(img_path), display=False
        )

        detected = False
        matched = card is not None

        # Check against ground truth
        image_key = img_path.stem
        truth_group = metadata.get(image_key, {})
        ground_truth = (truth_group.get("name", None), truth_group.get("set", None))
        is_correct = (card == ground_truth) if card else False

        # Check if correct answer is in top 5
        correct_in_top5 = (
            any(match["card"] == ground_truth for match in top_matches)
            if ground_truth[0] is not None and top_matches
            else False
        )

        # Good detection logic
        gt_corners = corners_dict.get(image_key)
        if corners is not None and gt_corners is not None:
            loss, _ = compute_polygon_loss(corners, gt_corners)
            detected = loss <= 20

        return {
            "image": img_path.name,
            "image_key": image_key,
            "detected": detected,
            "matched": matched,
            "card": card,
            "ground_truth": ground_truth,
            "correct": is_correct,
            "correct_in_top5": correct_in_top5,
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return {
            "image": img_path.name,
            "image_key": img_path.stem,
            "detected": False,
            "matched": False,
            "card": None,
            "ground_truth": (None, None),
            "correct": False,
            "correct_in_top5": False,
        }


class HistogramMatcher:
    def __init__(self, cards: list[tuple[str, str]], hists: list[list[float]]):
        self.cards = cards
        self.db_hists = np.array(hists, dtype=np.float32)

    def find_matches(self, query_hist, top_k=5):
        """
        Find top k matching cards
        EXPECTS ALL VECTORS TO BE NORMALIZED
        (or else it won't be cosine similarity any longer)
        """

        scores = np.dot(self.db_hists, np.array(query_hist).flatten())
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [{"card": self.cards[i], "score": scores[i]} for i in top_indices]


class CardMatcher:
    """Complete card detection and matching pipeline with perceptual hashing."""

    def __init__(
        self,
        database_path: str = "./data/card_database.json",
        bin_version: str = "bin_A",
    ):
        self.database_path = database_path
        self.database: dict[str, dict[str, Any]] = {}
        self.bin_name: str = bin_version
        self.bin_counts: tuple[int, int, int] = dict(BIN_VERSIONS).get(
            bin_version, (8, 8, 8)
        )
        self.matcher = None

    def load_database(self):
        """Load card database from JSON file."""
        if not Path(self.database_path).exists():
            print(f"Database not found: {self.database_path}")
            return

        with open(self.database_path, "r") as f:
            self.database = json.load(f)
        print(f"Loaded {len(self.database)} cards from database")

    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        corners = corners.reshape(4, 2)

        # Sort by y-coordinate to get top and bottom pairs
        sorted_by_y = corners[corners[:, 1].argsort()]
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]

        # Sort each pair by x-coordinate
        top_left, top_right = top_points[top_points[:, 0].argsort()]
        bottom_left, bottom_right = bottom_points[bottom_points[:, 0].argsort()]

        return np.array([top_left, top_right, bottom_right, bottom_left])

    def normalize_card(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: Tuple[int, int] = (
            488,
            680,
        ),  # card images in database are 488x680 pixels
    ) -> np.ndarray:
        """Apply perspective correction to extract normalized card."""
        corners = self.order_corners(corners).astype(np.float32)

        w, h = output_size
        target_corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(corners, target_corners)
        normalized = cv2.warpPerspective(image, M, output_size)

        return normalized

    def detect_and_extract_card(
        self, image_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect card in image and extract normalized version.
        Returns the normalized card and the corner points we found
        """

        corners, debug_img = detect_card_edges_with_border(
            image_path, display=False, show_steps=False
        )

        if corners is None:
            # print("  Border detection failed, trying edge detection...")
            corners, debug_img = detect_card_edges_with_sides(
                image_path, display=False, show_steps=False
            )

        if corners is None:
            # print("Card detection failed")
            return None, None

        # Load original image and normalize
        original_image = cv2.imread(image_path)
        if original_image is None:
            # print("Could not load image")
            return None, None

        try:
            corners_array = np.array(corners)
            normalized_card = self.normalize_card(original_image, corners_array)
            return normalized_card, corners_array
        except Exception as e:
            # print(f"  Normalization failed: {e}")
            return None, None

    def preprocess_card_image(self, card_image: np.ndarray) -> np.ndarray:
        # TODO: something we should investigate
        """Apply preprocessing to reduce glare and improve consistency."""

        # These steps are the same used in building the database
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(card_image, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel to reduce glare effects
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (3, 3), 0)

        return processed

    def compute_perceptual_hash(
        self, card_image: np.ndarray, hash_size: int = 16
    ) -> str:
        """Compute perceptual hash of card image."""
        # Preprocess the image
        processed = self.preprocess_card_image(card_image)

        # Resize to small fixed size for hashing
        small = cv2.resize(
            processed, (hash_size, hash_size), interpolation=cv2.INTER_AREA
        )

        # Convert to grayscale for hash computation
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Compute average pixel value
        avg = np.mean(gray)

        # Create binary hash based on whether each pixel is above/below average
        binary_hash = (gray > avg).flatten()

        # Convert to hex string
        hash_bytes = np.packbits(binary_hash)
        return "".join(f"{byte:02x}" for byte in hash_bytes)

    def compute_color_histogram(self, card_image: np.ndarray) -> np.ndarray:
        """Compute color histogram for additional matching."""
        # Use HSV for better color representation
        # HSV separates color information from brightness
        # H = Hue (actual color: 0-180°)
        # S = Saturation (color intensity: 0-255)
        # V = Value (brightness: 0-255)
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)

        # Compute 3D histogram
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, self.bin_counts, [0, 180, 0, 256, 0, 256]
        )

        # Normalize
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)

        return hist.flatten()

    def hamming_distance(self, hash1: str, hash2: str) -> float:
        """Calculate normalized Hamming distance between two hashes."""
        if len(hash1) != len(hash2):
            return 1.0

        # Convert hex strings to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

        # Calculate hamming distance
        distance = sum(b1 != b2 for b1, b2 in zip(bin1, bin2))

        # Normalize by total bits
        return distance / len(bin1)

    def match_perceptual_hash(
        self, card_image: np.ndarray, top_k: int = 5
    ) -> Optional[tuple[str, str]]:
        """Match card using perceptual hash only."""
        if len(self.database) == 0:
            return None

        # Extract features from query image
        query_hash = self.compute_perceptual_hash(card_image)

        best_match = None
        best_score = float("inf")
        matches_found = 0

        all_scores = []

        for card_key, card_data in self.database.items():
            try:
                # Skip cards without required features
                if "perceptual_hash" not in card_data:
                    continue

                # Calculate hash distance
                db_hash = card_data["perceptual_hash"]
                hash_distance = self.hamming_distance(query_hash, db_hash)

                all_scores.append(
                    {
                        "card": (card_data["name"], card_data["set"]),
                        "hash_distance": hash_distance,
                        "score": 1.0 - hash_distance,
                    }
                )
            except Exception as e:
                continue

        # Sort by hash distance (lower is better)
        all_scores.sort(key=lambda x: x["hash_distance"])

        return all_scores[:top_k]

    def match_histogram(
        self, card_image: np.ndarray, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Match card using color histogram."""
        if self.matcher is None:
            # Prepare matcher
            names = []
            hists = []
            for card_data in self.database.values():
                if self.bin_name in card_data:
                    names.append((card_data["name"], card_data["set"]))
                    hists.append(card_data[self.bin_name])
            self.matcher = HistogramMatcher(names, hists)

        query_hist = self.compute_color_histogram(card_image)
        matches = self.matcher.find_matches(query_hist, top_k=top_k)

        return matches

    def process_image(
        self, image_path: str, display: bool = True, match_func="perceptual"
    ) -> Tuple[Optional[tuple[str, str]], Optional[np.ndarray]]:
        """Complete pipeline: detect → normalize → match."""

        # Detection and normalization
        normalized_card, corners = self.detect_and_extract_card(image_path)

        if normalized_card is None:
            return None, None, None, None

        if match_func == "histogram":
            matches = self.match_histogram(normalized_card, top_k=5)
            card = matches[0]["card"] if matches else None
            top_matches = matches
        else:  # perceptual
            top_matches = self.match_perceptual_hash(normalized_card, top_k=5)
            card = top_matches[0]["card"] if top_matches else None

        # Display results
        if display:
            self.display_results(image_path, normalized_card, corners, card, 0.0)

        return card, normalized_card, corners, top_matches

    def display_results(
        self,
        image_path: str,
        normalized_card: np.ndarray,
        corners: np.ndarray,
        card_name: Optional[tuple[str, str]],
        score: float,
    ):
        """Display detection and matching results."""
        import matplotlib.pyplot as plt

        # Load original image
        original = cv2.imread(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original with detection
        img_with_corners = original.copy()
        if corners is not None:
            cv2.drawContours(img_with_corners, [corners.astype(int)], 0, (0, 255, 0), 3)
            for point in corners:
                cv2.circle(
                    img_with_corners, tuple(point.astype(int)), 8, (0, 0, 255), -1
                )

        axes[0].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Card Detection")
        axes[0].axis("off")

        # Normalized with match result
        axes[1].imshow(cv2.cvtColor(normalized_card, cv2.COLOR_BGR2RGB))
        title = f'Normalized Card\n{card_name or "No Match"}\n(Score: {score:.3f})'
        axes[1].set_title(title)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def get_card_info(self, card_name: str) -> dict:
        """Get full card info by name."""
        for card_data in self.database.values():
            if card_data["name"] == card_name:
                return card_data
        return {}


def main_histograms():
    bin_results = {}

    for bins_name, bin_counts in BIN_VERSIONS:
        print(
            f"\nTesting histogram matching with bin version: {bins_name} ({bin_counts} bins)"
        )
        matcher = CardMatcher(bin_version=bins_name)

        if len(matcher.database) == 0:
            print("No database loaded!")
            return

        # Get test images in proper order
        generations_dir = Path("./data/generations")
        test_images = list(generations_dir.glob("*.png"))

        if not test_images:
            print("No test images found in ./data/generations/")
            print("Run create_test_images.py first!")
            return

        print(f"Testing histogram pipeline on {len(test_images)} generated images...")

        results = []

        # Load generations metadata for ground truth
        metadata_path = Path("./data/generations.json")
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        corners_dict = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                for k, v in metadata.items():
                    if "corners" in v:
                        corners_dict[k] = np.array(v["corners"], dtype=np.float32)

        for img_path in test_images:
            card, normalized, corners = matcher.process_image(
                str(img_path), display=False, match_func="histogram"
            )
            detected = False
            matched = card is not None

            # Check against ground truth
            image_key = img_path.stem
            truth_group = metadata.get(image_key, {})
            ground_truth = (truth_group.get("name", None), truth_group.get("set", None))
            is_correct = (card == ground_truth) if card else False

            # Good detection logic
            gt_corners = corners_dict.get(image_key)
            if corners is not None and gt_corners is not None:
                loss, _ = compute_polygon_loss(corners, gt_corners)
                detected = loss <= 20  # Only count as detected if "good"

            results.append(
                {
                    "image": img_path.name,
                    "image_key": image_key,
                    "detected": detected,
                    "matched": card is not None,
                    "card": card,
                    "ground_truth": ground_truth,
                    "correct": is_correct,
                }
            )

        # Print summary
        print(f"\nSUMMARY for bin version {bins_name} ({bin_counts}):")
        detected = sum(1 for r in results if r["detected"])
        matched = sum(1 for r in results if r["matched"])
        correct = sum(1 for r in results if r["correct"])

        bin_results[bins_name] = {
            "bins_name": bins_name,
            "bin_counts": bin_counts,
            "detected": detected,
            "matched": matched,
            "correct": correct,
            "total": len(results),
        }

        print(
            f"  Detection: {detected}/{len(results)} successful ({detected/len(results)*100:.1f}%)"
        )
        print(
            f"  Accurately Matched: {correct}/{detected} correct ({correct/detected*100:.1f}%)"
        )
        print(
            f"  Overall Accuracy: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)"
        )

    with open("./data/histogram_bin_results.json", "w") as f:
        json.dump(bin_results, f, indent=4)


def main_perceptual():
    """Test the complete pipeline on generated test images."""
    matcher = CardMatcher()
    matcher.load_database()
    if len(matcher.database) == 0:
        print("No database loaded!")
        return
    database_data = matcher.database

    # Get test images
    generations_dir = Path("./data/generations")
    test_images = list(generations_dir.glob("*.png"))
    test_images.sort()

    if not test_images:
        print("No test images found in ./data/generations/")
        print("Run create_test_images.py first!")
        return

    print(f"Testing perceptual hash pipeline on {len(test_images)} generated images...")

    # Load generations metadata for ground truth
    metadata_path = Path("./data/generations.json")
    metadata = {}

    corners_dict = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            for k, v in metadata.items():
                if "corners" in v:
                    corners_dict[k] = np.array(v["corners"], dtype=np.float32)
    else:
        print("Warning: No generations.json found for ground truth comparison")

    # Prepare arguments for each worker
    database_path = "./data/card_database.json"
    bin_version = "bin_A"

    args_list = [
        (img_path, database_data, bin_version, metadata, corners_dict)
        for img_path in test_images
    ]

    # Process in parallel
    results = []
    total_images = len(args_list)

    # Use max_workers to control parallelism
    max_workers = min(6, total_images)

    print(f"Using {max_workers} worker processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, args) for args in args_list]

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            # Print progress every 100 images
            if (i + 1) % 100 == 0 or (i + 1) == total_images:
                print(
                    f"Processed {i + 1}/{total_images} images ({(i + 1)/total_images*100:.1f}%)"
                )

    # Sort results by image name to maintain order
    results.sort(key=lambda x: x["image"])

    # Print summary
    print(f"\nSUMMARY:")
    detected = sum(1 for r in results if r["detected"])
    matched = sum(1 for r in results if r["matched"])
    correct = sum(1 for r in results if r["correct"])
    correct_in_top5_count = sum(1 for r in results if r["correct_in_top5"])

    print(
        f"  Detection: {detected}/{len(results)} successful ({detected/len(results)*100:.1f}%)"
    )
    print(
        f"  Accuracy (on detected cards): {correct}/{detected} correct ({correct/detected*100:.1f}%)"
    )
    print(
        f"  Correct in Top-5 (in detected cards): {correct_in_top5_count}/{detected} ({correct_in_top5_count/detected*100:.1f}%)"
    )
    print(
        f"  Overall accuracy: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)"
    )

    results_file = Path("./data/match_results.csv")
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "predicted",
                "ground_truth",
                "correct",
                "detected",
                "matched",
                "correct_in_top5",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["image"],
                    result["card"][0] if result["card"] else "No match",
                    (
                        result["ground_truth"][0]
                        if result["ground_truth"][0]
                        else "Unknown"
                    ),
                    int(result["correct"]),
                    int(result["detected"]),
                    int(result["matched"]),
                    int(result["correct_in_top5"]),
                ]
            )

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main_perceptual()

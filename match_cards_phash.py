import cv2
import numpy as np
import json
from pathlib import Path
from typing import Any, Literal, Optional, Tuple
import csv

from extract import detect_card_edges_with_border, detect_card_edges_with_sides
from config import BIN_VERSIONS
from test_extract import compute_polygon_loss
from build_database import Card

class HistogramMatcher:
    def __init__(self, cards: list[Card], hists: list[list[float]]):
        if len(cards) != len(hists): 
            raise ValueError('each card must have a matching histogram')
        self.cards = cards
        self.db_hists = np.array(hists, dtype=np.float32)
        self.card_to_idx = {card.card_key: i for i, card in enumerate(cards)}

    def find_matches(self, query_hist, top_k=5) -> list[dict[str, Card | float]]:
        """
        Find top k matching cards
        EXPECTS ALL VECTORS TO BE NORMALIZED
        (or else it won't be cosine similarity any longer)
        """

        scores = np.dot(self.db_hists, np.array(query_hist).flatten())
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [{"card": self.cards[i], "score": scores[i]} for i in top_indices]

    def rerank_candidates(
        self, query_hist: np.ndarray, candidates: list[dict]
    ) -> list[dict]:
        """
        Re-rank pHash candidates using histogram similarity.

        Args:
            query_hist: Normalized histogram of query image
            candidates: List of dicts with 'card' and 'score' keys from pHash matching

        Returns:
            Re-ranked list with combined scores
        """
        query_hist = query_hist.flatten().astype(np.float32)

        for candidate in candidates:
            card: Card = candidate["card"]
            idx = self.card_to_idx.get(card.card_key)
            if idx is not None:
                hist_score = float(np.dot(self.db_hists[idx], query_hist))
            else:
                hist_score = 0.0

            # Combine: pHash score and histogram score
            phash_score = candidate.get("score", 1 - candidate.get("hash_distance", 0))
            candidate["hist_score"] = hist_score
            candidate["combined_score"] = 0.6 * phash_score + 0.4 * hist_score

        # Sort by combined score
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates


class CardMatcher:
    """Complete card detection and matching pipeline with perceptual hashing."""

    def __init__(
        self,
        database_path: str = "./data/card_database_phash.json",
        bin_version: str = "bin_G", # default to our best bin G
    ):
        self.database_path = database_path
        self.database: dict[str, Card] = {}
        self.card_keys: list[str] = []
    
        self.bin_name: str = bin_version
        self.bin_counts: tuple[int, int, int] = dict(BIN_VERSIONS).get(
            bin_version, (16, 4, 4)
        )

        self.matcher = None
        self.db_hashes = []
        self.db_hists = []

        self.db_hash_bits = None
        self.db_hist_bits = None

        self.load_database()

    def load_database(self):
        """Load card database from JSON file and convert to Card objects."""
        if not Path(self.database_path).exists():
            print(f"Database not found: {self.database_path}")
            return

        with open(self.database_path, "r") as f:
            raw_data = json.load(f)

        # Convert raw JSON data to Card objects
        for card_key, card_data in raw_data.items():
            card = Card(
                name=card_data["name"],
                collector_num=card_data["collector_num"],
                set_code=card_data["set_code"],
                image_url=card_data["image_url"],
                scryfall_url=card_data["scryfall_url"]
            )

            self.card_keys.append(card_key)
            self.database[card_key] = card

            self.db_hashes.append(card_data['perceptual_hash'])
            self.db_hists.append(card_data[self.bin_name])

        self.db_hash_bits = np.array([
            np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in self.db_hashes
        ])

        self.db_hist_bits = np.array(self.db_hists, dtype=np.float32)
            
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
            corners, debug_img = detect_card_edges_with_sides(image_path)

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

    def hamming_distance_vectorized(self, query_hash: str) -> np.ndarray:
        """Calculate Hamming distance using numpy unpackbits."""
        if len(self.db_hashes) == 0:
            return np.array([])
        
        # Convert hex to bit arrays
        query_bytes = np.frombuffer(bytes.fromhex(query_hash), dtype=np.uint8)
        query_bits = np.unpackbits(query_bytes)
    
        # XOR and count differences
        differences = np.sum(self.db_hash_bits != query_bits, axis=1)
        
        # Normalize
        total_bits = len(query_bits)
        return differences / total_bits

    def match_perceptual_hash(
        self, card_image: np.ndarray, top_k: int = 5
    ) -> Optional[list[dict]]:
        """Match card using perceptual hash only - vectorized version with numpy top-k."""
        if len(self.database) == 0:
            return None

        # Extract features from query image
        query_hash = self.compute_perceptual_hash(card_image)

        # Vectorized distance computation for all hashes at once
        hash_distances = self.hamming_distance_vectorized(query_hash)
        
        if len(hash_distances) == 0:
            return None
        
        # Use numpy argpartition to find top-k smallest distances efficiently
        # This is O(n) instead of O(n log n) for full sort
        k = min(top_k, len(hash_distances))
        top_k_indices = np.argpartition(hash_distances, k-1)[:k]
        
        # Sort only the top-k results
        top_k_indices = top_k_indices[np.argsort(hash_distances[top_k_indices])]
        
        # Build results for top-k only - use card_keys list to get cards
        return [
            {
                "card": self.database[self.card_keys[idx]],
                "hash_distance": float(hash_distances[idx]),
                "score": 1.0 - float(hash_distances[idx]),
            }
            for idx in top_k_indices
        ]

    
    def match_histogram(
        self, card_image: np.ndarray, top_k: int = 5
    ) -> list[dict[str, Card | float]]:
        """Match card using color histogram."""
        if self.matcher is None:
            # Prepare matcher - use the cached data from load_database
            cards = [self.database[key] for key in self.card_keys]
            self.matcher = HistogramMatcher(cards, self.db_hists)

        query_hist = self.compute_color_histogram(card_image)
        return self.matcher.find_matches(query_hist, top_k=top_k)

    def process_image(
        self, image_path: str, display: bool = True, 
        match_func: Literal['perceptual', 'histogram', 'combined'] = "perceptual", 
        top_k: int = 5
    ) -> Tuple[Optional[Card], Optional[np.ndarray], Optional[np.ndarray], Optional[list[dict]]]:
        """Complete pipeline: detect → normalize → match.
        Returns:
        - card dataclass with information,
        - normalized detected card image,
        - corners of the detected card (within the overall image passed in)
        - """

        # Detection and normalization
        normalized_card, corners = self.detect_and_extract_card(image_path)

        if normalized_card is None:
            return None, None, None, None

        if match_func == "histogram":
            matches = self.match_histogram(normalized_card, top_k)
            card = matches[0]["card"] if matches else None
            top_matches = matches
        elif match_func == "combined":
            # Get top-5 from pHash, then re-rank with histogram
            top_matches = self.match_perceptual_hash(normalized_card, top_k)
            if top_matches and self.matcher is None:
                # Initialize histogram matcher if needed
                self.match_histogram(normalized_card, top_k=1)
            if top_matches and self.matcher:
                query_hist = self.compute_color_histogram(normalized_card)
                top_matches = self.matcher.rerank_candidates(query_hist, top_matches)
            card = top_matches[0]["card"] if top_matches else None
        else:  # perceptual
            top_matches = self.match_perceptual_hash(normalized_card, top_k)
            card = top_matches[0]["card"] if top_matches else None

        # Display results
        if display:
            self.display_results(image_path, normalized_card, corners, card, top_matches[0]['score'])

        return card, normalized_card, corners, top_matches

    def display_results(
        self,
        image_path: str,
        normalized_card: np.ndarray,
        corners: np.ndarray,
        card: Optional[Card],
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
        card_display = f"{card.name} ({card.set_code} # {card.collector_num})" if card else "No Match"
        title = f'Normalized Card\n{card_display}\n(Score: {score:.3f})'
        axes[1].set_title(title)
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    def get_card_info(self, card_name: str) -> Optional[Card]:
        """Get full card info by name."""
        for card in self.database.values():
            if card.name == card_name:
                return card
        return None

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
        test_images.sort()

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
            card, normalized, corners, top_matches = matcher.process_image(
                str(img_path), display=False, match_func="histogram"
            )

            detected = False
            matched = card is not None

            # Check against ground truth
            image_key = img_path.stem
            truth_group = metadata.get(image_key, {})
            ground_truth = (truth_group.get("name", None), truth_group.get("set", None))

            # Check if top match is correct (regardless of border detection)
            is_correct_overall = (card == ground_truth) if card else False

            # Check if correct answer is in top 5 (regardless of border detection)
            correct_in_top5 = (
                any(match["card"] == ground_truth for match in top_matches)
                if top_matches is not None and ground_truth[0] is not None
                else False
            )

            # Good detection logic
            gt_corners = corners_dict.get(image_key)
            if corners is not None and gt_corners is not None:
                loss, _ = compute_polygon_loss(corners, gt_corners)
                detected = loss <= 20  # Only count as detected if "good"

            # check if top match is correct (only if detected well)
            is_correct_detected = (card == ground_truth) if card and detected else False

            # Check if correct answer is in top 5 (only correctly detected cards)
            correct_in_top5_detected = (
                any(match["card"] == ground_truth for match in top_matches)
                if detected and top_matches is not None and ground_truth[0] is not None
                else False
            )

            results.append(
                {
                    "image": img_path.name,
                    "image_key": image_key,
                    "detected": detected,
                    "matched": card is not None,
                    "card": card,
                    "ground_truth": ground_truth,
                    "correct": is_correct_overall,
                    "correct_detected": is_correct_detected,
                    "correct_in_top5": correct_in_top5,
                    "correct_in_top5_detected": correct_in_top5_detected,
                }
            )

        # Print summary
        print(f"\nSUMMARY for bin version {bins_name} ({bin_counts}):")
        detected = sum(1 for r in results if r["detected"])
        matched = sum(1 for r in results if r["matched"])
        correct = sum(1 for r in results if r["correct"])
        correct_detected = sum(1 for r in results if r["correct_detected"])
        correct_in_top5_count = sum(1 for r in results if r["correct_in_top5"])
        correct_in_top5_count_detected = sum(
            1 for r in results if r["correct_in_top5_detected"]
        )

        bin_results[bins_name] = {
            "bins_name": bins_name,
            "bin_counts": bin_counts,
            "detected": detected,
            "matched": matched,
            "correct": correct,
            "correct_detected": correct_detected,
            "correct_in_top5": correct_in_top5_count,
            "correct_in_top5_detected": correct_in_top5_count_detected,
            "total": len(results),
        }

        print(
            f"  Detection: {detected}/{len(results)} successful ({detected/len(results)*100:.1f}%)"
        )
        print(
            f"  Accuracy (on detected cards): {correct_detected}/{detected} correct ({correct_detected/detected*100:.1f}%)"
        )
        print(
            f"  Correct in Top-5 (in detected cards): {correct_in_top5_count_detected}/{detected} ({correct_in_top5_count_detected/detected*100:.1f}%)"
        )
        print(
            f"  Correct in Top-5 (overall): {correct_in_top5_count}/{len(results)} ({correct_in_top5_count/len(results)*100:.1f}%)"
        )
        print(
            f"  Overall accuracy: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)"
        )

    with open("./data/histogram_bin_results.json", "w") as f:
        json.dump(bin_results, f, indent=4)


def main_match_func(match_func: Literal['perceptual', 'histogram', 'combined'] = "perceptual"):
    """Test the complete pipeline on generated test images."""
    matcher = CardMatcher()

    if len(matcher.database) == 0:
        print("No database loaded!")
        return

    # Get test images in proper order
    generations_dir = Path("./data/generations")
    test_images = list(generations_dir.glob("*.png"))
    test_images.sort()

    if not test_images:
        print("No test images found in ./data/generations/")
        print("Run create_test_images.py first!")
        return

    print(f"Testing {match_func} hash pipeline on {len(test_images)} generated images...")

    # Load generations metadata for ground truth
    metadata_path = Path("./data/generations.json")
    corners_dict = {} # file name to corners
    true_keys = {} # file name to card key
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            for k, v in metadata.items():
                corners_dict[k] = np.array(v["corners"], dtype=np.float32)
                true_keys[k] = v['card_key']
    else:
        print("Warning: No generations.json found for ground truth comparison")

    results = []
    for img_path in test_images:
        card, normalized, corners, top_matches = matcher.process_image(
            str(img_path), display=False, match_func=match_func, top_k=10
        )
        
        true_key = true_keys[img_path.stem]
        
        # Check if detection is good (corners within threshold)
        detected = False
        if corners is not None and true_key in corners_dict:
            loss, _ = compute_polygon_loss(corners, corners_dict[img_path.stem], method="mean_distance")
            detected = loss <= 20

        # Check correctness
        correct = card is not None and card.card_key == true_key
        correct_in_top5 = any(m["card"].card_key == true_key for m in top_matches[:5]) if top_matches else False
        correct_in_top10 = any(m["card"].card_key == true_key for m in top_matches) if top_matches else False
        
        results.append({
            "image": img_path.name,
            "true_key": true_key,
            "detected": detected,
            "matched": int(card is not None),
            "card": card,
            "correct": int(correct),
            "correct_in_top5": int(correct_in_top5),
            "correct_in_top10": int(correct_in_top10),
        })

    # Print summary
    total = len(results)
    detected_count = sum(r["detected"] for r in results)
    correct_count = sum(r["correct"] for r in results)
    correct_in_top5_count = sum(r["correct_in_top5"] for r in results)
    correct_in_top10_count = sum(r["correct_in_top10"] for r in results)
    
    # Stats on detected cards only
    detected_results = [r for r in results if r["detected"]]
    correct_detected_count = sum(r["correct"] for r in detected_results)
    correct_in_top5_detected_count = sum(r["correct_in_top5"] for r in detected_results)

    print(f"\nSUMMARY:")
    print(f"  Detection: {detected_count}/{total} successful ({detected_count/total*100:.1f}%)")
    if detected_count > 0:
        print(f"  Accuracy (on detected cards): {correct_detected_count}/{detected_count} correct ({correct_detected_count/detected_count*100:.1f}%)")
        print(f"  Top-5 (on detected cards): {correct_in_top5_detected_count}/{detected_count} ({correct_in_top5_detected_count/detected_count*100:.1f}%)")
    print(f"  Overall accuracy: {correct_count}/{total} correct ({correct_count/total*100:.1f}%)")
    print(f"  Overall Top-5:    {correct_in_top5_count}/{total} ({correct_in_top5_count/total*100:.1f}%)")
    print(f"  Overall Top-10:   {correct_in_top10_count}/{total} ({correct_in_top10_count/total*100:.1f}%)")

    # Write results to CSV
    results_file = Path(f"./data/match_results_{match_func}.csv")
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted", "ground_truth", "correct", "detected", "matched"])
        for r in results:
            writer.writerow([
                r["image"],
                r["card"].card_key if r["card"] else "No match",
                r["true_key"],
                int(r["correct"]),
                int(r["detected"]),
                int(r["matched"]),
            ])


def main_combined():
    """Test combined pHash + histogram reranking with different bin versions."""
    bin_results = {}

    for bins_name, bin_counts in BIN_VERSIONS:
        print(
            f"\nTesting combined matching with bin version: {bins_name} ({bin_counts} bins)"
        )
        matcher = CardMatcher(bin_version=bins_name)

        if len(matcher.database) == 0:
            print("No database loaded!")
            return

        # Get test images in proper order
        generations_dir = Path("./data/generations")
        test_images = list(generations_dir.glob("*.png"))
        test_images.sort()

        if not test_images:
            print("No test images found in ./data/generations/")
            return

        print(f"Testing combined pipeline on {len(test_images)} generated images...")

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
                        corners_dict[v['card_key']] = np.array(v["corners"], dtype=np.float32)

        for img_path in test_images:
            card, normalized, corners, top_matches = matcher.process_image(
                str(img_path), display=False, match_func="combined"
            )

            detected = False
            matched = card is not None

            # Check against ground truth
            image_key = img_path.stem
            truth_group = metadata.get(image_key, {})
            ground_truth = (truth_group.get("name", None), truth_group.get("set", None))

            is_correct_overall = (card == ground_truth) if card else False

            correct_in_top5 = (
                any(match["card"] == ground_truth for match in top_matches)
                if top_matches is not None and ground_truth[0] is not None
                else False
            )

            gt_corners = corners_dict.get(image_key)
            if corners is not None and gt_corners is not None:
                loss, _ = compute_polygon_loss(corners, gt_corners)
                detected = loss <= 20

            is_correct_detected = (card == ground_truth) if card and detected else False

            correct_in_top5_detected = (
                any(match["card"] == ground_truth for match in top_matches)
                if detected and top_matches is not None and ground_truth[0] is not None
                else False
            )

            results.append(
                {
                    "image": img_path.name,
                    "image_key": image_key,
                    "detected": detected,
                    "matched": card is not None,
                    "card": card,
                    "ground_truth": ground_truth,
                    "correct": is_correct_overall,
                    "correct_detected": is_correct_detected,
                    "correct_in_top5": correct_in_top5,
                    "correct_in_top5_detected": correct_in_top5_detected,
                }
            )

        # Print summary
        detected = sum(1 for r in results if r["detected"])
        correct = sum(1 for r in results if r["correct"])
        correct_detected = sum(1 for r in results if r["correct_detected"])
        correct_in_top5_count = sum(1 for r in results if r["correct_in_top5"])
        correct_in_top5_count_detected = sum(
            1 for r in results if r["correct_in_top5_detected"]
        )

        bin_results[bins_name] = {
            "bins_name": bins_name,
            "bin_counts": bin_counts,
            "detected": detected,
            "correct": correct,
            "correct_detected": correct_detected,
            "correct_in_top5": correct_in_top5_count,
            "correct_in_top5_detected": correct_in_top5_count_detected,
            "total": len(results),
        }

        print(f"\nSUMMARY for bin version {bins_name} ({bin_counts}):")
        print(
            f"  Detection: {detected}/{len(results)} successful ({detected/len(results)*100:.1f}%)"
        )
        print(
            f"  Accuracy (on detected cards): {correct_detected}/{detected} correct ({correct_detected/detected*100:.1f}%)"
        )
        print(
            f"  Correct in Top-5 (in detected cards): {correct_in_top5_count_detected}/{detected} ({correct_in_top5_count_detected/detected*100:.1f}%)"
        )
        print(
            f"  Correct in Top-5 (overall): {correct_in_top5_count}/{len(results)} ({correct_in_top5_count/len(results)*100:.1f}%)"
        )
        print(
            f"  Overall accuracy: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)"
        )

    with open("./data/combined_bin_results.json", "w") as f:
        json.dump(bin_results, f, indent=4)

    print("\nResults saved to ./data/combined_bin_results.json")


if __name__ == "__main__":
    main_match_func('histogram')
    print('')
    main_match_func('perceptual')
    print('')
    main_match_func('combined')
    # main_combined()

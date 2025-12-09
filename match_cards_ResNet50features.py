import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple
import csv
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from extract import detect_card_edges_with_border, detect_card_edges_with_sides
from test_extract import compute_polygon_loss


class DeepEmbeddingExtractor:
    """Extract deep learning embeddings for card images."""

    def __init__(self, model_name="resnet50"):
        """Initialize with pre-trained ResNet model."""
        print(f"Loading {model_name} model...")

        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Standard ImageNet preprocessing
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
        """Extract normalized embedding from BGR image."""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocess
            input_tensor = self.transform(image_rgb)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                embedding = self.model(input_batch)

            # Flatten and normalize
            embedding = embedding.squeeze().cpu().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None


class DeepEmbeddingMatcher:
    """Card matching using deep learning embeddings."""

    def __init__(
        self,
        database_path: str = "./data/card_database.json",
        model_name: str = "resnet50",
    ):
        self.database_path = database_path
        self.database = {}
        self.load_database(max_cards=9669)

        # Initialize feature extractor
        self.extractor = DeepEmbeddingExtractor(model_name=model_name)

        # Pre-compute database embeddings array for fast similarity search
        self.card_list = []
        self.embeddings = []
        for card_data in self.database.values():
            if "deep_Embedding" in card_data:
                self.card_list.append((card_data["name"], card_data["set"]))
                self.embeddings.append(card_data["deep_Embedding"])

        if self.embeddings:
            self.embeddings = np.array(self.embeddings, dtype=np.float32)
            print(f"Loaded {len(self.embeddings)} deep Embedding embeddings")

    def load_database(self, max_cards: Optional[int] = None):
        """Load card database from JSON file."""
        if not Path(self.database_path).exists():
            print(f"Database not found: {self.database_path}")
            return

        with open(self.database_path, "r") as f:
            full_database = json.load(f)

        if max_cards and max_cards < len(full_database):
            # Convert to list, slice, then back to dict
            items = list(full_database.items())[:max_cards]
            self.database = dict(items)
            print(
                f"Loaded first {len(self.database)} cards from database (limited from {len(full_database)})"
            )
        else:
            self.database = full_database
            print(f"Loaded {len(self.database)} cards from database")

    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left"""
        corners = corners.reshape(4, 2)
        sorted_by_y = corners[corners[:, 1].argsort()]
        top_points = sorted_by_y[:2]
        bottom_points = sorted_by_y[2:]
        top_left, top_right = top_points[top_points[:, 0].argsort()]
        bottom_left, bottom_right = bottom_points[bottom_points[:, 0].argsort()]
        return np.array([top_left, top_right, bottom_right, bottom_left])

    def normalize_card(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: Tuple[int, int] = (488, 680),
    ) -> np.ndarray:
        """Apply perspective correction to extract normalized card."""
        corners = self.order_corners(corners).astype(np.float32)
        w, h = output_size
        target_corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(corners, target_corners)
        normalized = cv2.warpPerspective(image, M, output_size)
        return normalized

    def detect_and_extract_card(
        self, image_path: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect card in image and extract normalized version."""
        corners, _ = detect_card_edges_with_border(
            image_path, display=False, show_steps=False
        )

        if corners is None:
            corners, _ = detect_card_edges_with_sides(image_path)

        if corners is None:
            return None, None

        original_image = cv2.imread(image_path)
        if original_image is None:
            return None, None

        try:
            corners_array = np.array(corners)
            normalized_card = self.normalize_card(original_image, corners_array)
            return normalized_card, corners_array
        except Exception:
            return None, None

    def match_card(self, card_image: np.ndarray, top_k: int = 5) -> list:
        """Match card using deep Embedding embeddings with cosine similarity."""
        if len(self.embeddings) == 0:
            return []

        # Compute query embedding
        query_embedding = self.extractor.extract_embedding(card_image)
        if query_embedding is None:
            return []

        query_embedding = query_embedding.flatten().astype(np.float32)

        # Compute cosine similarity (embeddings are already normalized)
        similarities = np.dot(self.embeddings, query_embedding)

        # Get top k matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        matches = []
        for idx in top_indices:
            matches.append(
                {"card": self.card_list[idx], "score": float(similarities[idx])}
            )

        return matches

    def process_image(
        self, image_path: str
    ) -> Tuple[
        Optional[tuple[str, str]], Optional[np.ndarray], Optional[np.ndarray], list
    ]:
        """Complete pipeline: detect → normalize → match."""

        # Detection and normalization
        normalized_card, corners = self.detect_and_extract_card(image_path)

        if normalized_card is None:
            return None, None, None, []

        # Match using deep Embedding
        top_matches = self.match_card(normalized_card, top_k=5)
        card = top_matches[0]["card"] if top_matches else None

        return card, normalized_card, corners, top_matches


def main():
    """Test deep Embedding matching on generated test images."""
    matcher = DeepEmbeddingMatcher(model_name="resnet50")

    if len(matcher.database) == 0:
        print("No database loaded!")
        return

    # Get test images
    generations_dir = Path("./data/generations")
    test_images = list(generations_dir.glob("*.png"))
    test_images.sort()

    if not test_images:
        print("No test images found in ./data/generations/")
        return

    print(f"Testing deep Embedding pipeline on {len(test_images)} generated images...")

    # Load metadata for ground truth
    metadata_path = Path("./data/generations.json")
    metadata = {}
    corners_dict = {}

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            for k, v in metadata.items():
                if "corners" in v:
                    corners_dict[k] = np.array(v["corners"], dtype=np.float32)

    results = []

    for img_path in test_images:
        card, normalized, corners, top_matches = matcher.process_image(str(img_path))

        # Check against ground truth
        image_key = img_path.stem
        truth_group = metadata.get(image_key, {})
        ground_truth = (truth_group.get("name"), truth_group.get("set"))

        # Check detection quality
        detected = False
        gt_corners = corners_dict.get(image_key)
        if corners is not None and gt_corners is not None:
            loss, _ = compute_polygon_loss(corners, gt_corners)
            detected = loss <= 20

        # Check correctness
        is_correct_overall = (card == ground_truth) if card else False
        is_correct_detected = (card == ground_truth) if card and detected else False

        correct_in_top5 = (
            any(match["card"] == ground_truth for match in top_matches)
            if top_matches and ground_truth[0]
            else False
        )

        correct_in_top5_detected = correct_in_top5 if detected else False

        results.append(
            {
                "image": img_path.name,
                "detected": detected,
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
    correct_in_top5_detected = sum(1 for r in results if r["correct_in_top5_detected"])

    print(f"\nSUMMARY:")
    print(f"  Detection: {detected}/{len(results)} ({detected/len(results)*100:.1f}%)")
    print(
        f"  Accuracy (on detected): {correct_detected}/{detected} ({correct_detected/detected*100:.1f}%)"
    )
    print(
        f"  Top-5 (on detected): {correct_in_top5_detected}/{detected} ({correct_in_top5_detected/detected*100:.1f}%)"
    )
    print(
        f"  Overall accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)"
    )

    # Save results
    with open("./data/deep_Embedding_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "predicted", "ground_truth", "correct", "detected"])
        for r in results:
            writer.writerow(
                [
                    r["image"],
                    r["card"][0] if r["card"] else "No match",
                    r["ground_truth"][0] if r["ground_truth"][0] else "Unknown",
                    int(r["correct"]),
                    int(r["detected"]),
                ]
            )


if __name__ == "__main__":
    main()

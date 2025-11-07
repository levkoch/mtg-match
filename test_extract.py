"""
test_extract.py

testing just extracting the card from the overall background image.
assessing whether we are able to detect what it is will be done later.
"""

import json
import numpy as np
from scipy.spatial.distance import cdist
from extract import (
    DETECTOR,
    detect_card_edges_with_border,
    detect_card_edges_with_sides,
)


def compute_polygon_loss(
    predicted_points, true_points, method='mean_distance'
):
    """
    Compute loss between predicted and ground truth quadrilateral points.

    Args:
        predicted_points: numpy array of shape (4, 2) - predicted corner points
        true_points: list or numpy array of shape (4, 2) - ground truth corner points
        method: Loss calculation method
            - 'mean_distance': Mean Euclidean distance after optimal matching
            - 'iou': 1 - Intersection over Union of the polygons
            - 'hausdorff': Hausdorff distance between point sets
            - 'mse': Mean squared error of matched points

    Returns:
        loss: float value representing the loss
        matched_pairs: list of tuples showing which predicted points matched to which true points
    """
    if predicted_points is None:
        return float('inf'), None

    # Convert to numpy arrays
    pred = np.array(predicted_points, dtype=np.float32)
    true = np.array(true_points, dtype=np.float32)

    if pred.shape != (4, 2) or true.shape != (4, 2):
        raise ValueError(
            'Both predicted and true points must have shape (4, 2)'
        )

    # Find optimal matching between predicted and true points
    # Compute pairwise distances
    distances = cdist(pred, true, metric='euclidean')

    # Use Hungarian algorithm (greedy approximation for 4 points)
    # For 4 points, we can try all 24 permutations, but greedy works well
    matched_indices = []
    used_true = set()
    used_pred = set()

    # Simple greedy matching: assign each pred to closest unused true point
    for _ in range(4):
        min_dist = float('inf')
        best_pred, best_true = -1, -1

        for i in set(range(4)) - used_pred:
            for j in set(range(4)) - used_true:
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    best_pred, best_true = i, j

        matched_indices.append((best_pred, best_true))
        used_pred.add(best_pred)
        used_true.add(best_true)

    # Reorder predicted points to match true points order
    matched_pred = np.array(
        [pred[p] for p, t in sorted(matched_indices, key=lambda x: x[1])]
    )

    if method == 'mean_distance':
        # Mean Euclidean distance between matched points
        distances = np.linalg.norm(matched_pred - true, axis=1)
        loss = np.mean(distances)

    elif method == 'mse':
        # Mean squared error
        loss = np.mean((matched_pred - true) ** 2)

    elif method == 'hausdorff':
        # Hausdorff distance (maximum of min distances)
        dist_matrix = cdist(matched_pred, true, metric='euclidean')
        loss = max(
            np.min(dist_matrix, axis=1).max(),
            np.min(dist_matrix, axis=0).max(),
        )

    elif method == 'iou':
        # Intersection over Union (requires shapely or cv2)
        try:
            import cv2

            # Create binary masks
            mask_pred = np.zeros((1000, 1000), dtype=np.uint8)
            mask_true = np.zeros((1000, 1000), dtype=np.uint8)

            cv2.fillPoly(mask_pred, [pred.astype(np.int32)], 255)
            cv2.fillPoly(mask_true, [true.astype(np.int32)], 255)

            intersection = np.logical_and(mask_pred, mask_true).sum()
            union = np.logical_or(mask_pred, mask_true).sum()

            iou = intersection / union if union > 0 else 0
            loss = 1 - iou
        except ImportError:
            raise ImportError('IoU method requires cv2 (OpenCV)')

    else:
        raise ValueError(f'Unknown method: {method}')

    matched_pairs = [(pred[p], true[t]) for p, t in matched_indices]

    return loss, matched_pairs


def evaluate_detection(
    labels_dict, image_dir, detect_function: DETECTOR, method='mean_distance'
):
    """
    Evaluate detection performance across multiple labeled images.

    Args:
        labels_dict: Dictionary mapping image names to ground truth points
        image_dir: Directory containing the images
        detect_function: Function that takes image_path and returns (quadrilateral, output_img)
        method: Loss calculation method (see compute_polygon_loss)

    Returns:
        results: Dictionary with loss per image and summary statistics
    """
    import os

    results = {'per_image': {}, 'total_loss': 0, 'successful': 0, 'failed': 0}

    for img_name, group in labels_dict.items():
        true_corners: list[list[float]] = group['corners']

        test_path = os.path.join(image_dir, img_name + '.png')
        if os.path.exists(test_path):
            img_path = test_path
        else:
            print(f'Warning: Could not find image file for {img_name}')
            results['failed'] += 1
            continue

        # Run detection
        pred_points, img = detect_function(img_path, False, False)

        # Compute loss
        loss, _ = compute_polygon_loss(
            pred_points, true_corners, method=method
        )

        results['per_image'][img_name] = {
            'loss': loss,
            'predicted': pred_points.tolist()
            if pred_points is not None
            else None,
            'ground_truth': true_corners,
        }

        if loss != float('inf'):
            results['total_loss'] += loss
            results['successful'] += 1
        else:
            results['failed'] += 1

    # Compute average loss
    if results['successful'] > 0:
        results['average_loss'] = results['total_loss'] / results['successful']
    else:
        results['average_loss'] = float('inf')

    print(f'\n=== Summary {detect_function.__name__} ===')
    print(f"Successful detections: {results['successful']}/{len(labels_dict)}")
    print(f"Average loss: {results['average_loss']:.2f}")

    return results

def count_images_under_loss(results, threshold):
    """
    Count how many images have a loss below the given threshold.
    
    Args:
        results: Dictionary with 'per_image' key containing image results
        threshold: Maximum loss value (images with loss < threshold are counted)
    
    Returns:
        int: Number of images with loss below threshold
    """
    count = 0
    for _, img_results in results['per_image'].items():
        if img_results['loss'] is not None and img_results['loss'] <= threshold:
            count += 1
    return count


if __name__ == '__main__':

    with open('./data/generations.json') as fp:
        labels = json.load(fp)

    THRESHOLDS = [1, 3, 5, 10, 20, 40, 60, 100]

    results_border = evaluate_detection(
        labels, './data/generations', detect_card_edges_with_border
    )

    for t in THRESHOLDS:
        print(f'<={t} mean deviation: {count_images_under_loss(results_border, t)}')

    results_sides = evaluate_detection(
        labels, './data/generations', detect_card_edges_with_sides
    )

    for t in THRESHOLDS:
        print(f'<={t} mean deviation: {count_images_under_loss(results_sides, t)}')

    """
    === Summary detect_card_edges_with_border===
    Successful detections: 698/800
    Average loss: 169.74
    1 mean deviation: 0
    3 mean deviation: 0
    5 mean deviation: 0
    10 mean deviation: 0
    20 mean deviation: 12
    40 mean deviation: 40
    60 mean deviation: 81
    100 mean deviation: 153

    === Summary detect_card_edges_with_sides===
    Successful detections: 800/800
    Average loss: 680.20
    1 mean deviation: 0
    3 mean deviation: 0
    5 mean deviation: 0
    10 mean deviation: 0
    20 mean deviation: 0
    40 mean deviation: 0
    60 mean deviation: 0
    100 mean deviation: 0
    """

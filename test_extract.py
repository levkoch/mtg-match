"""
test_extract.py

testing just extracting the card from the overall background image.
assessing whether we are able to detect what it is will be done later.
"""

import json
from typing import Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from extract import (
    DETECTOR,
    detect_card_edges_with_border,
    detect_card_edges_with_border_hsearch,
    detect_card_edges_with_sides,
    detect_card_edges_with_sides_hsearch
)

THRESHOLDS = [1, 3, 5, 10, 20, 40, 60, 100]


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
    labels_dict, image_dir, detect_function: DETECTOR, method='mean_distance', display = False
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

        # we won't display everything, but a random sampling
        if display and np.random.randint(0, 40) == 0:
            # the returned image already has the contours drawn on it,
            # add true contours
            reordered_corners = [true_corners[1], true_corners[0], true_corners[2], true_corners[3]]

            cv2.polylines(img, [np.array(reordered_corners, dtype=np.int32)], True, (255, 0, 0), 3)
            for point in true_corners: 
                cv2.circle(img, (int(point[0]), int(point[1])), 8, (0, 0, 255), -1)

            cv2.imshow(f'[{img_name}] loss: {loss}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

def display_losses(results: dict[str, dict[str, Any]], kind: str) -> None:
    losses = [
        img_results['loss'] 
        for _, img_results in results['per_image'].items() 
        if img_results['loss'] is not None
    ]
    
    # Convert to array and separate inf values
    losses = np.array(losses)
    inf_count = np.sum(np.isinf(losses))
    finite_losses = losses[np.isfinite(losses)]
    finite_losses = finite_losses[finite_losses > 0]  # Filter out zeros and negatives

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create histogram with logarithmic bins for finite values
    if len(finite_losses) > 0:
        log_bins = np.logspace(np.log10(finite_losses.min()), np.log10(finite_losses.max()), 30)
        counts, bins, patches = ax.hist(finite_losses, bins=log_bins, edgecolor='black', alpha=0.7, label='Finite losses')
        
        # Add a bar for inf values at the end if they exist
        if inf_count > 0:
            # Position the inf bar after the last bin
            max_x = finite_losses.max()
            inf_x_position = max_x * 2  # Place it at 2x the maximum value
            bar_width = max_x * 0.5  # Make the bar width proportional
            
            ax.bar(inf_x_position, inf_count, width=bar_width, color='red', 
                alpha=0.7, edgecolor='black', label=f'Inf ({inf_count})')
            
            # Add text label on the inf bar
            ax.text(inf_x_position, inf_count, f'{inf_count}', 
                    ha='center', va='bottom', fontweight='bold')
            ax.text(inf_x_position, 0, 'inf', ha='center', va='top', fontsize=10, fontweight='bold')

    # Set x-axis to log scale
    ax.set_xscale('log')

    # Labels and title
    ax.set_xlabel('Loss Value (log scale)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of Losses using {kind} (Log Scale)', fontsize=14, fontweight='bold')

    # Add grid for better readability
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    # Add legend if there are inf values
    if inf_count > 0:
        ax.legend()

    # Add statistics as text
    if len(finite_losses) > 0:
        median_loss = np.median(finite_losses)
        mean_loss = np.mean(finite_losses)
        stats_text = f'Finite: {len(finite_losses)}\nNone Found (Inf): {inf_count}\nMedian: {median_loss:.4f}\nMean: {mean_loss:.4f}'
    else:
        stats_text = f'Finite: 0\nNot Found (Inf): {inf_count}'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

    for t in THRESHOLDS:
        print(f'<={t} mean deviation: {count_images_under_loss(results, t)}')


if __name__ == '__main__':
    with open('./data/generations.json') as fp:
        labels = json.load(fp)

    results_sides = evaluate_detection(
        labels, './data/generations', detect_card_edges_with_sides, # display=True
    )

    display_losses(results_sides, 'side detection (before tuning)')

    results_border = evaluate_detection(
        labels, './data/generations', detect_card_edges_with_border, # display=True
    )

    display_losses(results_border, 'border detection (before tuning)')

    def border_detect_tuned(img_path: str, A, B):
        return detect_card_edges_with_border_hsearch(
            img_path, black_threshold=30, blur_kernel=(3,3), close_iterations=1, open_iterations=2)
    
    results_border_tuned = evaluate_detection(
        labels, './data/generations', border_detect_tuned,
    )
    display_losses(results_border_tuned, 'tuned border detection')

    def side_detect_tuned(img_path: str, A, B):
        return detect_card_edges_with_sides_hsearch(
            img_path, thresh_c=5, kernel_size=(5,5), size_thresh=5000, approx_epsilon=0.01)
    
    results_sides_tuned = evaluate_detection(
        labels, './data/generations', side_detect_tuned,)
    display_losses(results_sides_tuned, 'tuned side detection')

    

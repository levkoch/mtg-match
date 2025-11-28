"""
hyperparameter search through with processpool to make it way more effective.
RESULTS: https://docs.google.com/spreadsheets/d/1KFkijCLsVVo6wjp99h0GelhTZ_oTIEeHKUaSyraX84g/
"""

import csv
import json
import numpy as np
from itertools import count, product
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from extract import detect_card_edges_with_sides_hsearch
from test_extract import compute_polygon_loss

def _print_progress(progress_tracker: str) -> None:
    """prints the progress of the loading process"""
    completed: int = progress_tracker.count('*')

    stride = max(1, len(progress_tracker) // 40)
    batches: list[str] = [
        progress_tracker[i : i + stride] for i in range(0, len(progress_tracker), stride)]
    output = ''.join("*" if any(c == "*" for c in batch) else " " for batch in batches)
    
    print(
        f'\r <>  [{completed:03d}/{len(progress_tracker):03d}] completed [{output}]',
        end='',
        flush=True,
    )


# Define parameters to test
kernels = [(3, 3), (5, 5), (7, 7), (9, 9)] 
threshold_constants = [1, 2, 3, 5, 7, 10, 15, 20]
size_thresholds = [5_000, 10_000, 15_000, 20_000]
approx_epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

with open('./data/generations.json') as fp:
    LABELS = json.load(fp)

GENERATIONS = './data/generations'


def process_image(args):
    img_path, img_name, true_corners, thresh_c, kernel_size, size_thresh, epsilon = args
    
    # Call the detection function with hyperparameters
    pred_points, _ = detect_card_edges_with_sides_hsearch(
        img_path,
        thresh_c=thresh_c,
        kernel_size=kernel_size,
        size_thresh=size_thresh,
        approx_epsilon=epsilon,
    )
    
    # Compute loss
    loss, _ = compute_polygon_loss(
        pred_points, true_corners, method='mean_distance'
    )
    
    return img_name, loss, pred_points

def search_parameters():
    # For initial search, sample a subset of parameter combinations
    # Full grid search would be: 5 * 4 * 4 * 4 * 3 * 5 * 4 * 3 = 115,200 combinations!
    # Start with a coarse grid search on most important parameters
    
    test_counter = count(start=1)
    total_tests = len(threshold_constants) * len(kernels) * len(size_thresholds) * len(approx_epsilons)

    results = []

    # Fixed parameters for initial search (use defaults)
    epsilon = 0.02

    for thresh_c, kernel_size, size_thresh, epsilon in product(
        threshold_constants, kernels, size_thresholds, approx_epsilons
    ):
        print(f"[{next(test_counter):02d}/{total_tests}] thresh_c={thresh_c}, kernel={kernel_size}, size_thresh={size_thresh}, epsilon={epsilon}")
        
        # Create arguments for each image with ground truth labels
        args_list = []
        for img_name, group in LABELS.items():
            true_corners = group['corners']
            
            test_path = f'{GENERATIONS}/{img_name}.png'
            import os
            if not os.path.exists(test_path):
                continue
                
            args_list.append((
                test_path, img_name, true_corners,
                thresh_c, kernel_size, size_thresh, epsilon
            ))

        # Aggregate results
        total_loss = 0.0
        successful_detections = 0
        failed_detections = 0
        good_detections = 0  # loss <= 20
        counter = count()
        
        # Process images in parallel
        with ProcessPoolExecutor() as executor:
            futures: list[Future[tuple[str, float, np.ndarray]]] = [
                executor.submit(process_image, args) for args in args_list
            ]
            
            for future in as_completed(futures):
                img_name, loss, pred_points = future.result()
                idx = next(counter)
                _print_progress('*' * (idx + 1) + ' ' * (len(args_list) - idx - 1))

                if loss != float('inf'):
                    total_loss += loss
                    successful_detections += 1
                    if loss <= 20:
                        good_detections += 1
                else:
                    failed_detections += 1

        print()
        
        avg_loss = total_loss / successful_detections if successful_detections > 0 else float('inf')
        success_rate = successful_detections / len(args_list) if len(args_list) > 0 else 0
        good_rate = good_detections / len(args_list) if len(args_list) > 0 else 0
        
        print(f'Successful: {successful_detections}/{len(args_list)}, Good (≤20): {good_detections}/{len(args_list)},'
              f' Average loss={avg_loss:.4f}, Success rate={success_rate:.4f}, Good rate={good_rate:.4f}')
        print('')

        results.append((thresh_c, kernel_size, size_thresh, epsilon, avg_loss, success_rate, successful_detections, good_detections, good_rate))

    # Sort by good detections (descending), then by average loss (ascending)
    results.sort(key=lambda x: (-x[7], x[4]))

    print("\nTop Results (by most good detections, then lowest average loss):")
    print("thresh_c, kernel_size, size_thresh, epsilon Avg_Loss, Success_Rate, Successful, Good_Detections, Good_Rate")
    for thresh_c, kernel_size, size_thresh, epsilon, avg_loss, success_rate, successful, good_detections, good_rate in results:
        print(f"{thresh_c}, {kernel_size}, {size_thresh}, {epsilon} {avg_loss:.4f}, {success_rate:.4f}, {successful}, {good_detections}, {good_rate:.4f}")

    return results

if __name__ == "__main__":
    results = search_parameters()

    with open('param_search.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
"""
hyperparameter search through with processpool to make it way more effective.
RESULTS: https://docs.google.com/spreadsheets/d/1HhvWy_OC3u75TpjFCGrbpaV7kKGuCHgZWpw8QpAPHF8/
         https://public.flourish.studio/visualisation/26393086/
"""

import csv
import json
import numpy as np
from itertools import count, product
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from extract import detect_card_edges_with_border_hsearch
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
black_thresholds = [20]
kernels = [(3, 3), (5, 5), (7, 7), (9, 9)] 
polygon_epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
close_iterations = [1, 2, 3, 4]
open_iterations = [1, 2, 3, 4]

with open('./data/generations.json') as fp:
    LABELS = json.load(fp)

GENERATIONS = './data/generations'


def process_image(args):
    img_path, img_name, true_corners, threshold, blur_kernel, close_iter, open_iter, epsilon = args
    
    # Call the detection function with hyperparameters
    pred_points, _ = detect_card_edges_with_border_hsearch(
        img_path,
        display=False,
        black_threshold=threshold,
        blur_kernel=blur_kernel,
        close_iterations=close_iter,
        open_iterations=open_iter,
        polygon_epsilon=epsilon,
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
    total_tests = len(black_thresholds) * len(kernels) * len(close_iterations) * len(open_iterations)

    results = []

    # Fixed parameters for initial search (use defaults)
    epsilon = 0.02

    for threshold, blur_kernel, close_iter, open_iter in product(
        black_thresholds, kernels, close_iterations, open_iterations
    ):
        print(f"[{next(test_counter):02d}/{total_tests}] threshold={threshold}, blur_kernel={blur_kernel}, close={close_iter}, open={open_iter}")
        
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
                threshold, blur_kernel,
                close_iter, open_iter, epsilon
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
        
        print(f'Successful: {successful_detections}/{len(args_list)}, Good (≤20): {good_detections}/{len(args_list)}, Average loss={avg_loss:.4f}, Success rate={success_rate:.4f}, Good rate={good_rate:.4f}')
        print('')

        results.append((threshold, blur_kernel[0], close_iter, open_iter, avg_loss, success_rate, successful_detections, good_detections, good_rate))

    # Sort by good detections (descending), then by average loss (ascending)
    results.sort(key=lambda x: (-x[7], x[4]))

    print("\nTop Results (by most good detections, then lowest average loss):")
    print("Threshold, Morph_Kernel, Close_Iter, Open_Iter, Avg_Loss, Success_Rate, Successful, Good_Detections, Good_Rate")
    for threshold, morph_kernel, close_iter, open_iter, avg_loss, success_rate, successful, good_detections, good_rate in results:
        print(f"{threshold}, {morph_kernel}, {close_iter}, {open_iter}, {avg_loss:.4f}, {success_rate:.4f}, {successful}, {good_detections}, {good_rate:.4f}")

    return results

if __name__ == "__main__":
    results = search_parameters()

    with open('param_search.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
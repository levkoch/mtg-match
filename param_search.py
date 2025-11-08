"""
hyperparameter search with processpool to make it way more effective.
(example, we will be changing what we actually do with it, i just wanted the template here)
"""

import numpy as np
from skimage import io
from os import listdir
from itertools import count, product
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from edge import canny

def _print_progress(progress_tracker: str) -> None:
    """prints the progress of the loading process"""
    completed: int = progress_tracker.count('*')
    print(
        f'\r <>  [{completed:02d}/{len(progress_tracker):02d}] completed [{progress_tracker}]',
        end='',
        flush=True,
    )

# Define parameters to test
sigmas = [0.7]
highs = [18.0]
gaps = [2.0]

def process_image(args):
    img_file, sigma, high, low = args
    
    img = io.imread('images/objects/' + img_file, as_gray=True)
    gt = io.imread('images/gt/' + img_file + '.gtf.pgm', as_gray=True)
    
    mask = (gt != 5)  # 'don't care' region
    gt = (gt == 0)  # binary image of GT edges
    
    # this is the call that takes forever, and is CPU heavy,
    # so we want to process-pool it
    edges = canny(img, kernel_size=5, sigma=sigma, high=high, low=low)
    edges = edges * mask
    
    n_detected = np.sum(edges)
    n_gt = np.sum(gt)
    n_correct = np.sum(edges * gt)
    
    return n_detected, n_gt, n_correct

def search_parameters():
    test_counter = count(start=1)
    total_tests = len(sigmas) * len(highs) * len(gaps)

    results = []

    for sigma, high, gap in product(sigmas, highs, gaps):
        low = round(high - gap, 2)
        
        print(f"[{next(test_counter):02d}/{total_tests}] sigma={sigma}, low={low}, high={high}")
        
        img_files = listdir('images/objects')
        
        # Create arguments for each image
        args_list = [
            (img_file, sigma, high, low) for img_file in img_files
        ]

        # Aggregate results
        n_detected = 0.0
        n_gt = 0.0
        n_correct = 0.0
        counter = count()
        
        # Process images in parallel
        with ProcessPoolExecutor() as executor:
            futures: list[Future[tuple[float, float, float]]] = [
                executor.submit(process_image, args) for args in args_list]
            
            for future in as_completed(futures):
                detected, gt, correct = future.result()
                idx = next(counter)
                _print_progress('*' * (idx + 1) + ' ' * (len(img_files) - idx - 1))

                n_detected += detected
                n_gt += gt
                n_correct += correct

        print()
        
        p_total = n_correct / n_detected
        r_total = n_correct / n_gt
        f1 = 2 * (p_total * r_total) / (p_total + r_total)
        print('Total precision={:.4f}, Total recall={:.4f}'.format(p_total, r_total))
        print('F1 score={:.4f}'.format(f1))
        print('')

        results.append((sigma, low, high, p_total, r_total, f1))

    print("Sigma, Low, High, Precision, Recall, F1 Score")
    for sigma, low, high, p_total, r_total, f1 in results:
        print(f"{sigma:.2f}, {low:.2f}, {high:.2f}, {p_total:.4f}, {r_total:.4f}, {f1:.4f} ")

if __name__ == '__main__':
    search_parameters()
"""
extract.py

pulling the mtg card from out of the image
"""

from typing import Callable, Optional, TypeAlias, Union
import cv2

import numpy as np
from pathlib import Path

DETECTOR: TypeAlias = Callable[
    [Union[Path, str], bool, bool], 
    tuple[Optional[list[list[float]]], Optional[np.ndarray]]]

def detect_card_edges_with_border(image_path: str, display=True, show_steps=True):
    """
    Detect MTG card edges by finding the thick black border.

    NOTE: this is less robust as not all of the mtg cards have black borders
    (but we can restrict our dataset space to just be that if we want)
    but it is more effective than whatever is happening above.
    but there are tuning and things that need to be done.

    Args:
        image_path: Path to the image file
        display: Whether to display the result
        show_steps: Whether to show intermediate processing steps

    Returns:
        quadrilateral: 4 corner points of the detected card
        processed_image: Image with the bounding box drawn
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error: Could not read image {image_path}')
        return None, None

    # Create a copy for drawing
    output = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to detect black border
    # MTG cards have a dark black border, so we threshold for dark pixels
    _, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up and close gaps
    kernel = np.ones((5, 5), np.uint8)
    # Close small gaps in the border
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate image area for filtering
    image_area = img.shape[0] * img.shape[1]
    min_area = 0.1 * image_area  # Card should be at least 10% of image
    max_area = 0.9 * image_area  # But not more than 90%

    # Filter contours by area
    filtered_contours = [
        c for c in contours if min_area < cv2.contourArea(c) < max_area
    ]
    # Create visualization of filtered contours
    contours_img = img.copy()
    cv2.drawContours(contours_img, filtered_contours, -1, (0, 255, 0), 2)

    # Find the largest valid contour (should be the card border)
    if not filtered_contours:
        if display: print('No valid contours found after filtering')
        return None, output

    # Sort by area and select the largest
    sorted_contours = sorted(
        filtered_contours, key=cv2.contourArea, reverse=True
    )
    card_contour = sorted_contours[0]

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)

    # If we have 4 points, we found a quadrilateral
    if len(approx) == 4:
        quadrilateral = approx.reshape(4, 2)
    else:
        # If not exactly 4 points, use minimum area rectangle
        rect = cv2.minAreaRect(card_contour)
        box = cv2.boxPoints(rect)
        quadrilateral = np.intp(box)

    # Draw the quadrilateral
    cv2.drawContours(output, [quadrilateral], 0, (0, 255, 0), 3)

    # Draw corner points
    for i, point in enumerate(quadrilateral):
        cv2.circle(output, tuple(point), 8, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(i),
            tuple(point + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    if display:
        # Prepare all images for display
        def resize_for_display(image, max_dim=800):
            """Resize image if too large"""
            if len(image.shape) == 2:  # Grayscale
                height, width = image.shape
            else:  # Color
                height, width = image.shape[:2]

            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                return cv2.resize(image, None, fx=scale, fy=scale)
            return image

        if show_steps:
            # Show all intermediate steps
            steps = [
                ('1. Original Image', img),
                ('2. Grayscale', gray),
                ('3. Gaussian Blur', blurred),
                ('4. Binary (Black Detection)', binary),
                ('5. Morphology Closed', closed),
                ('6. Morphology Opened', opened),
                ('7. Filtered Contours', contours_img),
                ('8. Final Detection', output),
            ]

            for title, step_img in steps:
                display_img = resize_for_display(step_img)
                cv2.imshow(title, display_img)

            print('\nPress any key to continue to next image...')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Show only final result
            display_img = resize_for_display(output)
            cv2.imshow('Card Detection', display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return quadrilateral, output

def detect_card_edges_with_border_hsearch(
        image_path: str, 
        black_threshold: int = 60, blur_kernel: tuple[int, int] = (5, 5), 
        polygon_epsilon: float = 0.02, close_iterations: int = 2, open_iterations: int = 1, 
        display=False):
    """
    Detect MTG card edges by finding the thick black border.

    NOTE: this is less robust as not all of the mtg cards have black borders
    (but we can restrict our dataset space to just be that if we want)
    but it is more effective than whatever is happening above.
    but there are tuning and things that need to be done.

    Args:
        image_path: Path to the image file
        display: Whether to display the result

    Returns:
        quadrilateral: 4 corner points of the detected card
        processed_image: Image with the bounding box drawn
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f'Error: Could not read image {image_path}')
        return None, None

    # Create a copy for drawing
    output = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # Threshold to detect black border
    # MTG cards have a dark black border, so we threshold for dark pixels
    _, binary = cv2.threshold(blurred, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up and close gaps
    kernel = np.ones(blur_kernel, np.uint8)
    # Close small gaps in the border
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    # Find contours
    contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Calculate image area for filtering
    image_area = img.shape[0] * img.shape[1]
    min_area = 0.1 * image_area  # Card should be at least 10% of image
    max_area = 0.9 * image_area  # But not more than 90%

    # Filter contours by area
    filtered_contours = [
        c for c in contours if min_area < cv2.contourArea(c) < max_area
    ]

    # Find the largest valid contour (should be the card border)
    if not filtered_contours:
        if display:
            print('No valid contours found after filtering')
        return None, output

    # Sort by area and select the largest
    sorted_contours = sorted(
        filtered_contours, key=cv2.contourArea, reverse=True
    )
    card_contour = sorted_contours[0]

    # Approximate the contour to a polygon
    epsilon = polygon_epsilon * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)

    # If we have 4 points, we found a quadrilateral
    if len(approx) == 4:
        quadrilateral = approx.reshape(4, 2)
    else:
        # If not exactly 4 points, use minimum area rectangle
        rect = cv2.minAreaRect(card_contour)
        box = cv2.boxPoints(rect)
        quadrilateral = np.intp(box)

    # Draw the quadrilateral
    cv2.drawContours(output, [quadrilateral], 0, (0, 255, 0), 3)

    # Draw corner points
    for i, point in enumerate(quadrilateral):
        cv2.circle(output, tuple(point), 8, (0, 0, 255), -1)
        cv2.putText(
            output,
            str(i),
            tuple(point + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    if display:
        def resize_for_display(image, max_dim=800):
            """Resize image if too large"""
            if len(image.shape) == 2:  # Grayscale
                height, width = image.shape
            else:  # Color
                height, width = image.shape[:2]

            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                return cv2.resize(image, None, fx=scale, fy=scale)
            return image

        display_img = resize_for_display(output)
        cv2.imshow('Card Detection', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return quadrilateral, output


def detect_card_edges_with_sides(image_path: str, a=True, b=True) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Detect a single card from the given image
    Args:
        image_path: string path to the image to extract information from
    Returns:
        tuple of (quadrilateral 4 corner points, loaded image).
        Returns (None, image) if no card is detected

    Based off of https://github.com/hj3yoo/mtg_card_detector/blob/master/opencv_dnn.py
    """

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    # Find card contours in the image HPARAMS
    thresh_c=5
    kernel_size=(3, 3)
    size_thresh=10000

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, thresh_c)

    # Dilute the image, then erode them to remove minor noises
    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    # Find the contour
    cnts, hier = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, img

    
    cnts_rect = []
    for cnt in cnts:
        # Check if contour meets size requirement
        size = cv2.contourArea(cnt)
        if size < size_thresh:
            continue
        
        # Check if contour can be approximated as a 4-sided polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        
        if len(approx) == 4:
            cnts_rect.append(approx)

    # Sort by area (largest first) to prioritize bigger cards
    cnts_rect.sort(key=cv2.contourArea, reverse=True)

    # If no cards found, return None for contour
    if len(cnts_rect) == 0:
        return None, img
    
    # Get the first detected card contour
    cnt = cnts_rect[0]
    
    # Extract the 4 corner points and convert to proper format
    pts = np.array([p[0] for p in cnt], dtype=np.float32)
    
    # Order the points (top-left, top-right, bottom-right, bottom-left)
    ordered_pts = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered_pts[0] = pts[np.argmin(s)]
    ordered_pts[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    ordered_pts[1] = pts[np.argmin(diff)]
    ordered_pts[3] = pts[np.argmax(diff)]

    return ordered_pts, img

def detect_card_edges_with_sides_hsearch(
        image_path: str, thresh_c: int = 5, 
        kernel_size: tuple[int, int] = (3, 3), 
        size_thresh: int = 10_000, approx_epsilon: float = 0.04
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Detect a single card from the given image
    Args:
        image_path: string path to the image to extract information from
    Returns:
        tuple of (quadrilateral 4 corner points, loaded image).
        Returns (None, image) if no card is detected

    Based off of https://github.com/hj3yoo/mtg_card_detector/blob/master/opencv_dnn.py
    """

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    # find card contours
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, thresh_c)

    # Dilute the image, then erode them to remove minor noises
    kernel = np.ones(kernel_size, np.uint8)
    img_dilate = cv2.dilate(img_thresh, kernel, iterations=1)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)

    # Find the contour
    cnts, hier = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, img

    cnts_rect = []
    for cnt in cnts:
        # Check if contour meets size requirement
        size = cv2.contourArea(cnt)
        if size < size_thresh:
            continue
        
        # Check if contour can be approximated as a 4-sided polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon * peri, True)
        
        if len(approx) == 4:
            cnts_rect.append(approx)
            
    # Sort by area (largest first) to prioritize bigger cards
    cnts_rect.sort(key=cv2.contourArea, reverse=True)

    # If no cards found, return None for contour
    if len(cnts_rect) == 0:
        return None, img
    
    # Get the first detected card contour
    cnt = cnts_rect[0]
    
    # Extract the 4 corner points and convert to proper format
    pts = np.array([p[0] for p in cnt], dtype=np.float32)
    
    # Order the points (top-left, top-right, bottom-right, bottom-left)
    ordered_pts = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    ordered_pts[0] = pts[np.argmin(s)]
    ordered_pts[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    ordered_pts[1] = pts[np.argmin(diff)]
    ordered_pts[3] = pts[np.argmax(diff)]

    return ordered_pts, img
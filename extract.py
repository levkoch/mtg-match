"""
extract.py

pulling the mtg card from out of the image
"""

from typing import Callable, TypeAlias
import cv2

import numpy as np
from pathlib import Path

DETECTOR: TypeAlias = Callable[[Path, bool, bool], tuple[list[list[float]], np.ndarray]]

def detect_card_edges_with_border(image_path, display=True, show_steps=True):
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
    max_area = 0.8 * image_area  # But not more than 80%

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

def detect_card_edges_with_sides(image_path, display=True, show_steps=True):
    """
    Detect MTG card edges and create a quadrilateral bounding box.

    NOTE: kind of wonky and doesn't quite work on noisy backgrounds. 
    the article used some thresholding thing, but i haven't tried that out yet.
    
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
        print(f"Error: Could not read image {image_path}")
        return None, None
    # Create a copy for drawing
    output = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding or Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter out unhelpful contours (sometimes it decides to contour the whole image)
    image_area = img.shape[0] * img.shape[1]
    threshold_area = 0.9 * image_area
    filtered_contours = [c for c in contours if cv2.contourArea(c) < threshold_area]
        
    contours_img = img.copy()
    cv2.drawContours(contours_img, filtered_contours, -1, (0, 255, 0), 2)
    
    if not filtered_contours:
        print("No contours found after filtering")
        return None, output
    
    largest_contour = filtered_contours[0]
    
    # Approximate the contour to a polygon
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have 4 points, we found a quadrilateral
    if len(approx) == 4:
        quadrilateral = approx.reshape(4, 2)
    else:
        # If not exactly 4 points, use minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        quadrilateral = np.intp(box)
    
    # Draw the quadrilateral
    cv2.drawContours(output, [quadrilateral], 0, (0, 255, 0), 3)
    
    # Draw corner points
    for point in quadrilateral:
        cv2.circle(output, tuple(point), 8, (0, 0, 255), -1)
    
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
                ("1. Original Image", img),
                ("4. Canny Edges", edges),
                ("5. Dilated Edges", dilated),
                ("6. All Contours", contours_img),
                ("7. Final Detection", output)
            ]
            
            for title, step_img in steps:
                display_img = resize_for_display(step_img)
                cv2.imshow(title, display_img)
            
            print("\nPress any key to continue to next image...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Show only final result
            display_img = resize_for_display(output)
            cv2.imshow('Card Detection', display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return quadrilateral, output
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
    tuple[Optional[list[list[float]]], Optional[np.ndarray]],
]


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
        print(f"Error: Could not read image {image_path}")
        return None, None

    # Create a copy for drawing
    output = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold to detect black border
    # MTG cards have a dark black border, so we threshold for dark pixels
    _, binary = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up and close gaps
    kernel = np.ones((3, 3), np.uint8)
    # Close small gaps in the border
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        if display:
            print("No valid contours found after filtering")
        return None, output

    # Sort by area and select the largest
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
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
                ("1. Original Image", img),
                ("2. Grayscale", gray),
                ("3. Gaussian Blur", blurred),
                ("4. Binary (Black Detection)", binary),
                ("5. Morphology Closed", closed),
                ("6. Morphology Opened", opened),
                ("7. Filtered Contours", contours_img),
                ("8. Final Detection", output),
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
            cv2.imshow("Card Detection", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return quadrilateral, output


def detect_card_edges_with_border_hsearch(
    image_path: str,
    black_threshold: int = 60,
    blur_kernel: tuple[int, int] = (5, 5),
    polygon_epsilon: float = 0.02,
    close_iterations: int = 2,
    open_iterations: int = 1,
    display=False,
):
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
        print(f"Error: Could not read image {image_path}")
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
    closed = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=close_iterations
    )
    # Remove small noise
    opened = cv2.morphologyEx(
        closed, cv2.MORPH_OPEN, kernel, iterations=open_iterations
    )

    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            print("No valid contours found after filtering")
        return None, output

    # Sort by area and select the largest
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
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
        cv2.imshow("Card Detection", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return quadrilateral, output


def detect_card_edges_with_sides(image_path: str, display=True, show_steps=True):
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
                ("7. Final Detection", output),
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
            cv2.imshow("Card Detection", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return quadrilateral, output


def detect_card_edges_with_clahe(image_path: str, display=True, show_steps=True):
    """
    Detect MTG card edges using CLAHE and edge detection.

    This approach detects the card outline based on contrast changes,
    regardless of whether the border is black, white, silver, or any other color.

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

    # Apply CLAHE for adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Use Canny edge detection to find edges regardless of color
    # The card outline should create strong edges
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to connect nearby edge pixels
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Close gaps in the edges
    kernel_close = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # Fill in the interior of detected shapes
    # This helps create solid contours from edge outlines
    contours_temp, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours_temp, -1, 255, thickness=cv2.FILLED)

    # Find contours on the filled image
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate image area for filtering
    image_area = img.shape[0] * img.shape[1]
    min_area = 0.1 * image_area  # Card should be at least 10% of image
    max_area = 0.9 * image_area  # But not more than 90%

    # Filter contours by area and aspect ratio
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            filtered_contours.append(c)

    # Create visualization of filtered contours
    contours_img = img.copy()
    cv2.drawContours(contours_img, filtered_contours, -1, (0, 255, 0), 2)

    # Find the largest valid contour (should be the card)
    if not filtered_contours:
        if display:
            print("No valid contours found after filtering")
        return None, output

    # Sort by area and select the largest
    sorted_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
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
                ("1. Original Image", img),
                ("2. Grayscale", gray),
                ("3. CLAHE Enhanced", clahe_img),
                ("4. Gaussian Blur", blurred),
                ("5. Canny Edges", edges),
                ("6. Dilated Edges", dilated),
                ("7. Closed Edges", closed),
                ("8. Filled Contours", filled),
                ("9. Filtered Contours", contours_img),
                ("10. Final Detection", output),
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
            cv2.imshow("Card Detection", display_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return quadrilateral, output


"""
Module for detecting and recognizing Magic: the Gathering cards from an image.
author: Timo Ikonen
email: timo.ikonen (at) iki.fi
"""

from itertools import product
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import cv2


class SimplePolygon:
    """Simple polygon class to replace Shapely."""

    def __init__(self, coords):
        self.coords = np.array(coords, dtype=np.float32)
        if len(self.coords) > 0 and not np.allclose(self.coords[0], self.coords[-1]):
            self.coords = np.vstack([self.coords, self.coords[0:1]])
        self._area = None
        self._length = None

    @property
    def exterior_coords(self):
        return self.coords

    @property
    def area(self):
        if self._area is None:
            coords = self.coords[:-1]
            x = coords[:, 0].flatten()
            y = coords[:, 1].flatten()
            self._area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
        return self._area

    @property
    def length(self):
        if self._length is None:
            diffs = np.diff(self.coords, axis=0)
            self._length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        return self._length

    def contains(self, other):
        """Check if this polygon contains another polygon using ray casting."""
        other_coords = other.coords[:-1]
        for point in other_coords:
            if not self._point_inside(point):
                return False
        return True

    def _point_inside(self, point):
        """Ray casting algorithm to check if point is inside polygon."""
        x, y = point
        coords = self.coords[:-1]
        n = len(coords)
        inside = False

        p1x, p1y = coords[0]
        for i in range(1, n + 1):
            p2x, p2y = coords[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def intersection_with_line(self, line_p0, line_p1):
        """Find intersection points between polygon edges and a line."""
        intersections = []
        coords = self.coords[:-1]

        for i in range(len(coords)):
            p0 = coords[i]
            p1 = coords[(i + 1) % len(coords)]

            intersection = line_segment_intersection(
                line_p0[0],
                line_p0[1],
                line_p1[0],
                line_p1[1],
                p0[0],
                p0[1],
                p1[0],
                p1[1],
            )
            if intersection is not None:
                intersections.append(intersection)

        return intersections

    def to_list(self):
        # Get the 4 corner points (excluding the duplicate last point)
        coords = self.coords[:-1]

        # Extract x and y coordinates
        x_coords = coords[:, 0].tolist()
        y_coords = coords[:, 1].tolist()

        return [x_coords, y_coords]


def line_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """Find intersection point of two line segments."""
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None


def order_polygon_points(x, y):
    """Orders polygon points into a counterclockwise order."""
    angle = np.arctan2(y - np.average(y), x - np.average(x))
    ind = np.argsort(angle)
    return (x[ind], y[ind])


def four_point_transform(image, poly):
    """A perspective transform for a quadrilateral polygon."""
    pts = poly.coords[:-1]
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = order_polygon_points(pts[:, 0], pts[:, 1])

    width_a = np.sqrt(
        ((rect[1, 0] - rect[0, 0]) ** 2) + ((rect[1, 1] - rect[0, 1]) ** 2)
    )
    width_b = np.sqrt(
        ((rect[3, 0] - rect[2, 0]) ** 2) + ((rect[3, 1] - rect[2, 1]) ** 2)
    )
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(
        ((rect[0, 0] - rect[3, 0]) ** 2) + ((rect[0, 1] - rect[3, 1]) ** 2)
    )
    height_b = np.sqrt(
        ((rect[1, 0] - rect[2, 0]) ** 2) + ((rect[1, 1] - rect[2, 1]) ** 2)
    )
    max_height = max(int(height_a), int(height_b))

    rect = np.array(
        [
            [rect[0, 0], rect[0, 1]],
            [rect[1, 0], rect[1, 1]],
            [rect[2, 0], rect[2, 1]],
            [rect[3, 0], rect[3, 1]],
        ],
        dtype="float32",
    )

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))
    return warped


def line_intersection(x, y):
    """
    Calculates the intersection point of two lines.
    Returns (nan, nan) if lines are parallel.
    """
    slope_0 = (x[0] - x[1]) * (y[2] - y[3])
    slope_2 = (y[0] - y[1]) * (x[2] - x[3])
    if slope_0 == slope_2:
        return (np.nan, np.nan)

    xy_01 = x[0] * y[1] - y[0] * x[1]
    xy_23 = x[2] * y[3] - y[2] * x[3]
    denom = slope_0 - slope_2

    xis = (xy_01 * (x[2] - x[3]) - (x[0] - x[1]) * xy_23) / denom
    yis = (xy_01 * (y[2] - y[3]) - (y[0] - y[1]) * xy_23) / denom

    return (xis, yis)


def simplify_polygon(in_poly, length_cutoff=0.15, maxiter=None, segment_to_remove=None):
    """
    Removes segments from a convex polygon by continuing neighboring
    segments to a new point of intersection.
    """
    x_in = in_poly.coords[:-1, 0].copy()
    y_in = in_poly.coords[:-1, 1].copy()
    len_poly = len(x_in)
    niter = 0
    if segment_to_remove is not None:
        maxiter = 1

    while len_poly > 4:
        d_in = np.sqrt(
            np.ediff1d(x_in, to_end=x_in[0] - x_in[-1]) ** 2.0
            + np.ediff1d(y_in, to_end=y_in[0] - y_in[-1]) ** 2.0
        )
        d_tot = np.sum(d_in)

        if segment_to_remove is not None:
            k = segment_to_remove
        else:
            k = np.argmin(d_in)

        if d_in[k] < length_cutoff * d_tot:
            ind = generate_point_indices(k - 1, k + 1, len_poly)
            (xis, yis) = line_intersection(x_in[ind], y_in[ind])
            x_in[k] = xis
            y_in[k] = yis
            x_in = np.delete(x_in, (k + 1) % len_poly)
            y_in = np.delete(y_in, (k + 1) % len_poly)
            len_poly = len(x_in)
            niter += 1
            if (maxiter is not None) and (niter >= maxiter):
                break
        else:
            break

    out_poly = SimplePolygon([[ix, iy] for (ix, iy) in zip(x_in, y_in)])
    return out_poly


def generate_point_indices(index_1, index_2, max_len):
    """Returns the four indices for polygon segment endpoints."""
    return np.array(
        [
            index_1 % max_len,
            (index_1 + 1) % max_len,
            index_2 % max_len,
            (index_2 + 1) % max_len,
        ]
    )


def generate_quad_corners(indices, x, y):
    """Returns the four intersection points from the segments."""
    (i, j, k, l) = indices

    def gpi(index_1, index_2):
        return generate_point_indices(index_1, index_2, len(x))

    xis = np.empty(4)
    yis = np.empty(4)
    xis.fill(np.nan)
    yis.fill(np.nan)

    if j <= i or k <= j or l <= k:
        pass
    else:
        (xis[0], yis[0]) = line_intersection(x[gpi(i, j)], y[gpi(i, j)])
        (xis[1], yis[1]) = line_intersection(x[gpi(j, k)], y[gpi(j, k)])
        (xis[2], yis[2]) = line_intersection(x[gpi(k, l)], y[gpi(k, l)])
        (xis[3], yis[3]) = line_intersection(x[gpi(l, i)], y[gpi(l, i)])

    return (xis, yis)


def generate_quad_candidates(in_poly):
    """Generates bounding quadrilaterals for a polygon."""
    (x_s, y_s) = order_polygon_points(in_poly.coords[:-1, 0], in_poly.coords[:-1, 1])
    x_s_ave = np.average(x_s)
    y_s_ave = np.average(y_s)
    x_shrunk = x_s_ave + 0.9999 * (x_s - x_s_ave)
    y_shrunk = y_s_ave + 0.9999 * (y_s - y_s_ave)
    shrunk_poly = SimplePolygon([[x, y] for (x, y) in zip(x_shrunk, y_shrunk)])
    quads = []
    len_poly = len(x_s)

    for indices in product(range(len_poly), repeat=4):
        (xis, yis) = generate_quad_corners(indices, x_s, y_s)
        if (np.sum(np.isnan(xis)) + np.sum(np.isnan(yis))) > 0:
            pass
        else:
            (xis, yis) = order_polygon_points(xis, yis)
            quad = SimplePolygon(
                [(xis[0], yis[0]), (xis[1], yis[1]), (xis[2], yis[2]), (xis[3], yis[3])]
            )
            if quad.contains(shrunk_poly):
                quads.append(quad)
    return quads


def convex_hull_polygon(contour):
    """Returns the convex hull of the given contour as a polygon."""
    hull = cv2.convexHull(contour)
    phull = SimplePolygon([[x, y] for (x, y) in zip(hull[:, :, 0], hull[:, :, 1])])
    return phull


def get_bounding_quad(hull_poly):
    """Returns the minimum area quadrilateral that bounds the convex hull."""
    simple_poly = simplify_polygon(hull_poly)

    # Check if simplification resulted in a valid polygon
    if len(simple_poly.coords) < 5:  # Less than 4 unique points + closing point
        return None

    bounding_quads = generate_quad_candidates(simple_poly)

    # Check if any valid quads were found
    if len(bounding_quads) == 0:
        return None

    bquad_areas = np.zeros(len(bounding_quads))
    for iquad, bquad in enumerate(bounding_quads):
        bquad_areas[iquad] = bquad.area
    min_area_quad = bounding_quads[np.argmin(bquad_areas)]
    return min_area_quad


def quad_corner_diff(hull_poly, bquad_poly, region_size=0.9):
    """
    Returns the difference between areas in the corners of a rounded
    corner and the approximating sharp corner quadrilateral.
    """
    bquad_corners = bquad_poly.coords[:-1]

    interior_points = np.zeros((4, 2))
    interior_points[:, 0] = np.average(bquad_corners[:, 0]) + region_size * (
        bquad_corners[:, 0] - np.average(bquad_corners[:, 0])
    )
    interior_points[:, 1] = np.average(bquad_corners[:, 1]) + region_size * (
        bquad_corners[:, 1] - np.average(bquad_corners[:, 1])
    )

    p0_x = interior_points[:, 0] + (
        bquad_corners[:, 1] - np.average(bquad_corners[:, 1])
    )
    p1_x = interior_points[:, 0] - (
        bquad_corners[:, 1] - np.average(bquad_corners[:, 1])
    )
    p0_y = interior_points[:, 1] - (
        bquad_corners[:, 0] - np.average(bquad_corners[:, 0])
    )
    p1_y = interior_points[:, 1] + (
        bquad_corners[:, 0] - np.average(bquad_corners[:, 0])
    )

    corner_area_polys = []
    for i in range(len(interior_points[:, 0])):
        intersections = bquad_poly.intersection_with_line(
            (p0_x[i], p0_y[i]), (p1_x[i], p1_y[i])
        )
        if len(intersections) >= 2:
            corner_area_polys.append(
                SimplePolygon(
                    [
                        intersections[0],
                        intersections[1],
                        (bquad_corners[i, 0], bquad_corners[i, 1]),
                    ]
                )
            )

    hull_corner_area = 0
    quad_corner_area = 0
    for capoly in corner_area_polys:
        quad_corner_area += capoly.area
        # Approximate intersection by checking if hull contains corner poly points
        hull_coords = hull_poly.coords[:-1]
        contained_area = 0
        for point in capoly.coords[:-1]:
            if hull_poly._point_inside(point):
                contained_area += capoly.area / len(capoly.coords[:-1])
        hull_corner_area += contained_area

    if quad_corner_area == 0:
        return 1.0
    return 1.0 - hull_corner_area / quad_corner_area


def polygon_form_factor(poly):
    """
    The ratio between the polygon area and circumference length,
    scaled by the length of the shortest segment.
    """
    d_0 = np.amin(np.sqrt(np.sum(np.diff(poly.coords, axis=0) ** 2.0, axis=1)))
    return poly.area / (poly.length * d_0)


def detect_card_clahe(
    image_path, display=True, show_steps=True
) -> tuple[Optional[SimplePolygon], Optional[np.ndarray]]:
    """Detect card in image using CLAHE transformation."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness, redness, yellowness = cv2.split(lab)
    corrected_lightness = clahe.apply(lightness)
    limg = cv2.merge((corrected_lightness, redness, yellowness))
    adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    if show_steps:
        plt.imshow(thresh)
        plt.show()

    contours, _ = cv2.findContours(
        np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_area = img.shape[0] * img.shape[1]

    for card_contour in contours:
        phull = convex_hull_polygon(card_contour)
        if phull.area < min(0.001, image_area / 1000.0):
            continue

        bounding_poly = get_bounding_quad(phull)
        if bounding_poly is None:
            continue

        qc_diff = quad_corner_diff(phull, bounding_poly)
        is_card_candidate = bool(
            0.001 < bounding_poly.area < image_area * 0.99
            and qc_diff < 0.35
            and 0.25 < polygon_form_factor(bounding_poly) < 0.33
        )

        if is_card_candidate:
            output = img.copy()
            cv2.drawContours(output, [bounding_poly.to_list()], 0, (0, 255, 0), 3)
            return (bounding_poly.to_list(), output)

    return (None, None)


if __name__ == "__main__":
    for bg in range(9):
        image_path = f"./data/generations/c1bg{bg}.png"
        detect_card_edges_with_clahe(image_path, display=True, show_steps=True)

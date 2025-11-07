"""
create_test_images.py

creates test images to then train and assess the network on.
dumps data into:
    - ./data/generations for data
    - ./data/generations.json for labels
"""

import cv2
import json
import numpy as np
import random
import requests

from concurrent import futures
from pathlib import Path
from typing import Optional


def collect_test_cards_from_scryfall(
    total_cards: int = 100,
) -> list[tuple[np.ndarray, str, str]]:
    """
    collects `total_cards` random card images from scryfall.

    each one includes the card image of shape (height, width, 4) where the 4th channel is png alpha,
    card name and set to later be able to find it again.
    """

    SETS = ['fut', 'uma', 'rna', 'war', 'ktk', 'm19', 'otj', 'blb']

    url = f'https://api.scryfall.com/cards/search?q=' + (
        ' or '.join(f'set:{s}' for s in SETS)
    )
    response = requests.get(
        url, headers={'User-Agent': 'mtg-match/0.1', 'Accept': '*/*'}
    )
    data = response.json()

    candidates = []

    def process_data(data: list[dict[str, str]]):
        for group in data:
            try:
                candidates.append(
                    [group['image_uris']['png'], group['name'], group['set']]  # type: ignore
                )
            except KeyError:
                pass

    process_data(data['data'])

    while data.get('has_more', False):
        url = data['next_page']
        response = requests.get(
            url, headers={'User-Agent': 'mtg-match/0.1', 'Accept': '*/*'}
        )
        data = response.json()
        process_data(data['data'])

    print(f'found {len(candidates)} total cards')

    if total_cards < len(candidates):
        candidates = random.sample(candidates, total_cards)

    def collect_test_card(
        image_url, card_name, card_set
    ) -> tuple[np.ndarray, str, str]:
        """collect a card for testing from scryfall"""
        # load the card image into a numpy array4
        img_response = requests.get(image_url)
        img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
        img: np.ndarray = cv2.imdecode(
            img_array, cv2.IMREAD_UNCHANGED
        )   # type: ignore
        # img has shape (height, width, 4) where 4th channel is alpha
        # (becasue it's a png and supports clear in the corners)

        return (img, card_name, card_set)

    tasks: list[futures.Future[tuple[np.ndarray, str, str]]] = []
    result: list[tuple[np.ndarray, str, str]] = []

    # process with threads to make this faster
    with futures.ThreadPoolExecutor() as exc:
        for group in candidates:
            tasks.append(exc.submit(collect_test_card, *group))

    for task in futures.as_completed(tasks):
        group = task.result()
        if group[1] == '':
            continue
        result.append(group)
        print(
            f'collected [{len(result):03d}/{total_cards:03d}] test cards',
            end='\r',
        )

    print('')

    return result


def get_rotated_bounding_box(image_shape, angle, center):
    """
    Calculate the bounding box size needed to fit a rotated image.

    Args:
        image_shape: (height, width) of original image
        angle: Rotation angle in degrees
        center: Rotation center (x, y)

    Returns:
        new_width, new_height: Dimensions of bounding box
    """
    h, w = image_shape[:2]

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return new_w, new_h, M


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation to points.

    Args:
        points: Array of shape (N, 2) containing (x, y) coordinates
        matrix: 2x3 affine transformation matrix

    Returns:
        transformed_points: Array of shape (N, 2)
    """
    points = np.array(points, dtype=np.float32)

    # Add homogeneous coordinate
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])

    # Apply transformation
    transformed = matrix @ points_homogeneous.T

    return transformed.T


def overlay_png_with_alpha(
    card_img: np.ndarray, background: np.ndarray, x_offset: int, y_offset: int
) -> np.ndarray:
    """
    Overlay a PNG with transparency onto a background.
    """
    h, w = card_img.shape[:2]

    if card_img.shape[2] != 4:
        # No alpha channel, simple paste
        background[y_offset : y_offset + h, x_offset : x_offset + w] = card_img
        return background

    # Split channels
    card_bgr = card_img[:, :, :3]
    alpha = card_img[:, :, 3:4] / 255.0

    bg_region = background[
        y_offset : y_offset + h, x_offset : x_offset + w, :3
    ]  # Only first 3 channels

    # Vectorized alpha blending
    blended = (alpha * card_bgr + (1 - alpha) * bg_region).astype(np.uint8)
    background[
        y_offset : y_offset + h, x_offset : x_offset + w, :3
    ] = blended  # Write back to first 3 channels

    return background


def augment_card_with_corners(
    card_img: np.ndarray,
    background_img: Optional[np.ndarray] = None,
    rotation_range: tuple[float, float] = (-30, 30),
    scale_range: tuple[float, float] = (0.5, 1.5),
    margin: float = 50,
    keystone_range: tuple[float, float] = (0, 0.15),
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Apply transformations to card and composite onto background.
    Returns the four corner positions for ground truth labels.

    Args:
        card_img: Card image (H, W, 4)
        background_img: Optional background image (picks random color otherwise)
        rotation_range: (min_angle, max_angle) in degrees
        scale_range: (min_scale, max_scale)
        margin: Minimum pixel margin from edges when ensure_fully_visible=True
        keystone_range: (min_strength, max_strength) for perspective warp.
                       0 = no warp, 0.15 = moderate perspective distortion.
                       Each corner can move up to strength * dimension pixels.

    Returns:
        result_img: Final composite image (H, W, 3) !! FLATTENS THE ALPHA CHANNEL !!
        corners: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                 in order: top-left, top-right, bottom-left, bottom-right
    """
    h, w = card_img.shape[:2]

    # Random transformations
    angle = np.random.uniform(*rotation_range)
    scale = np.random.uniform(*scale_range)
    keystone_strength = np.random.uniform(*keystone_range)

    # Initialize corners in original card space
    corners = np.array(
        [
            [0, 0],  # top-left
            [w, 0],  # top-right
            [0, h],  # bottom-left
            [w, h],  # bottom-right
        ],
        dtype=np.float32,
    )

    # Get rotation matrix and new dimensions
    center = (w // 2, h // 2)
    new_w, new_h, rotation_matrix = get_rotated_bounding_box(
        (h, w), angle, center
    )

    # Apply rotation to image
    rotated = cv2.warpAffine(
        card_img,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Apply rotation to corners
    # Convert to homogeneous coordinates for affine transform
    corners_homogeneous = np.hstack([corners, np.ones((4, 1))])
    rotated_corners = (rotation_matrix @ corners_homogeneous.T).T

    # Apply scaling to image
    scaled_w = int(new_w * scale)
    scaled_h = int(new_h * scale)
    scaled = cv2.resize(
        rotated, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR
    )

    # Apply scaling to corners
    scale_x = scaled_w / new_w
    scale_y = scaled_h / new_h
    scaled_corners = rotated_corners * [scale_x, scale_y]

    # Apply keystone warping (perspective transformation)
    if keystone_strength > 0:
        # Source corners (current rectangle bounds)
        src_corners = np.array(
            [
                [0, 0],  # top-left
                [scaled_w, 0],  # top-right
                [0, scaled_h],  # bottom-left
                [scaled_w, scaled_h],  # bottom-right
            ],
            dtype=np.float32,
        )

        # Destination corners (warped positions)
        # Each corner randomly displaced within keystone_strength
        max_shift_x = scaled_w * keystone_strength
        max_shift_y = scaled_h * keystone_strength

        dst_corners = src_corners.copy()
        for i in range(4):
            dst_corners[i, 0] += np.random.uniform(-max_shift_x, max_shift_x)
            dst_corners[i, 1] += np.random.uniform(-max_shift_y, max_shift_y)

        # Compute perspective transform
        perspective_matrix = cv2.getPerspectiveTransform(
            src_corners, dst_corners
        )

        # Calculate bounds of warped image
        all_x = dst_corners[:, 0]
        all_y = dst_corners[:, 1]
        min_x, max_x = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
        min_y, max_y = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))

        # Adjust transform to account for negative coordinates
        translation = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32
        )
        adjusted_matrix = translation @ perspective_matrix

        # Apply perspective warp to image
        warped_w = max_x - min_x
        warped_h = max_y - min_y
        scaled = cv2.warpPerspective(
            scaled,
            adjusted_matrix,
            (warped_w, warped_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        # Apply perspective transform to corners
        corners_homogeneous = np.hstack([scaled_corners, np.ones((4, 1))])
        warped_corners_homogeneous = (adjusted_matrix @ corners_homogeneous.T).T
        # Convert from homogeneous coordinates
        warped_corners = warped_corners_homogeneous[:, :2] / warped_corners_homogeneous[:, 2:]

        # Update dimensions and corners
        scaled_w, scaled_h = warped_w, warped_h
        scaled_corners = warped_corners
    # else: scaled_corners already computed above

    # Create or use background
    if background_img is None:
        bg_h, bg_w = 1000, 1000
        color = np.random.randint(50, 200, 4).tolist()
        background = np.ones((bg_h, bg_w, 4), dtype=np.uint8) * color
    else:
        background = background_img.copy()
        bg_h, bg_w = background.shape[:2]

    # Determine position on background
    max_x = bg_w - scaled_w - margin
    max_y = bg_h - scaled_h - margin

    if max_x < margin or max_y < margin:
        # Card is too big for background, scale it down
        scale_factor = min(
            (bg_w - 2 * margin) / scaled_w, (bg_h - 2 * margin) / scaled_h
        )
        scaled_w_new = int(scaled_w * scale_factor)
        scaled_h_new = int(scaled_h * scale_factor)
        scaled = cv2.resize(scaled, (scaled_w_new, scaled_h_new))

        # Rescale corners proportionally
        center_x, center_y = scaled_w / 2, scaled_h / 2
        scaled_corners = (
            scaled_corners - [center_x, center_y]
        ) * scale_factor + [scaled_w_new / 2, scaled_h_new / 2]

        scaled_w, scaled_h = scaled_w_new, scaled_h_new

        max_x = bg_w - scaled_w - margin
        max_y = bg_h - scaled_h - margin

    x = np.random.randint(int(margin), int(max(margin + 1, max_x + 1)))
    y = np.random.randint(int(margin), int(max(margin + 1, max_y + 1)))

    # Overlay the card on the background and translate corners
    result = overlay_png_with_alpha(scaled, background, x, y)
    final_corners = scaled_corners + [x, y]

    return result.astype(np.uint8), final_corners.tolist()


def load_backgrounds(background_path: str) -> list[np.ndarray]:
    """loads all of the image backgrounds"""
    data_dir = Path(background_path)

    if not data_dir.exists():
        print(f'Error: Directory {background_path} does not exist')
        return []

    image_paths = [
        f
        for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() == '.png'
    ]

    return [
        cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths
    ]   # type: ignore


if __name__ == '__main__':
    cards: list[
        tuple[np.ndarray, str, str]
    ] = collect_test_cards_from_scryfall()
    card = cards[0][0]
    backgrounds = load_backgrounds('./data/backgrounds')
    background = backgrounds[0]

    output = {}

    card_count = 0
    for card_img, card_name, card_set in cards:
        card_count += 1

        for background_count, background in enumerate(backgrounds):
            result, corners = augment_card_with_corners(
                card_img,
                background,
                margin=50,
            )

            filename = f'c{card_count}bg{background_count}'

            output[filename] = {
                'corners': corners,
                'name': card_name,
                'set': card_set,
            }
            print(
                f'created test image: {filename}',
                end='\r',
            )

            cv2.imwrite(f'./data/generations/{filename}.png', result)

    print('')

    with open('./data/generations.json', 'w') as fp:
        json.dump(output, fp, indent=2)

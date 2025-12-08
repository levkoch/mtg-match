import cv2 as cv
import numpy as np
from pathlib import Path
from collections import deque

from build_database import Card
from match_cards_phash import CardMatcher

base_path = Path(__file__).parent.resolve().as_posix()

matcher = CardMatcher(f'{base_path}/data/card_database_phash.json')
matcher.match_filter.set_threshold(0.2)

cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get frame dimensions
ret, test_frame = cap.read()
if ret:
    frame_h, frame_w = test_frame.shape[:2]
else:
    frame_h, frame_w = 720, 1280  # Default fallback

last_card = None       
last_top_matches = None    
recent_confidences = deque(maxlen=10)  


while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    result = frame.copy()

    temp_path = "./temp_frame.jpg"
    cv.imwrite(temp_path, frame)

    card, normalized, corners, top_matches = matcher.process_image(
        temp_path, display=False, match_func='combined', top_k=3, use_filter=True
    )

    if top_matches is not None and len(top_matches) > 0:
        new_conf = top_matches[0]["score"]
    else:
        new_conf = None

    if new_conf is not None:
        recent_confidences.append(new_conf)

    avg_conf = (sum(recent_confidences) / len(recent_confidences)) if recent_confidences else 0.0

    should_update = False
    if card is not None and new_conf is not None:
        if last_card is None:
            should_update = True  
        elif new_conf > avg_conf:
            should_update = True

    if should_update:
        last_card = card
        last_top_matches = top_matches

    stable_card = last_card
    stable_matches = last_top_matches

    if corners is not None:
        corners_int = corners.astype(int)
        cv.drawContours(result, [corners_int], 0, (0, 255, 0), 3)
        for point in corners_int:
            cv.circle(result, tuple(point), 8, (0, 0, 255), -1)


    if normalized is not None:
        right_margin = 40
        top_margin = 120
        spacing = 40

        display_h = 400
        display_w = int(normalized.shape[1] * (display_h / normalized.shape[0]))
        normalized_small = cv.resize(normalized, (display_w, display_h))

        norm_y = top_margin
        norm_x = frame_w - display_w - right_margin

        if norm_y + display_h <= frame_h and norm_x >= 0:
            result[norm_y:norm_y + display_h, norm_x:norm_x + display_w] = normalized_small
            cv.rectangle(result, (norm_x, norm_y),
                         (norm_x + display_w, norm_y + display_h), (255, 255, 255), 2)

    # If no stable card yet, tell user
    if stable_card is None:
        cv.putText(result, "No Stable Match Yet", (frame_w - 350, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        # Draw stable card info + current geometry positions
        card = stable_card
        text_y = norm_y + display_h + 40 + 30  # spacing + offset
        text_x = frame_w - 300 - right_margin

        # Background for text
        cv.rectangle(result, (text_x - 10, text_y - 30),
                     (frame_w - right_margin, text_y + 60), (0, 0, 0), -1)
        cv.rectangle(result, (text_x - 10, text_y - 30),
                     (frame_w - right_margin, text_y + 60), (255, 255, 255), 2)

        cv.putText(result, card.name, (text_x, text_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.putText(result, f"{card.set_code} #{card.collector_num}", (text_x, text_y + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv.putText(result,
                   f"Conf: {stable_matches[0]['score']:.2f}  "
                   f"Filter: {stable_matches[0]['filter_confidence']:.3f}",
                   (text_x, text_y + 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw reference image (stable identity)
        ref_image = cv.imread(f'{base_path}/data/card_images/{card.card_key}.png')
        if ref_image is not None:
            ref_h = 400
            ref_w = int(ref_image.shape[1] * (ref_h / ref_image.shape[0]))
            ref_small = cv.resize(ref_image, (ref_w, ref_h))

            ref_y = text_y + 80
            ref_x = frame_w - ref_w - right_margin

            if ref_y + ref_h <= frame_h and ref_x >= 0:
                result[ref_y:ref_y + ref_h, ref_x:ref_x + ref_w] = ref_small
                cv.rectangle(result, (ref_x, ref_y),
                             (ref_x + ref_w, ref_y + ref_h), (255, 255, 255), 2)
                cv.putText(result, "Reference", (ref_x, ref_y - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv.imshow('Mtg-Match Detection', result)

    if cv.waitKey(1) == ord('q'):
        break

# Cleanup
Path("./temp_frame.jpg").unlink(missing_ok=True)
cap.release()
cv.destroyAllWindows()

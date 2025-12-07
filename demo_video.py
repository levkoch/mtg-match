import cv2 as cv
import numpy as np
from pathlib import Path
from match_cards_phash import CardMatcher

base_path = Path(__file__).parent.resolve().as_posix()

matcher = CardMatcher(f'{base_path}/data/card_database_phash.json')
card_image_cache = {}

def get_card_reference_image(card_name, card_set):
    """Load reference image for a card from the database."""
    cache_key = (card_name, card_set)
    
    if cache_key in card_image_cache:
        return card_image_cache[cache_key]
    
    norm_name = (
            card_name.replace(" ", "_")
            .replace(",", "")
            .replace(":", "")
            .replace("/", "")
            .replace("'", "")
            .replace('"', "")
        )
    card_key = f"{card_set}_{norm_name}"
    
    card_path = Path(f"{base_path}/data/card_images/{card_key}.png")
    
    if card_path.exists():
        img = cv.imread(str(card_path))
        card_image_cache[cache_key] = img
        return img
    
    # If not found, return a placeholder
    placeholder = np.zeros((680, 488, 3), dtype=np.uint8)
    cv.putText(placeholder, "No Image", (150, 340), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    card_image_cache[cache_key] = placeholder
    return placeholder

cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get frame dimensions
ret, test_frame = cap.read()
if ret:
    frame_h, frame_w = test_frame.shape[:2]
else:
    frame_h, frame_w = 720, 1280  # Default

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Create result frame (copy of original)
    result = frame.copy()
    
    # Detect card using the matcher's detection method
    # Save frame temporarily to disk for processing
    temp_path = "./temp_frame.jpg"
    cv.imwrite(temp_path, frame)
    
    try:
        # Detect and extract card
        normalized_card, corners = matcher.detect_and_extract_card(temp_path)
        
        if normalized_card is not None and corners is not None:
            # Draw corners on the live feed
            corners_int = corners.astype(int)
            cv.drawContours(result, [corners_int], 0, (0, 255, 0), 3)
            for point in corners_int:
                cv.circle(result, tuple(point), 8, (0, 0, 255), -1)
            
            # Match the card
            top_matches = matcher.match_perceptual_hash(normalized_card, top_k=1)
            confidence = 0.0
            
            if top_matches and len(top_matches) > 0:
                card_name, card_set = top_matches[0]['card']        
                confidence = top_matches[0]['score']
                                                                             
            if confidence > 0.80:
                # Calculate layout positions (right side of frame)
                right_margin = 40
                top_margin = 40
                spacing = 40
                
                # Resize normalized card for display (smaller)
                display_h = 400
                display_w = int(normalized_card.shape[1] * (display_h / normalized_card.shape[0]))
                normalized_small = cv.resize(normalized_card, (display_w, display_h))
                
                # Position for normalized card (top right)
                norm_y = top_margin
                norm_x = frame_w - display_w - right_margin
                
                # Overlay normalized card
                if norm_y + display_h <= frame_h and norm_x >= 0:
                    result[norm_y:norm_y+display_h, norm_x:norm_x+display_w] = normalized_small
                    # Add border
                    cv.rectangle(result, (norm_x, norm_y), 
                               (norm_x+display_w, norm_y+display_h), (255, 255, 255), 2)
                
                # Card name text (center right)
                text_y = norm_y + display_h + spacing + 30
                text_x = frame_w - 300 - right_margin
                
                # Background for text
                cv.rectangle(result, (text_x - 10, text_y - 30), 
                           (frame_w - right_margin, text_y + 60), (0, 0, 0), -1)
                cv.rectangle(result, (text_x - 10, text_y - 30), 
                           (frame_w - right_margin, text_y + 60), (255, 255, 255), 2)
                
                # Draw card name
                cv.putText(result, card_name, (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv.putText(result, f"Set: {card_set}", (text_x, text_y + 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv.putText(result, f"Conf: {confidence:.2f}", (text_x, text_y + 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Reference card image (bottom right)
                ref_image = get_card_reference_image(card_name, card_set)
                ref_h = 400
                ref_w = int(ref_image.shape[1] * (ref_h / ref_image.shape[0]))
                ref_small = cv.resize(ref_image, (ref_w, ref_h))
                
                ref_y = text_y + 80
                ref_x = frame_w - ref_w - right_margin
                
                if ref_y + ref_h <= frame_h and ref_x >= 0:
                    result[ref_y:ref_y+ref_h, ref_x:ref_x+ref_w] = ref_small
                    cv.rectangle(result, (ref_x, ref_y), 
                               (ref_x+ref_w, ref_y+ref_h), (255, 255, 255), 2)
                    cv.putText(result, "Reference", (ref_x, ref_y - 5), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                # Card detected but not matched or detected with low confidence
                cv.putText(result, "Card Detected", (frame_w - 250, 50), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(result, "No Match Found", (frame_w - 250, 80), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            # No card detected
            cv.putText(result, "No Card Detected", (frame_w - 250, 50), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    except Exception as e:
        # Handle any errors gracefully
        cv.putText(result, f"Error: {str(e)[:30]}", (20, 50), 
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv.imshow('Mtg-Match Detection', result)

    # Break the loop when 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Cleanup
Path("./temp_frame.jpg").unlink(missing_ok=True)
cap.release()
cv.destroyAllWindows()
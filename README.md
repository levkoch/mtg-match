# CSE 445 Final Project

## Data
bulk download from https://scryfall.com/docs/api/images with information about the card set, name and unique id.

## Processing Pipeline
- Detect card image from background (single card with no overlapping)
- Normalize card image and match database shape
- Processing match with database options
- A: Compute global descriptor and match against database
- B: Detect text areas inside image and match against database
- C: Compute perceptual hash for images
- Output prediction
- Add in live video stream view for each segment, notifying user when we get a match.

## Inspiration
- https://tmikonen.github.io/quantitatively/2020-01-01-magic-card-detector/
- https://github.com/hj3yoo/mtg_card_detector/tree/master

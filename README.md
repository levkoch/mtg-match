# CSE 445 Final Project
Lev Kochergin & Riyosha Sharma 

## How To Run
1. Download the card image dataset into `data/card_images/` **!! (56.31 GB on disk) for 85,577 items !!**. This will also calculate perceptual hash and color histograms for the card printings into `data/card_database_phash.json`.
```bash
python build_database.py
```

2. Create test images for testing into `data/generations/` and the corresponding descriptor file in `data/generations.json`.
```bash
python create_test_images.py
```

3. Train Random Forest false positive detector into `data/match_filter_model.pkl`.
```bash
python match_forest.py
```

4. Run video matching demo.
```bash
python demo_video.py
```

You can view our analysis and findings in `writing/CSE 455 MTG MATCH.pdf`.
import json
import threading
from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from match_cards_phash import CardMatcher


def main_filter_fp():
    """Test the complete pipeline on generated test images (parallelized)."""
    matcher = CardMatcher()

    if len(matcher.database) == 0:
        print("No database loaded!")
        return

    # Get test images in proper order
    generations_dir = Path("./data/generations")
    test_images = sorted(generations_dir.glob("*.png"))

    if not test_images:
        print("No test images found in ./data/generations/")
        print("Run create_test_images.py first!")
        return

    print(f"Testing combined hash pipeline on {len(test_images)} generated images...")

    # Load generations metadata for ground truth
    metadata_path = Path("./data/generations.json")
    corners_dict = {} 
    true_keys = {}

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            for k, v in metadata.items():
                corners_dict[k] = np.array(v["corners"], dtype=np.float32)
                true_keys[k] = v['card_key']
    else:
        print("Warning: No generations.json found for ground truth comparison")

    # ---- NEW: parallel processing function ----
    def process_one(img_path):
        card, normalized, corners, top_matches = matcher.process_image(
            str(img_path), display=False, match_func='combined', top_k=5
        )

        true_key = true_keys.get(img_path.stem)

        correct = card is not None and card.card_key == true_key
        correct_in_top5 = any(
            m["card"].card_key == true_key for m in top_matches
        ) if top_matches else False

        return {
            "image": img_path.stem,
            "true_key": true_key,
            "matched": int(card is not None),
            "card": card,
            "correct": int(correct),
            "correct_in_top5": int(correct_in_top5),
            "top_matches": top_matches
        }

    # ---- NEW: run in thread pool with progress tracking ----
    results = []
    total = len(test_images)
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one, img): img for img in test_images}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            with lock:
                completed += 1
                print(f"[{completed}/{total}] processed", end="\r")

    print("\nAll images processed.")

    # ---- Save results as before ----
    matches_fp = Path(f"./data/match_results_fp.json")
    matches_json = {}

    for idx, match in enumerate(results):
        if match['card'] is not None:
            item = deepcopy(match)
            item['card'] = match['card'].to_dict()
            for i in range(len(match['top_matches'])):
                item['top_matches'][i]['card'] = match['top_matches'][i]['card'].to_dict()
        else:
            item = match

        matches_json[idx] = item

    with open(matches_fp, 'w', encoding="utf-8") as f:
        json.dump(matches_json, f, indent=2)

    # ---- Print summary ----
    correct_count = sum(r["correct"] for r in results)
    correct_in_top5_count = sum(r["correct_in_top5"] for r in results)

    print(f"\nSUMMARY:")
    print(f"  Overall accuracy: {correct_count}/{total} correct ({correct_count/total*100:.1f}%)")
    print(f"  Overall Top-5:    {correct_in_top5_count}/{total} ({correct_in_top5_count/total*100:.1f}%)")


def main_assess_fp():
    # Load the data
    with open('./data/match_results_fp.json', 'r') as f:
        data = json.load(f)

    # Analyze the results
    results = {
        'true_positives': [],  # Matched and correct
        'false_positives': [],  # Matched but incorrect
        'false_negatives': [],  # Not matched (should have been)
        'true_negatives': []   # Not matched (correctly rejected)
    }

    all_matches = []  # For scatter plot
    top1_matches = []  # For decision boundary learning

    for idx, entry in data.items():
        true_key = entry['true_key']
        matched = entry['matched']
        correct = entry['correct']
        
        if matched and correct:
            results['true_positives'].append(entry)
        elif matched and not correct:
            results['false_positives'].append(entry)
        elif not matched:
            results['false_negatives'].append(entry)

        # no way to make a true negative, as all of the test images 
        # do actually have a card in them somewhere.
        
        # Collect all top matches for analysis
        if entry['top_matches']:
            for match in entry['top_matches']:
                is_correct = match['card']['card_key'] == true_key
                match_data = {
                    'hash_distance': match['hash_distance'],
                    'hist_score': match['hist_score'],
                    'combined_score': match['combined_score'],
                    'is_correct': is_correct,
                    'rank': entry['top_matches'].index(match) + 1
                }
                all_matches.append(match_data)
                
                # Collect top-1 matches for decision boundary
                if entry['top_matches'].index(match) == 0:
                    top1_matches.append(match_data)

    # Print summary statistics
    print("=" * 60)
    print("MATCH RESULTS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal samples: {len(data)}")
    print(f"True Positives (TP): {len(results['true_positives'])}")
    print(f"False Positives (FP): {len(results['false_positives'])}")
    print(f"False Negatives (FN): {len(results['false_negatives'])}")
    print(f"True Negatives (TN): {len(results['true_negatives'])}")

    if len(results['true_positives']) + len(results['false_positives']) > 0:
        precision = len(results['true_positives']) / (len(results['true_positives']) + len(results['false_positives']))
        print(f"\nPrecision: {precision:.2%}")

    if len(results['true_positives']) + len(results['false_negatives']) > 0:
        recall = len(results['true_positives']) / (len(results['true_positives']) + len(results['false_negatives']))
        print(f"Recall: {recall:.2%}")

    # Analyze score distributions
    correct_matches = [m for m in top1_matches if m['is_correct']]
    incorrect_matches = [m for m in top1_matches if not m['is_correct']]

    print(f"\n" + "=" * 60)
    print("SCORE DISTRIBUTIONS (All Top-5 Matches)")
    print("=" * 60)

    if correct_matches:
        print("\nCORRECT MATCHES:")
        print(f"  Hash Distance: {np.mean([m['hash_distance'] for m in correct_matches]):.4f} ± {np.std([m['hash_distance'] for m in correct_matches]):.4f}")
        print(f"  Hist Score: {np.mean([m['hist_score'] for m in correct_matches]):.4f} ± {np.std([m['hist_score'] for m in correct_matches]):.4f}")
        print(f"  Combined Score: {np.mean([m['combined_score'] for m in correct_matches]):.4f} ± {np.std([m['combined_score'] for m in correct_matches]):.4f}")

    if incorrect_matches:
        print("\nINCORRECT MATCHES:")
        print(f"  Hash Distance: {np.mean([m['hash_distance'] for m in incorrect_matches]):.4f} ± {np.std([m['hash_distance'] for m in incorrect_matches]):.4f}")
        print(f"  Hist Score: {np.mean([m['hist_score'] for m in incorrect_matches]):.4f} ± {np.std([m['hist_score'] for m in incorrect_matches]):.4f}")
        print(f"  Combined Score: {np.mean([m['combined_score'] for m in incorrect_matches]):.4f} ± {np.std([m['combined_score'] for m in incorrect_matches]):.4f}")

    # Learn linear decision boundary
    print(f"\n" + "=" * 60)
    print("LINEAR DECISION BOUNDARY LEARNING")
    print("=" * 60)

    if len(all_matches) > 1:
        # Prepare training data
        X = np.array([[m['hash_distance'], m['hist_score']] for m in all_matches])
        y = np.array([1 if m['is_correct'] else 0 for m in all_matches])
        
        # Train logistic regression (linear decision boundary)
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)
        
        # Get coefficients
        w1, w2 = clf.coef_[0]
        b = clf.intercept_[0]
        
        print(f"\nDecision Boundary Equation:")
        print(f"  {w1:.4f} * hash_distance + {w2:.4f} * hist_score + {b:.4f} = 0")
        print(f"\nSimplified form:")
        print(f"  hist_score = {-w1/w2:.4f} * hash_distance + {-b/w2:.4f}")
        
        # Predict on training data
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        
        print(f"\nClassification Report (All Top-5 Matches):")
        print(classification_report(y, y_pred, target_names=['Incorrect', 'Correct']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y, y_pred)
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        # Calculate accuracy if we only accept predictions above different thresholds
        print(f"\n" + "=" * 60)
        print("THRESHOLD ANALYSIS (for filtering false positives)")
        print("=" * 60)
        
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        print(f"\n{'Threshold':<12} {'Accepted':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'FP Rate':<10}")
        print("-" * 70)
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            accepted = np.sum(y_pred_thresh)
            
            if accepted > 0:
                correct_accepted = np.sum((y_pred_thresh == 1) & (y == 1))
                accuracy = correct_accepted / accepted if accepted > 0 else 0
                
                tp = np.sum((y_pred_thresh == 1) & (y == 1))
                fp = np.sum((y_pred_thresh == 1) & (y == 0))
                fn = np.sum((y_pred_thresh == 0) & (y == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                fp_rate = fp / (fp + np.sum((y_pred_thresh == 0) & (y == 0))) if (fp + np.sum((y_pred_thresh == 0) & (y == 0))) > 0 else 0
                
                print(f"{threshold:<12.2f} {accepted:<10} {accuracy:<10.2%} {precision:<10.2%} {recall:<10.2%} {fp_rate:<10.2%}")
            else:
                print(f"{threshold:<12.2f} {accepted:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        
        # Create visualizations
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Decision boundary visualization
        ax1 = plt.subplot(2, 3, 1)
        
        # Create mesh for decision boundary
        h = 0.001
        x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
        y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = ax1.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'green'])
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        
        # Plot points
        if correct_matches:
            ax1.scatter([m['hash_distance'] for m in correct_matches], 
                        [m['hist_score'] for m in correct_matches],
                        c='green', alpha=0.8, s=100, label='Correct', edgecolors='black', linewidths=1.5)
        if incorrect_matches:
            ax1.scatter([m['hash_distance'] for m in incorrect_matches], 
                        [m['hist_score'] for m in incorrect_matches],
                        c='red', alpha=0.8, s=100, label='Incorrect', edgecolors='black', linewidths=1.5)
        
        ax1.set_xlabel('Hash Distance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Histogram Score', fontsize=12, fontweight='bold')
        ax1.set_title('Linear Decision Boundary', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curve
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Probability distribution
        ax3 = plt.subplot(2, 3, 3)
        correct_probs = y_prob[y == 1]
        incorrect_probs = y_prob[y == 0]
        
        if len(correct_probs) > 0:
            ax3.hist(correct_probs, bins=40, alpha=0.7, range=(0,1), color='green', label='Correct', edgecolor='black')
        if len(incorrect_probs) > 0:
            ax3.hist(incorrect_probs, bins=40, alpha=0.7, range=(0,1), color='red', label='Incorrect', edgecolor='black')
        
        ax3.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold')
        ax3.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency (log)', fontsize=12, fontweight='bold')
        ax3.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Hash Distance distribution
        ax4 = plt.subplot(2, 3, 4)
        if correct_matches:
            ax4.hist([m['hash_distance'] for m in correct_matches], 
                    bins=20, alpha=0.7, color='green', label='Correct', edgecolor='black')
        if incorrect_matches:
            ax4.hist([m['hash_distance'] for m in incorrect_matches], 
                    bins=20, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
        ax4.set_xlabel('Hash Distance', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Hash Distance Distribution', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Histogram Score distribution
        ax5 = plt.subplot(2, 3, 5)
        if correct_matches:
            ax5.hist([m['hist_score'] for m in correct_matches], 
                    bins=20, alpha=0.7, color='green', label='Correct', edgecolor='black')
        if incorrect_matches:
            ax5.hist([m['hist_score'] for m in incorrect_matches], 
                    bins=20, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
        ax5.set_xlabel('Histogram Score', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax5.set_title('Histogram Score Distribution', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Combined Score distribution
        ax6 = plt.subplot(2, 3, 6)
        if correct_matches:
            ax6.hist([m['combined_score'] for m in correct_matches], 
                    bins=20, alpha=0.7, color='green', label='Correct', edgecolor='black')
        if incorrect_matches:
            ax6.hist([m['combined_score'] for m in incorrect_matches], 
                    bins=20, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
        ax6.set_xlabel('Combined Score', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax6.set_title('Combined Score Distribution', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('match_analysis_with_boundary.png', dpi=300, bbox_inches='tight')
        print("\n" + "=" * 60)
        print("Visualization saved as 'match_analysis_with_boundary.png'")
        print("=" * 60)
        plt.show()
        
        # Save the model parameters for future use
        print(f"\n" + "=" * 60)
        print("IMPLEMENTATION CODE")
        print("=" * 60)
        print("\nUse this code to filter matches in your pipeline:")
        print(f"""
    def should_accept_match(hash_distance, hist_score, threshold=0.8):
        z = {w1:.6f} * hash_distance + {w2:.6f} * hist_score + {b:.6f}
        probability = 1 / (1 + np.exp(-z))
        return probability >= threshold
    """)

    else:
        print("\nNot enough data to learn decision boundary")

if __name__ == "__main__":
    # main_filter_fp()
    main_assess_fp()

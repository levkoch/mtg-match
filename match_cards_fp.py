import json
import threading
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from match_cards_phash import CardMatcher


def main_filter_fp():
    """Test the complete pipeline on the generated images for false positive filtering."""
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

    correct_count = sum(r["correct"] for r in results)
    correct_in_top5_count = sum(r["correct_in_top5"] for r in results)

    print(f"\nSUMMARY:")
    print(f"  Overall accuracy: {correct_count}/{total} correct ({correct_count/total*100:.1f}%)")
    print(f"  Overall Top-5:    {correct_in_top5_count}/{total} ({correct_in_top5_count/total*100:.1f}%)")


def extract_features(top_matches, true_key):
    """Extract rich features from top matches including score gaps."""
    features = {}
    
    if not top_matches or len(top_matches) == 0:
        return None
    
    # Basic features from top match
    top1 = top_matches[0]
    features['hash_distance'] = top1['hash_distance']
    features['hist_score'] = top1['hist_score']
    features['combined_score'] = top1['combined_score']
    
    # Score gaps between consecutive matches
    if len(top_matches) >= 2:
        top2 = top_matches[1]
        features['hash_gap_1_2'] = top2['hash_distance'] - top1['hash_distance']
        features['hist_gap_1_2'] = top1['hist_score'] - top2['hist_score']
        features['combined_gap_1_2'] = top1['combined_score'] - top2['combined_score']
    else:
        features['hash_gap_1_2'] = 0
        features['hist_gap_1_2'] = 0
        features['combined_gap_1_2'] = 0
    
    if len(top_matches) >= 3:
        top3 = top_matches[2]
        features['hash_gap_2_3'] = top3['hash_distance'] - top2['hash_distance']
        features['hist_gap_2_3'] = top2['hist_score'] - top3['hist_score']
        features['combined_gap_2_3'] = top2['combined_score'] - top3['combined_score']
        
        # Total gap from 1st to 3rd
        features['hash_gap_1_3'] = top3['hash_distance'] - top1['hash_distance']
        features['hist_gap_1_3'] = top1['hist_score'] - top3['hist_score']
        features['combined_gap_1_3'] = top1['combined_score'] - top3['combined_score']
    else:
        features['hash_gap_2_3'] = 0
        features['hist_gap_2_3'] = 0
        features['combined_gap_2_3'] = 0
        features['hash_gap_1_3'] = 0
        features['hist_gap_1_3'] = 0
        features['combined_gap_1_3'] = 0
    
    # Ratio features (how much better is top1 vs top2)
    if len(top_matches) >= 2:
        features['combined_ratio_1_2'] = top1['combined_score'] / (top2['combined_score'] + 1e-10)
        features['hist_ratio_1_2'] = top1['hist_score'] / (top2['hist_score'] + 1e-10)
    else:
        features['combined_ratio_1_2'] = 1.0
        features['hist_ratio_1_2'] = 1.0
    
    # Average of top-3
    if len(top_matches) >= 3:
        features['avg_hash_top3'] = np.mean([m['hash_distance'] for m in top_matches[:3]])
        features['avg_hist_top3'] = np.mean([m['hist_score'] for m in top_matches[:3]])
        features['avg_combined_top3'] = np.mean([m['combined_score'] for m in top_matches[:3]])
    else:
        features['avg_hash_top3'] = features['hash_distance']
        features['avg_hist_top3'] = features['hist_score']
        features['avg_combined_top3'] = features['combined_score']
    
    # Label
    features['is_correct'] = int(top1['card']['card_key'] == true_key)
    
    return features


def k_fold_validation(X, y, feature_names, k=10):
    """
    Perform k-fold cross-validation on multiple models.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        k: Number of folds (default=10)
    
    Returns:
        Dictionary with cross-validation results for each model
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    models = {
        'Logistic (2 features)': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'features': ['hash_distance', 'hist_score'],
            'use_scaler': False
        },
        'Logistic (with gaps)': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'features': ['hash_distance', 'hist_score', 'hash_gap_1_2', 'hist_gap_1_2', 'combined_gap_1_2'],
            'use_scaler': False
        },
        'Logistic (all features)': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'features': feature_names,
            'use_scaler': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            'features': feature_names,
            'use_scaler': False
        }
    }
    
    results = {}
    
    for model_name, config in models.items():
        print(f"\n{'='*80}")
        print(f"K-FOLD VALIDATION: {model_name} (k={k})")
        print(f"{'='*80}")
        
        # Select features
        feature_indices = [feature_names.index(f) for f in config['features']]
        X_selected = X[:, feature_indices]
        
        fold_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': [],
            'accepted_ratio': []
        }
        
        threshold_metrics = {t: {'accuracy': [], 'precision': [], 'recall': []} 
                           for t in [0.5, 0.6, 0.7, 0.8, 0.9]}
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_selected), 1):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale if needed
            if config['use_scaler']:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
            
            # Train model
            model = deepcopy(config['model'])
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Compute metrics
            fold_metrics['accuracy'].append(accuracy_score(y_val, y_pred))
            fold_metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            fold_metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_val, y_prob)
            fold_metrics['auc'].append(auc(fpr, tpr))
            
            # Acceptance ratio (at threshold 0.5)
            fold_metrics['accepted_ratio'].append(np.mean(y_pred))
            
            # Threshold analysis
            for threshold in threshold_metrics.keys():
                y_pred_thresh = (y_prob >= threshold).astype(int)
                accepted = np.sum(y_pred_thresh)
                
                if accepted > 0:
                    tp = np.sum((y_pred_thresh == 1) & (y_val == 1))
                    fp = np.sum((y_pred_thresh == 1) & (y_val == 0))
                    fn = np.sum((y_pred_thresh == 0) & (y_val == 1))
                    
                    accuracy = tp / accepted if accepted > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    threshold_metrics[threshold]['accuracy'].append(accuracy)
                    threshold_metrics[threshold]['precision'].append(precision)
                    threshold_metrics[threshold]['recall'].append(recall)
        
        # Print fold-by-fold results
        print(f"\n{'Fold':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 66)
        for i in range(k):
            print(f"{i+1:<6} {fold_metrics['accuracy'][i]:<10.4f} "
                  f"{fold_metrics['precision'][i]:<10.4f} {fold_metrics['recall'][i]:<10.4f} "
                  f"{fold_metrics['f1'][i]:<10.4f} {fold_metrics['auc'][i]:<10.4f}")
        
        # Compute statistics
        print(f"\n{'Metric':<12} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
        print("-" * 62)
        for metric_name, values in fold_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{metric_name:<12} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
        
        # Threshold analysis summary
        print(f"\nThreshold Analysis (averaged across {k} folds):")
        print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
        print("-" * 60)
        for threshold, metrics in sorted(threshold_metrics.items()):
            if metrics['accuracy']:
                acc_mean = np.mean(metrics['accuracy'])
                prec_mean = np.mean(metrics['precision'])
                rec_mean = np.mean(metrics['recall'])
                print(f"{threshold:<12.2f} {acc_mean:<12.2%} {prec_mean:<12.2%} {rec_mean:<12.2%}")
        
        # Store results
        results[model_name] = {
            'fold_metrics': fold_metrics,
            'threshold_metrics': threshold_metrics,
            'mean_auc': np.mean(fold_metrics['auc']),
            'std_auc': np.std(fold_metrics['auc'])
        }
    
    return results


def main_assess_fp():
    # Load the data
    with open('./data/match_results_fp.json', 'r') as f:
        data = json.load(f)

    # Analyze the results
    results = {
        'true_positives': [],
        'false_positives': [],
        'false_negatives': [],
        'true_negatives': []
    }

    all_features = []  # Rich features for all top-1 matches

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
        
        # Extract rich features from top matches
        if entry['top_matches']:
            features = extract_features(entry['top_matches'], true_key)
            if features:
                all_features.append(features)

    # Print summary statistics
    print("=" * 80)
    print("MATCH RESULTS ANALYSIS")
    print("=" * 80)
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

    # Analyze feature distributions
    correct_features = [f for f in all_features if f['is_correct'] == 1]
    incorrect_features = [f for f in all_features if f['is_correct'] == 0]

    print(f"\n" + "=" * 80)
    print("FEATURE DISTRIBUTIONS")
    print("=" * 80)

    if correct_features and incorrect_features:
        feature_names = [k for k in correct_features[0].keys() if k != 'is_correct']
        
        print(f"\n{'Feature':<25} {'Correct Mean':<15} {'Incorrect Mean':<15} {'Separation':<10}")
        print("-" * 80)
        
        for feat in feature_names:
            correct_vals = [f[feat] for f in correct_features]
            incorrect_vals = [f[feat] for f in incorrect_features]
            
            correct_mean = np.mean(correct_vals)
            incorrect_mean = np.mean(incorrect_vals)
            separation = abs(correct_mean - incorrect_mean) / (np.std(correct_vals + incorrect_vals) + 1e-10)
            
            print(f"{feat:<25} {correct_mean:<15.4f} {incorrect_mean:<15.4f} {separation:<10.4f}")

    # Train multiple models
    print(f"\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    if not len(all_features):
        print('not enough data to train the model')
        return 
    
    # Prepare data
    feature_names = [k for k in all_features[0].keys() if k != 'is_correct']
    X = np.array([[f[feat] for feat in feature_names] for f in all_features])
    y = np.array([f['is_correct'] for f in all_features])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'Logistic (2 features)': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'features': ['hash_distance', 'hist_score'],
            'X': X[:, [feature_names.index('hash_distance'), feature_names.index('hist_score')]]
        },
        'Logistic (with gaps)': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'features': ['hash_distance', 'hist_score', 'hash_gap_1_2', 'hist_gap_1_2', 'combined_gap_1_2'],
            'X': X[:, [feature_names.index(f) for f in ['hash_distance', 'hist_score', 'hash_gap_1_2', 'hist_gap_1_2', 'combined_gap_1_2']]]
        },
        'Logistic (all features)': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'features': feature_names,
            'X': X_scaled
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
            'features': feature_names,
            'X': X
        }
    }
    
    best_model = None
    best_auc = 0
    best_name = None
    
    for name, config in models.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {name}")
        print(f"Features: {', '.join(config['features'][:5])}{'...' if len(config['features']) > 5 else ''}")
        print(f"{'='*80}")
        
        model = config['model']
        X_train = config['X']
        
        model.fit(X_train, y)
        y_pred = model.predict(X_train)
        y_prob = model.predict_proba(X_train)[:, 1]
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Incorrect', 'Correct']))
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = (model, config, scaler if 'scaled' in name.lower() else None)
            best_name = name
        
        # Feature importance for logistic regression
        if 'Logistic' in name:
            print("\nFeature Coefficients:")
            for feat, coef in zip(config['features'], model.coef_[0]):
                print(f"  {feat:<25} {coef:>10.4f}")
        elif 'Forest' in name:
            print("\nFeature Importances:")
            importances = sorted(zip(config['features'], model.feature_importances_), 
                                key=lambda x: x[1], reverse=True)
            for feat, imp in importances[:10]:
                print(f"  {feat:<25} {imp:>10.4f}")
        
        # Threshold analysis
        print(f"\nThreshold Analysis:")
        print(f"{'Threshold':<12} {'Accepted':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 60)
        
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            accepted = np.sum(y_pred_thresh)
            
            if accepted > 0:
                tp = np.sum((y_pred_thresh == 1) & (y == 1))
                fp = np.sum((y_pred_thresh == 1) & (y == 0))
                fn = np.sum((y_pred_thresh == 0) & (y == 1))
                
                accuracy = tp / accepted if accepted > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                print(f"{threshold:<12.2f} {accepted:<10} {accuracy:<10.2%} {precision:<10.2%} {recall:<10.2%}")
    
    print(f"\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION (k=10)")
    print("=" * 80)
    print("\nPerforming 10-fold cross-validation to assess generalization performance...")
    
    cv_results = k_fold_validation(X, y, feature_names, k=10)
    
    # Find best model based on CV results
    print(f"\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Mean AUC':<12} {'Std AUC':<12} {'Mean Acc':<12} {'Mean Prec':<12} {'Mean Recall':<12}")
    print("-" * 110)
    
    best_cv_model = None
    best_cv_auc = 0
    
    for model_name, results in cv_results.items():
        mean_auc = results['mean_auc']
        std_auc = results['std_auc']
        mean_acc = np.mean(results['fold_metrics']['accuracy'])
        mean_prec = np.mean(results['fold_metrics']['precision'])
        mean_rec = np.mean(results['fold_metrics']['recall'])
        
        print(f"{model_name:<30} {mean_auc:<12.4f} {std_auc:<12.4f} "
              f"{mean_acc:<12.4f} {mean_prec:<12.4f} {mean_rec:<12.4f}")
        
        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_cv_model = model_name
    
    print(f"\nBest model by cross-validation: {best_cv_model} (AUC: {best_cv_auc:.4f})")
    print(f"Note: Cross-validation provides a more reliable estimate of generalization performance")
    
    # Generate visualizations for best model
    print(f"\n" + "=" * 80)
    print(f"BEST MODEL: {best_name} (AUC: {best_auc:.4f})")
    print("=" * 80)
    
    model, config, scaler = best_model
    X_plot = config['X']
    y_prob = model.predict_proba(X_plot)[:, 1]
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. ROC Curve
    ax1 = plt.subplot(2, 4, 1)
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title(f'ROC Curve - {best_name}', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Probability distribution
    ax2 = plt.subplot(2, 4, 2)
    correct_probs = y_prob[y == 1]
    incorrect_probs = y_prob[y == 0]
    ax2.hist(correct_probs, bins=30, alpha=0.7, color='green', label='Correct', edgecolor='black')
    ax2.hist(incorrect_probs, bins=30, alpha=0.7, color='red', label='Incorrect', edgecolor='black')
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Probability', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Probability Distribution', fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3-8. Key feature comparisons
    plot_features = [
        ('hash_distance', 'hist_score'),
        ('hash_distance', 'combined_gap_1_2'),
        ('hist_score', 'combined_gap_1_2'),
        ('hash_gap_1_2', 'hist_gap_1_2'),
        ('combined_score', 'combined_ratio_1_2'),
        ('hash_distance', 'hist_gap_1_2')
    ]
    
    for idx, (feat1, feat2) in enumerate(plot_features, start=3):
        if feat1 not in feature_names or feat2 not in feature_names:
            continue
            
        ax = plt.subplot(2, 4, idx)
        
        f1_idx = feature_names.index(feat1)
        f2_idx = feature_names.index(feat2)
        
        correct_mask = y == 1
        incorrect_mask = y == 0
        
        ax.scatter(X[incorrect_mask, f1_idx], X[incorrect_mask, f2_idx],
                    c='red', alpha=0.6, s=50, label='Incorrect', edgecolors='black', linewidths=0.5)
        ax.scatter(X[correct_mask, f1_idx], X[correct_mask, f2_idx],
                    c='green', alpha=0.6, s=50, label='Correct', edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(feat1.replace('_', ' ').title(), fontweight='bold', fontsize=9)
        ax.set_ylabel(feat2.replace('_', ' ').title(), fontweight='bold', fontsize=9)
        ax.set_title(f'{feat1} vs {feat2}', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('match_analysis_advanced.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'match_analysis_advanced.png'")
    
    # Save best model parameters
    print(f"\n" + "=" * 80)
    print("IMPLEMENTATION CODE")
    print("=" * 80)
    
    if 'Logistic' in best_name:
        print("\nPython implementation:")
        print(f"# Model: {best_name}")
        print(f"# Features: {config['features']}")
        print(f"\ndef should_accept_match(top_matches, threshold=0.8):")
        print(f"    '''")
        print(f"    Enhanced decision boundary with score gaps.")
        print(f"    Args:")
        print(f"        top_matches: List of match dictionaries with hash_distance, hist_score, etc.")
        print(f"        threshold: Probability threshold (0.8 recommended for high precision)")
        print(f"    '''")
        print(f"    if not top_matches:")
        print(f"        return False")
        print(f"    ")
        print(f"    # Extract features")
        print(f"    features = extract_match_features(top_matches)")
        print(f"    ")
        print(f"    # Model coefficients")
        print(f"    coef = {model.coef_[0].tolist()}")
        print(f"    intercept = {model.intercept_[0]}")
        print(f"    ")
        print(f"    # Calculate probability")
        print(f"    X = [features[f] for f in {config['features']}]")
        if scaler:
            print(f"    # Standardize")
            print(f"    X_scaled = [(x - m) / s for x, m, s in zip(X, {scaler.mean_.tolist()}, {scaler.scale_.tolist()})]")
            print(f"    z = sum(c * x for c, x in zip(coef, X_scaled)) + intercept")
        else:
            print(f"    z = sum(c * x for c, x in zip(coef, X)) + intercept")
        print(f"    probability = 1 / (1 + np.exp(-z))")
        print(f"    ")
        print(f"    return probability >= threshold")
    
    plt.show()


if __name__ == "__main__":
    # main_filter_fp()
    main_assess_fp()
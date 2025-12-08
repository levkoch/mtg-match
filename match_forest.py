import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple


class MatchFilter:
    """Filters matches using trained Random Forest model to reduce false positives."""
    
    def __init__(self, model_path: str = "./data/match_filter_model.pkl"):
        """Load the trained model and parameters."""
        self.model = None
        self.feature_names = None
        self.threshold = 0.80
        
        if Path(model_path).exists():
            self.load_model(model_path)
        else:
            print(f"Warning: No trained model found at {model_path}")
            print("Run train_match_filter() first to create the model.")
    
    def load_model(self, model_path: str):
        """Load trained model from disk."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.threshold = data.get('threshold', 0.80)
        print(f"Loaded match filter model (threshold={self.threshold})")
    
    def extract_features(self, top_matches: list[dict]) -> Optional[dict]:
        """Extract rich features from top matches including score gaps."""
        if not top_matches or len(top_matches) == 0:
            return None
        
        features = {}
        
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
        
        return features
    
    def should_accept_match(self, top_matches: list[dict]) -> Tuple[bool, float]:
        """
        Determine if the top match should be accepted.
        
        Args:
            top_matches: List of top-k (3 or more) match dictionaries
            
        Returns:
            (should_accept, confidence_score)
        """
        if self.model is None:
            # No model loaded - accept all matches
            return True, 1.0
        
        features = self.extract_features(top_matches)
        if features is None:
            return False, 0.0
        
        # Prepare feature vector in correct order
        X = np.array([[features[f] for f in self.feature_names]])
        
        # Predict probability
        probability = self.model.predict_proba(X)[0, 1]
        
        # Apply threshold
        should_accept = probability >= self.threshold
        
        return should_accept, probability
    
    def set_threshold(self, threshold: float):
        """
        Set the acceptance threshold.
        
        Higher threshold = Higher precision, lower recall
        - 0.60: ~88% precision, ~90% recall (balanced)
        - 0.70: ~96% precision, ~86% recall (recommended)
        - 0.80: ~98% precision, ~82% recall (high precision)
        - 0.90: ~99% precision, ~70% recall (very conservative)
        """
        self.threshold = threshold
        print(f"Match filter threshold set to {threshold:.2f}")


def train_match_filter(results_path: str = "./data/match_results_fp.json",
                       output_path: str = "./data/match_filter_model.pkl",
                       threshold: float = 0.80,
                       n_folds: int = 10):
    """
    Train the Random Forest match filter with k-fold cross validation.
    
    Args:
        results_path: Path to match_results_fp.json
        output_path: Where to save the trained model
        threshold: Acceptance threshold (0.80 recommended for high precision)
        n_folds: Number of folds for cross-validation (default: 10)
    """
    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    
    # Load data
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract features
    filter_obj = MatchFilter.__new__(MatchFilter)  # Create without loading model
    all_features = []
    labels = []
    
    for idx, entry in data.items():
        if not entry['top_matches']:
            continue
            
        true_key = entry['true_key']
        features = filter_obj.extract_features(entry['top_matches'])
        
        if features:
            all_features.append(features)
            is_correct = entry['top_matches'][0]['card']['card_key'] == true_key
            labels.append(1 if is_correct else 0)
    
    # Prepare data
    feature_names = list(all_features[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in all_features])
    y = np.array(labels)
    
    print("=" * 80)
    print(f"K-FOLD CROSS VALIDATION (k={n_folds})")
    print("=" * 80)
    print(f"Total samples: {len(X)} ({sum(y)} correct, {len(y)-sum(y)} incorrect)")
    print(f"Threshold: {threshold}")
    print()
    
    # K-fold cross validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = {
        'precision': [],
        'recall': [],
        'roc_auc': [],
        'tp': [],
        'fp': [],
        'fn': [],
        'tn': [],
        'accepted': []
    }
    
    print(f"{'Fold':<6} {'ROC-AUC':<10} {'Precision':<12} {'Recall':<10} {'Accepted':<10} {'FP Filtered':<12}")
    print("-" * 80)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model on this fold
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        
        # Predict on test fold
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        tn = np.sum((y_pred == 0) & (y_test == 0))
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        accepted = tp + fp
        fp_filtered = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        
        # Store results
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['roc_auc'].append(roc_auc)
        fold_results['tp'].append(tp)
        fold_results['fp'].append(fp)
        fold_results['fn'].append(fn)
        fold_results['tn'].append(tn)
        fold_results['accepted'].append(accepted)
        
        print(f"{fold_idx:<6} {roc_auc:<10.4f} {precision:<12.2%} {recall:<10.2%} {accepted:<10} {fp_filtered:<12.1f}%")
    
    # Print summary statistics
    print("-" * 80)
    print(f"{'Mean':<6} {np.mean(fold_results['roc_auc']):<10.4f} {np.mean(fold_results['precision']):<12.2%} {np.mean(fold_results['recall']):<10.2%} {np.mean(fold_results['accepted']):<10.1f}")
    print(f"{'Std':<6} {np.std(fold_results['roc_auc']):<10.4f} {np.std(fold_results['precision']):<12.2%} {np.std(fold_results['recall']):<10.2%} {np.std(fold_results['accepted']):<10.1f}")
    
    print()
    print("=" * 80)
    print("CONFUSION MATRIX (Aggregated across all folds)")
    print("=" * 80)
    total_tp = sum(fold_results['tp'])
    total_fp = sum(fold_results['fp'])
    total_fn = sum(fold_results['fn'])
    total_tn = sum(fold_results['tn'])
    
    print(f"True Positives:  {total_tp:>6} (correct matches accepted)")
    print(f"False Positives: {total_fp:>6} (incorrect matches accepted)")
    print(f"False Negatives: {total_fn:>6} (correct matches rejected)")
    print(f"True Negatives:  {total_tn:>6} (incorrect matches rejected)")
    print()
    print(f"False Positive Rate: {total_fp/(total_fp+total_tn)*100:.1f}% (of incorrect matches)")
    print(f"False Negative Rate: {total_fn/(total_fn+total_tp)*100:.1f}% (of correct matches)")
    
    # Train final model on all data
    print()
    print("=" * 80)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 80)
    
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    final_model.fit(X, y)
    
    # Evaluate final model
    y_prob_final = final_model.predict_proba(X)[:, 1]
    y_pred_final = (y_prob_final >= threshold).astype(int)
    
    tp_final = np.sum((y_pred_final == 1) & (y == 1))
    fp_final = np.sum((y_pred_final == 1) & (y == 0))
    fn_final = np.sum((y_pred_final == 0) & (y == 1))
    tn_final = np.sum((y_pred_final == 0) & (y == 0))
    
    precision_final = tp_final / (tp_final + fp_final) if (tp_final + fp_final) > 0 else 0
    recall_final = tp_final / (tp_final + fn_final) if (tp_final + fn_final) > 0 else 0
    roc_auc_final = roc_auc_score(y, y_prob_final)
    
    print(f"ROC-AUC: {roc_auc_final:.4f}")
    print(f"Precision: {precision_final:.2%}")
    print(f"Recall: {recall_final:.2%}")
    print(f"Will accept: {tp_final + fp_final}/{len(y)} matches")
    print(f"False positives filtered: {tn_final}/{tn_final+fp_final} ({tn_final/(tn_final+fp_final)*100:.1f}%)")
    
    # Feature importance
    print()
    print("Top 10 Most Important Features:")
    importances = sorted(zip(feature_names, final_model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importances[:10]:
        print(f"  {feat:<25} {imp:>8.4f}")
    
    # Save model
    model_data = {
        'model': final_model,
        'feature_names': feature_names,
        'threshold': threshold,
        'cv_results': {
            'n_folds': n_folds,
            'mean_precision': float(np.mean(fold_results['precision'])),
            'mean_recall': float(np.mean(fold_results['recall'])),
            'mean_roc_auc': float(np.mean(fold_results['roc_auc'])),
            'std_precision': float(np.std(fold_results['precision'])),
            'std_recall': float(np.std(fold_results['recall'])),
            'std_roc_auc': float(np.std(fold_results['roc_auc']))
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {output_path}")
    
    return final_model, fold_results

if __name__ == "__main__":
    # we would like to train and save our model

    train_match_filter(
        results_path="./data/match_results_fp.json",
        output_path="./data/match_filter_model.pkl",
        threshold=0.80,  
        n_folds=10       
    )
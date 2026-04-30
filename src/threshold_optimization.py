"""
Threshold Optimization for Drowsiness Detection
Demonstrates how to optimize recall for safety-critical systems
"""

import numpy as np
from sklearn.metrics import (
    recall_score, precision_score, f1_score, 
    confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt


def analyze_threshold_tradeoff(y_true, y_prob, 
                                thresholds=np.arange(0.1, 1.0, 0.05)):
    """
    Analyze recall/precision tradeoff across different thresholds
    
    Args:
        y_true: Ground truth labels (0=alert, 1=drowsy)
        y_prob: Model probability predictions
        thresholds: Thresholds to test
    
    Returns:
        DataFrame with metrics for each threshold
    """
    results = []
    
    for threshold in thresholds:
        y_pred = [1 if prob >= threshold else 0 for prob in y_prob]
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results.append({
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'tp': tp,  # Caught drowsy drivers
            'fn': fn,  # Missed drowsy drivers (DANGEROUS)
            'fp': fp,  # False alarms (annoying but safe)
            'tn': tn,  # Correctly identified alert drivers
        })
    
    return results


def print_threshold_analysis(results):
    """Print detailed threshold analysis"""
    
    print("\n" + "="*100)
    print("THRESHOLD OPTIMIZATION FOR DROWSINESS DETECTION")
    print("="*100)
    
    print(f"\n{'Thresh':<8} {'Recall':<10} {'Precision':<10} {'F1':<8} " +
          f"{'Caught':<8} {'Missed':<8} {'False Alarms':<13}")
    print("-"*100)
    
    for r in results:
        print(f"{r['threshold']:<8.2f} {r['recall']:<10.2%} {r['precision']:<10.2%} " +
              f"{r['f1']:<8.4f} {r['tp']:<8} {r['fn']:<8} {r['fp']:<13}")
    
    print("\n" + "="*100)
    print("SAFETY ANALYSIS")
    print("="*100)
    
    # Find threshold for 95% recall
    for r in results:
        if r['recall'] >= 0.95:
            print(f"\n✅ To achieve 95% recall (catch 95% of drowsy drivers):")
            print(f"   Threshold: {r['threshold']:.2f}")
            print(f"   Caught drowsy drivers: {r['tp']} (95% of total)")
            print(f"   Missed drowsy drivers: {r['fn']} (5% of total) - ACCEPTABLE")
            print(f"   False alarms (alert flagged as drowsy): {r['fp']}")
            print(f"   Precision: {r['precision']:.2%} (some false alarms acceptable for safety)")
            break
    
    # Current best (highest f1)
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\n📊 Best F1-Score (balanced metric):")
    print(f"   Threshold: {best_f1['threshold']:.2f}")
    print(f"   F1: {best_f1['f1']:.4f}")
    print(f"   Recall: {best_f1['recall']:.2%}")
    print(f"   Precision: {best_f1['precision']:.2%}")
    
    print("\n" + "="*100)
    print("RECOMMENDATION FOR SAFETY-CRITICAL SYSTEM")
    print("="*100)
    print("""
Priority: RECALL (catching drowsy drivers) > PRECISION (avoiding false alarms)

Rationale:
- Missing 1 drowsy driver → Accident → Potential death ❌
- 1 false alarm → Minor inconvenience → Everyone safe ✓

Recommended Strategy:
1. Choose threshold that achieves ≥95% recall
2. Accept higher false alarm rate (lower precision)
3. Deploy with recall-optimized threshold
4. Monitor false alarm rate to refine over time

Business Impact:
- Higher recall = Fewer missed accidents (saves lives)
- Higher false alarms = Slightly reduced user experience (acceptable tradeoff)
- Overall = Safer system for everyone 🚗💤➡️🚨
    """)


def plot_threshold_metrics(results):
    """Plot recall/precision/f1 vs threshold"""
    
    thresholds = [r['threshold'] for r in results]
    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]
    f1s = [r['f1'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Recall vs Precision
    ax1.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall (Catch Drowsy)')
    ax1.plot(thresholds, precisions, 'r-', linewidth=2, label='Precision (Avoid False Alarms)')
    ax1.axhline(y=0.95, color='g', linestyle='--', label='95% Recall Target')
    ax1.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Recommended Threshold')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Recall vs Precision Trade-off', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: F1-Score
    ax2.plot(thresholds, f1s, 'g-', linewidth=2, label='F1-Score')
    max_f1_idx = f1s.index(max(f1s))
    ax2.scatter([thresholds[max_f1_idx]], [f1s[max_f1_idx]], 
               color='red', s=100, zorder=5, label='Best F1')
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Plot saved: threshold_analysis.png")
    plt.show()


def example_usage():
    """Demonstrate with example data"""
    
    # Simulate predictions on 300 test images
    # (In reality, these come from model.predict())
    np.random.seed(42)
    
    # 150 drowsy images
    drowsy_probs = np.random.beta(7, 3, 150)  # Higher predictions for drowsy
    
    # 150 alert images
    alert_probs = np.random.beta(3, 7, 150)   # Lower predictions for alert
    
    y_true = np.concatenate([np.ones(150), np.zeros(150)])  # 1=drowsy, 0=alert
    y_prob = np.concatenate([drowsy_probs, alert_probs])
    
    # Analyze thresholds
    results = analyze_threshold_tradeoff(y_true, y_prob)
    
    # Print analysis
    print_threshold_analysis(results)
    
    # Plot
    plot_threshold_metrics(results)
    
    # Show ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
    plt.ylabel('True Positive Rate (Caught Drowsy)', fontsize=12)
    plt.title('ROC Curve: Model Performance at All Thresholds', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
    print("📊 Plot saved: roc_curve.png")
    plt.show()


if __name__ == "__main__":
    example_usage()
    
    print("\n💡 TAKEAWAY:")
    print("For safety-critical drowsiness detection:")
    print("✓ Prioritize RECALL over precision")
    print("✓ Lower confidence threshold to catch more drowsy drivers")
    print("✓ Accept higher false alarm rate as acceptable tradeoff")
    print("✓ Target: ≥95% recall (catch 95% of drowsy drivers)")

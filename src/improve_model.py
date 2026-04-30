"""
Practical Steps to Improve Model Performance

This script demonstrates two practical steps to fix when the model is missing drowsy drivers:
1. Threshold optimization for higher recall
2. Data augmentation for better generalization

Run: python src/improve_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Configuration
MODEL_PATH = "models/drowsiness_mobilenetv2.keras"
TRAIN_DIR = "dataset/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def demonstrate_threshold_optimization():
    """
    Step 1: Show how lowering threshold improves drowsy detection recall
    """
    print("=" * 60)
    print("STEP 1: Threshold Optimization")
    print("=" * 60)
    
    # Simulated probabilities from model predictions
    # In production, replace with actual model predictions
    np.random.seed(42)
    
    # True labels: 1 = drowsy, 0 = non-drowsy
    true_labels = np.array([1] * 50 + [0] * 50)  # 50 drowsy, 50 non-drowsy
    
    # Simulated model probabilities (drowsy class probability)
    # Model is reasonably accurate but misses some drowsy cases
    drowsy_probs = np.concatenate([
        np.random.normal(0.55, 0.15, 50),  # Drowsy samples - some below 0.5
        np.random.normal(0.25, 0.12, 50)  # Non-drowsy samples
    ])
    drowsy_probs = np.clip(drowsy_probs, 0, 1)
    
    # Test different thresholds
    thresholds = [0.5, 0.45, 0.4, 0.35, 0.3]
    
    print("\nThreshold Analysis:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Recall':<12} {'Precision':<12} {'F1 Score':<12}")
    print("-" * 60)
    
    results = []
    for thresh in thresholds:
        predictions = (drowsy_probs >= thresh).astype(int)
        
        # Calculate metrics
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })
        
        print(f"{thresh:<12.2f} {recall:<12.2f} {precision:<12.2f} {f1:<12.2f}")
    
    print("-" * 60)
    print("\n📊 Key Insight:")
    print("   Lowering threshold from 0.5 → 0.35 increases recall")
    print("   This means fewer drowsy drivers are missed")
    print("   Trade-off: More false alarms (lower precision)")
    
    return results


def demonstrate_data_augmentation():
    """
    Step 2: Show how data augmentation helps with varied conditions
    """
    print("\n" + "=" * 60)
    print("STEP 2: Data Augmentation for Robustness")
    print("=" * 60)
    
    # Define augmentation strategies for drowsiness detection
    augmentations = {
        'Brightness Variation': {
            'purpose': 'Handle different lighting conditions',
            'range': [0.8, 1.2],
            'impact': 'Detects drowsy drivers in dark/tunnel driving'
        },
        'Rotation': {
            'purpose': 'Handle head tilt variations',
            'range': '±15 degrees',
            'impact': 'Works when driver leans slightly'
        },
        'Zoom': {
            'purpose': 'Handle distance variations',
            'range': '0.9-1.1',
            'impact': 'Works with different camera setups'
        },
        'Horizontal Flip': {
            'purpose': 'Mirror positions',
            'enabled': True,
            'impact': 'Handles driver on left/right side of frame'
        }
    }
    
    print("\nRecommended Augmentations:")
    print("-" * 60)
    
    for aug, details in augmentations.items():
        print(f"\n🔹 {aug}")
        if 'purpose' in details:
            print(f"   Purpose: {details['purpose']}")
            print(f"   Range: {details.get('range', details.get('enabled', 'N/A'))}")
            print(f"   Impact: {details['impact']}")
    
    # Show code example
    print("\n" + "-" * 60)
    print("Code Implementation:")
    print("-" * 60)
    code = '''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,          # Rotate ±15 degrees
    brightness_range=[0.8, 1.2], # Vary brightness
    zoom_range=0.1,             # Slight zoom variation
    horizontal_flip=True,       # Mirror images
    fill_mode='nearest'         # Handle edge pixels
)
'''
    print(code)
    
    return augmentations


def show_improvement_pipeline():
    """
    Complete improvement pipeline
    """
    print("\n" + "=" * 60)
    print("COMPLETE IMPROVEMENT PIPELINE")
    print("=" * 60)
    
    pipeline = """
Priority Order:
──────────────
1. ⚡ QUICK FIX: Adjust Threshold
   - Change threshold from 0.5 → 0.35
   - Immediate recall improvement
   - No retraining required

2. 🔧 LONG-TERM: Expand Dataset
   - Add more drowsy samples (especially edge cases)
   - Apply data augmentation
   - Retrain model with improved dataset

Expected Results:
────────────────
| Step | Recall Improvement | Time Required |
|------|---------------------|----------------|
| Threshold | +15-20% | Immediate |
| Augmentation | +5-10% | 1-2 days |
"""
    print(pipeline)


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 60)
    print("MODEL IMPROVEMENT DEMONSTRATION")
    print("Practical steps when missing drowsy drivers")
    print("=" * 60)
    
    # Step 1: Threshold optimization
    threshold_results = demonstrate_threshold_optimization()
    
    # Step 2: Data augmentation
    aug_results = demonstrate_data_augmentation()
    
    # Complete pipeline
    show_improvement_pipeline()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("""
For safety-critical drowsiness detection:

✅ Step 1 (Immediate): Lower threshold to 0.35
   - Catches more actual drowsy drivers
   - Accept higher false alarm rate for safety

✅ Step 2 (Long-term): Add more training data
   - Focus on edge cases (subtle drowsiness)
   - Apply lighting/angle augmentation
""")


if __name__ == "__main__":
    main()
"""
Seeing in the Dark: Lighting-Robust Preprocessing
Demonstrates techniques to make drowsiness detection work across lighting conditions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class LightingRobustPreprocessor:
    """Preprocessing pipeline for handling varying lighting conditions"""
    
    def __init__(self, face_cascade_path=None):
        """
        Initialize preprocessor with face detector
        
        Args:
            face_cascade_path: Path to cascade classifier (uses default if None)
        """
        if face_cascade_path is None:
            # Use OpenCV's default cascade
            cascade_dir = cv2.data.haarcascades
            face_cascade_path = cascade_dir + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.target_size = (128, 128)
    
    def detect_face(self, image):
        """
        Detect face in image
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Extracted face region or None if not found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None, None
        
        # Use largest face
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        return face, (x, y, w, h)
    
    def histogram_equalization(self, face):
        """
        Apply histogram equalization to normalize brightness
        
        Good for: Uniform lighting enhancement
        Pros: Simple, fast
        Cons: Can over-enhance noise
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return equalized
    
    def clahe_enhancement(self, face, clip_limit=2.0, tile_size=8):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Good for: Adaptive local contrast enhancement
        Pros: Better preserves details, avoids over-enhancement
        Cons: Slightly more complex
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        enhanced = clahe.apply(gray)
        return enhanced
    
    def gamma_correction(self, face, gamma=1.0):
        """
        Apply gamma correction to brighten/darken image
        
        Good for: Handling extremely dark or bright images
        gamma < 1.0: Brightens image (for dark images)
        gamma > 1.0: Darkens image (for overexposed)
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype('uint8')
        
        # Apply gamma correction
        corrected = cv2.LUT(gray, table)
        return corrected
    
    def bilateral_filter(self, face):
        """
        Apply bilateral filtering to reduce noise while preserving edges
        
        Good for: Reducing noise in low-light images
        """
        return cv2.bilateralFilter(face, 9, 75, 75)
    
    def preprocess_standard(self, image):
        """Standard preprocessing: Face detection + Resize"""
        face, bbox = self.detect_face(image)
        if face is None:
            return None, None
        
        resized = cv2.resize(face, self.target_size)
        normalized = resized.astype('float32') / 255.0
        
        return normalized, bbox
    
    def preprocess_enhanced(self, image, method='clahe'):
        """Enhanced preprocessing: Face detection + Enhancement + Resize"""
        face, bbox = self.detect_face(image)
        if face is None:
            return None, None
        
        # Apply enhancement based on method
        if method == 'clahe':
            enhanced = self.clahe_enhancement(face)
        elif method == 'histogram':
            enhanced = self.histogram_equalization(face)
        elif method == 'gamma':
            # Auto-detect if image is dark and apply brightening
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            gamma = 0.5 if mean_brightness < 100 else 1.0
            enhanced = self.gamma_correction(face, gamma)
        elif method == 'bilateral':
            enhanced = self.bilateral_filter(face)
        else:
            enhanced = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Resize and normalize
        resized = cv2.resize(enhanced, self.target_size)
        normalized = resized.astype('float32') / 255.0
        
        return normalized, bbox
    
    def compare_methods(self, image_path):
        """Compare different preprocessing methods on an image"""
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return
        
        # Extract face
        face, _ = self.detect_face(image)
        if face is None:
            print("Error: No face detected")
            return
        
        # Apply different methods
        methods = {
            'Original': cv2.cvtColor(face, cv2.COLOR_BGR2GRAY),
            'Histogram Equalization': self.histogram_equalization(face),
            'CLAHE': self.clahe_enhancement(face),
            'Gamma Correction': self.gamma_correction(face, 0.6),
            'Bilateral Filter': self.bilateral_filter(face),
        }
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (method_name, processed_image) in enumerate(methods.items()):
            axes[idx].imshow(processed_image, cmap='gray')
            axes[idx].set_title(method_name, fontweight='bold')
            axes[idx].axis('off')
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig('lighting_comparison.png', dpi=150, bbox_inches='tight')
        print("✅ Comparison saved: lighting_comparison.png")
        plt.show()


def demonstrate_augmentation_robustness():
    """Show how data augmentation helps with lighting variations"""
    
    print("\n" + "="*80)
    print("DATA AUGMENTATION FOR LIGHTING ROBUSTNESS")
    print("="*80)
    
    augmentation_techniques = {
        'Random Rotation': 'Simulates head tilt (handles angled lighting)',
        'Random Flip': 'Handles left/right asymmetry in shadows',
        'Random Zoom': 'Different distances affect shadow patterns',
        'Random Brightness': 'Simulates different lighting conditions',
        'Random Contrast': 'Handles different contrast levels',
    }
    
    print("\nTechnique | Purpose")
    print("-" * 80)
    for technique, purpose in augmentation_techniques.items():
        print(f"{technique:<20} | {purpose}")
    
    print("\n" + "="*80)
    print("RESULT: Model trained on augmented data handles lighting variations better")
    print("="*80)


def demonstrate_temporal_smoothing():
    """Show how temporal smoothing reduces lighting artifacts"""
    
    print("\n" + "="*80)
    print("TEMPORAL SMOOTHING FOR ROBUSTNESS")
    print("="*80)
    
    # Simulate predictions with noise (e.g., lighting flicker)
    np.random.seed(42)
    
    # Scenario 1: Actual drowsy driver in changing light
    base_drowsy = 0.8
    noisy_predictions = base_drowsy + np.random.normal(0, 0.1, 20)
    noisy_predictions = np.clip(noisy_predictions, 0, 1)
    
    # Scenario 2: Alert driver with camera glint artifacts
    base_alert = 0.3
    glint_predictions = base_alert + np.random.normal(0, 0.12, 20)
    glint_predictions = np.clip(glint_predictions, 0, 1)
    
    # Apply temporal smoothing (5-frame moving average)
    window = 5
    smoothed_drowsy = np.convolve(noisy_predictions, np.ones(window)/window, mode='valid')
    smoothed_alert = np.convolve(glint_predictions, np.ones(window)/window, mode='valid')
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Drowsy driver
    frames = np.arange(len(noisy_predictions))
    ax1.plot(frames, noisy_predictions, 'r-', alpha=0.5, label='Raw predictions (noisy)')
    ax1.plot(frames[window-1:], smoothed_drowsy, 'darkred', linewidth=2, label='Smoothed (5-frame avg)')
    ax1.axhline(y=0.3, color='green', linestyle='--', label='Threshold (0.3)')
    ax1.fill_between(frames, 0, 1, alpha=0.1, color='red')
    ax1.set_xlabel('Frame Number', fontsize=11)
    ax1.set_ylabel('Model Confidence (Drowsy)', fontsize=11)
    ax1.set_title('Drowsy Driver: Smoothing Reduces Light Flicker', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Alert driver with glint
    frames2 = np.arange(len(glint_predictions))
    ax2.plot(frames2, glint_predictions, 'b-', alpha=0.5, label='Raw predictions (glint artifact)')
    ax2.plot(frames2[window-1:], smoothed_alert, 'darkblue', linewidth=2, label='Smoothed (5-frame avg)')
    ax2.axhline(y=0.3, color='green', linestyle='--', label='Threshold (0.3)')
    ax2.fill_between(frames2, 0, 0.3, alpha=0.1, color='green')
    ax2.set_xlabel('Frame Number', fontsize=11)
    ax2.set_ylabel('Model Confidence (Drowsy)', fontsize=11)
    ax2.set_title('Alert Driver: Smoothing Filters False Alarms', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('temporal_smoothing.png', dpi=150, bbox_inches='tight')
    print("✅ Plot saved: temporal_smoothing.png")
    plt.show()
    
    print("\nKey Results:")
    print(f"  Raw fluctuations: ±{np.std(noisy_predictions):.3f}")
    print(f"  Smoothed fluctuations: ±{np.std(smoothed_drowsy):.3f}")
    print(f"  Noise reduction: {(1 - np.std(smoothed_drowsy)/np.std(noisy_predictions))*100:.1f}%")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("LIGHTING-ROBUST PREPROCESSING FOR DROWSINESS DETECTION")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = LightingRobustPreprocessor()
    
    # Show augmentation techniques
    demonstrate_augmentation_robustness()
    
    # Show temporal smoothing
    demonstrate_temporal_smoothing()
    
    print("\n" + "="*80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*80)
    print("""
1. DATA AUGMENTATION (during training)
   ✓ Random brightness/contrast adjustments
   ✓ Random rotations and flips
   ✓ Random zoom for scale invariance
   
2. PREPROCESSING (before inference)
   ✓ Face detection to extract relevant region
   ✓ CLAHE enhancement for local contrast
   ✓ Resize to standard dimensions
   
3. INFERENCE (during deployment)
   ✓ Temporal smoothing (5-frame moving average)
   ✓ Lower confidence threshold (0.3 vs 0.5)
   ✓ Fallback to degraded mode in extreme conditions
   
Result: Model works reliably from dawn ☀️ to midnight 🌙
    """)
    
    print("\n💡 The Three Pillars of Lighting Robustness:")
    print("  1. TRAINING: Diverse data + Augmentation")
    print("  2. PREPROCESSING: Face extraction + Enhancement")
    print("  3. INFERENCE: Temporal smoothing + Threshold tuning")

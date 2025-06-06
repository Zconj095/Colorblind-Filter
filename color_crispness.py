import numpy as np
import sys
from PIL import Image
try:
    import cupy as cp
    xp = cp
except Exception:
    cp = None
    xp = np

from ColorBlind_Filter_Base import apply_colorblind_filter

FILTER_TYPES = [
    'protanopia',
    'deuteranopia',
    'tritanopia',
    'monochrome',
    'no_purple',
    'neutral_difficulty',
    'warm_color_difficulty',
    'neutral_greyscale',
    'warm_greyscale',
    'complex_color_difficulty',
    'simple_color_difficulty',
    'high_contrast',
    'enhanced_contrast',
    'color_boost',
]

def calculate_crispness(image: np.ndarray, method: str = 'gradient') -> float:
    """Calculate crispness metric using different methods."""
    if image is None or image.size == 0:
        return 0.0

    arr = xp.asarray(image, dtype=xp.float32)
    
    if method == 'gradient':
        grad_x = xp.diff(arr, axis=1, prepend=arr[:, :1, :])
        grad_y = xp.diff(arr, axis=0, prepend=arr[:1, :, :])
        gradient = xp.sqrt(grad_x ** 2 + grad_y ** 2)
        score = xp.mean(gradient)
    elif method == 'laplacian':
        # Laplacian edge detection
        if len(arr.shape) == 3:
            gray = xp.mean(arr, axis=2)
        else:
            gray = arr
        # Simple convolution approximation
        diff_y = xp.abs(xp.diff(gray, 2, axis=0))
        diff_x = xp.abs(xp.diff(gray, 2, axis=1))
        # Crop to matching dimensions
        min_h = min(diff_y.shape[0], diff_x.shape[0])
        min_w = min(diff_y.shape[1], diff_x.shape[1])
        laplacian = diff_y[:min_h, :min_w] + diff_x[:min_h, :min_w]
        score = xp.mean(laplacian)
    else:
        score = xp.std(arr)  # fallback to standard deviation
    
    if cp is not None:
        score = cp.asnumpy(score)
    return float(score)

def evaluate_crispness(image: np.ndarray, intensity: float = 1.0, adaptive: bool = False) -> dict:
    """Apply each colorblind filter and compute crispness with intensity scaling."""
    results = {}
    base_crispness = calculate_crispness(image)
    
    for f_type in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, f_type)
        crispness = calculate_crispness(filtered)
        
        # Apply intensity scaling
        if intensity != 1.0:
            crispness *= intensity
        
        # Apply adaptive scaling based on original image crispness
        if adaptive and base_crispness > 0:
            adaptation_factor = 1.0 + (crispness / base_crispness - 1.0) * 0.5
            crispness *= adaptation_factor
        
        results[f_type] = crispness
    
    return results

def analyze_image_quality(image: np.ndarray) -> dict:
    """Comprehensive image quality analysis."""
    return {
        'crispness_gradient': calculate_crispness(image, 'gradient'),
        'crispness_laplacian': calculate_crispness(image, 'laplacian'),
        'contrast': float(xp.std(xp.asarray(image))),
        'brightness': float(xp.mean(xp.asarray(image))),
        'dynamic_range': float(xp.max(xp.asarray(image)) - xp.min(xp.asarray(image)))
    }

def find_best_filter(image: np.ndarray, intensity: float = 1.0, adaptive: bool = False) -> str:
    """Find the filter that maximizes crispness."""
    scores = evaluate_crispness(image, intensity, adaptive)
    return max(scores, key=scores.get)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python color_crispness.py <image_path> [intensity] [adaptive]")
        sys.exit(1)

    img_path = sys.argv[1]
    intensity = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    adaptive = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    
    img = np.array(Image.open(img_path).convert("RGB"))
    
    print("Image Quality Analysis:")
    quality = analyze_image_quality(img)
    for k, v in quality.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nFilter Crispness Scores (intensity={intensity}, adaptive={adaptive}):")
    scores = evaluate_crispness(img, intensity, adaptive)
    for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.4f}")
    
    best_filter = find_best_filter(img, intensity, adaptive)
    print(f"\nBest filter for crispness: {best_filter}")

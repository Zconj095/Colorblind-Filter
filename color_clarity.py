import argparse
import os

# Try to use CuPy for GPU acceleration if available
try:
    import cupy as xp
except Exception:
    import numpy as xp

# Try importing qiskit_aer for demonstration (not required in script logic)
try:
    from qiskit_aer import Aer
    _aer_available = True
except Exception:
    _aer_available = False

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
    'color_boost'
]

def compute_color_clarity(image_array):
    """Compute a simple clarity metric based on color variance."""
    img = xp.asarray(image_array, dtype=xp.float32)
    mean_color = xp.mean(img, axis=(0, 1), keepdims=True)
    diff = img - mean_color
    clarity = xp.sqrt(xp.mean(diff ** 2))
    return float(clarity)

def evaluate_filters(image):
    results = {}
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, ftype)
        results[ftype] = compute_color_clarity(filtered)
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate color clarity across colorblind filters")
    parser.add_argument('--image', type=str, help='Path to image file (optional)')
    args = parser.parse_args()

    # In this environment we fall back to random data if image loading is unavailable
    image = xp.random.randint(0, 256, (256, 256, 3), dtype=xp.uint8)

    if args.image and os.path.exists(args.image):
        try:
            # Minimal image loading using numpy from .npy format
            if args.image.lower().endswith('.npy'):
                image = xp.load(args.image)
            else:
                print('Only .npy images can be loaded in this environment; using random image.')
        except Exception:
            print('Failed to load image; using random image instead.')

    results = evaluate_filters(xp.asarray(image).get() if hasattr(image, 'get') else image)

    if not _aer_available:
        print('qiskit_aer not available; skipping quantum integration.')

    for ftype, score in results.items():
        print(f"{ftype:25s} clarity: {score:.2f}")

if __name__ == '__main__':
    main()

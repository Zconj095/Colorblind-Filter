import numpy as np
from imageio import imread

from ColorBlind_Filter_Base import apply_colorblind_filter

# List of filter types supported in ColorBlind_Filter_Base
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

# Conversion helpers for CCT calculation
_XYZ_MATRIX = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
])

_CHROMATICITY_WHITE = (0.3320, 0.1858)

def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _rgb_to_cct(rgb):
    linear = _srgb_to_linear(np.array(rgb, dtype=float))
    X, Y, Z = _XYZ_MATRIX.dot(linear)
    denom = X + Y + Z
    if denom == 0:
        return 0.0
    x = X / denom
    y = Y / denom
    n = (x - _CHROMATICITY_WHITE[0]) / (y - _CHROMATICITY_WHITE[1])
    cct = -449 * n**3 + 3525 * n**2 - 6823.3 * n + 5520.33
    return float(cct)

def image_cct(image: np.ndarray) -> float:
    avg_rgb = image.reshape(-1, 3).mean(axis=0)
    return _rgb_to_cct(avg_rgb)

def analyze_cold_changes(image: np.ndarray):
    base_cct = image_cct(image)
    results = {}
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, ftype)
        filtered_cct = image_cct(filtered)
        results[ftype] = filtered_cct - base_cct
    return base_cct, results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze color temperature changes across colorblind filters")
    parser.add_argument('image', help='Path to input image file')
    args = parser.parse_args()

    img = imread(args.image)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    base_cct, changes = analyze_cold_changes(img)
    print(f"Base image CCT: {base_cct:.2f} K")
    for ftype, delta in changes.items():
        status = 'cooler' if delta < 0 else 'warmer'
        print(f"{ftype:25s}: {delta:.2f} K ({status})")

if __name__ == '__main__':
    main()

from PIL import Image
import numpy as np
from ColorBlind_Filter_Base import apply_colorblind_filter

# List of filter types available in ColorBlind_Filter_Base
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
def srgb_to_linear(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_cct(r, g, b):
    """Approximate correlated color temperature from RGB values."""
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)
    X = r_lin * 0.4124 + g_lin * 0.3576 + b_lin * 0.1805
    Y = r_lin * 0.2126 + g_lin * 0.7152 + b_lin * 0.0722
    Z = r_lin * 0.0193 + g_lin * 0.1192 + b_lin * 0.9505
    if X + Y + Z == 0:
        return float('nan')
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    z = Z / (X + Y + Z)
    return 449.0 * z ** 3 + 3525.0 * z ** 2 + 6823.3 * z + 5520.33


def measure_warm_temperature(arr):
    """Measure average color temperature of warm pixels in an image array."""
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    warm_mask = (r > b) & (r >= g)
    if not np.any(warm_mask):
        return float('nan')
    warm_pixels = np.stack([r[warm_mask], g[warm_mask], b[warm_mask]], axis=-1)
    mean_rgb = warm_pixels.mean(axis=0)
    return rgb_to_cct(*mean_rgb)

def analyze_image(image_path):
    image = Image.open(image_path).convert('RGB')
    original_arr = np.array(image)
    results = {'original': measure_warm_temperature(original_arr)}
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(original_arr.copy(), ftype)
        results[ftype] = measure_warm_temperature(filtered)
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Measure warm color temperature across colorblind filters")
    parser.add_argument('image', help='Path to input image')
    args = parser.parse_args()
    results = analyze_image(args.image)
    for k, v in results.items():
        temp_str = f"{v:.2f}K" if not np.isnan(v) else 'N/A'
        print(f"{k}: {temp_str}")

if __name__ == '__main__':
    main()

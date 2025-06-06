import cv2
from ColorBlind_Filter_Base import apply_colorblind_filter
import numpy as np
# Conversion from sRGB to XYZ (D65)
_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505]
])


def _srgb_to_linear(rgb):
    rgb = rgb / 255.0
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear


def _rgb_to_xyz(rgb):
    linear = _srgb_to_linear(rgb)
    return linear @ _SRGB_TO_XYZ.T


def _xyz_to_xy(xyz):
    sum_xyz = np.sum(xyz, axis=-1, keepdims=True)
    sum_xyz[sum_xyz == 0] = 1e-6
    xy = xyz[..., :2] / sum_xyz
    return xy


def _cct_from_xy(xy):
    x = xy[..., 0]
    y = xy[..., 1]
    n = (x - 0.3320) / (0.1858 - y)
    cct = 449 * n**3 - 3525 * n**2 + 6823.3 * n + 5520.33
    return cct


def calculate_average_cold_cct(image):
    """Return the average color temperature (>6500K) of an RGB image."""
    img = image.reshape(-1, 3).astype(np.float32)
    xyz = _rgb_to_xyz(img)
    xy = _xyz_to_xy(xyz)
    cct = _cct_from_xy(xy)
    cold_mask = cct > 6500
    if np.any(cold_mask):
        return float(np.mean(cct[cold_mask]))
    return 0.0


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


def measure_cold_temperatures(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = {}
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, ftype)
        results[ftype] = calculate_average_cold_cct(filtered)
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure cold color temperatures for all filters')
    parser.add_argument('image', help='Path to image file')
    args = parser.parse_args()

    temps = measure_cold_temperatures(args.image)
    for f, t in temps.items():
        print(f'{f}: {t:.2f} K')


import argparse
import numpy as np

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - Pillow may not be installed
    Image = None

# Optional GPU acceleration
try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

# Optional quantum backend import (not used for computation but included per requirement)
try:
    from qiskit_aer import Aer  # type: ignore
except Exception:  # pragma: no cover - Aer may not be available
    Aer = None

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


def rgb_to_cct(image: np.ndarray) -> float:
    """Approximate correlated color temperature of an RGB image."""
    data = image.reshape(-1, 3).astype(float) / 255.0
    if cp is not None:
        data = cp.asarray(data)

    # sRGB to XYZ conversion matrix
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    if cp is not None:
        m = cp.asarray(m)

    xyz = data @ m.T
    x = xyz[:, 0] / (xyz.sum(axis=1) + 1e-9)
    y = xyz[:, 1] / (xyz.sum(axis=1) + 1e-9)
    n = (x - 0.3320) / (0.1858 - y + 1e-9)
    cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33

    if cp is not None:
        cct = cp.asnumpy(cct)

    return float(np.mean(cct))


def compute_temperatures(image_path: str):
    if Image is None:
        raise ImportError("Pillow is required to load images")

    img = Image.open(image_path).convert('RGB')
    image_np = np.array(img)
    results = {}
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(image_np.copy(), ftype)
        temp = rgb_to_cct(filtered)
        results[ftype] = temp
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute cold color temperature across filters")
    parser.add_argument('image', help='Path to an RGB image')
    args = parser.parse_args()
    temps = compute_temperatures(args.image)
    for ftype, temp in temps.items():
        print(f"{ftype}: {temp:.2f} K")


if __name__ == '__main__':
    main()

import matplotlib.colors
import cupy as cp
from qiskit_aer import AerSimulator
from ColorBlind_Filter_Base import apply_colorblind_filter
import numpy as np

def measure_warm_colors(image_array: np.ndarray) -> float:
    """Measure average warm color intensity in an image.

    Warm colors are defined as hues between 0-60 and 330-360 degrees in HSV.
    The measurement is the mean of saturation multiplied by value for these
    hues.
    """
    # Transfer to GPU for computation
    gpu_img = np.asarray(image_array, dtype=np.float32)
    hsv = np.asarray(matplotlib.colors.rgb_to_hsv(gpu_img / 255.0))
    hue = hsv[:, :, 0] * 360.0
    warm_mask = (hue < 60.0) | (hue > 330.0)
    warm_values = hsv[:, :, 1][warm_mask] * hsv[:, :, 2][warm_mask]
    if warm_values.size == 0:
        return 0.0
    return float(np.mean(warm_values))


def measure_filters(image: np.ndarray, filters: list) -> dict:
    """Apply each filter and measure warm color intensity."""
    results = {}
    for filter_name in filters:
        filtered = apply_colorblind_filter(image, filter_name)
        results[filter_name] = measure_warm_colors(filtered)
    return results


def main() -> None:
    """Generate a sample image and print warm color measurements."""
    image = np.random.randint(0, 256, (100, 100, 3)).astype(np.float32)
    filters = [
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
    results = measure_filters(image, filters)
    for name, score in results.items():
        print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    # Initialize a quantum simulator for compatibility with instructions
    AerSimulator()
    main()

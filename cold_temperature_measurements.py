import cv2
from ColorBlind_Filter_Base import apply_colorblind_filter
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Conversion from sRGB to XYZ (D65)
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
], dtype=np.float32)

# Temperature thresholds
COLD_TEMP_THRESHOLD = 6500
WARM_TEMP_THRESHOLD = 3500

FILTER_TYPES = [
    'protanopia', 'deuteranopia', 'tritanopia', 'monochrome',
    'no_purple', 'neutral_difficulty', 'warm_color_difficulty',
    'neutral_greyscale', 'warm_greyscale', 'complex_color_difficulty',
    'simple_color_difficulty', 'high_contrast', 'enhanced_contrast',
    'color_boost'
]


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB with improved precision."""
    rgb_norm = rgb.astype(np.float32) / 255.0
    mask = rgb_norm <= 0.04045
    return np.where(mask, rgb_norm / 12.92, np.power((rgb_norm + 0.055) / 1.055, 2.4))


def _rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB to XYZ color space."""
    linear = _srgb_to_linear(rgb)
    return linear @ _SRGB_TO_XYZ.T


def _xyz_to_xy(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to chromaticity coordinates."""
    sum_xyz = np.sum(xyz, axis=-1, keepdims=True)
    sum_xyz = np.maximum(sum_xyz, 1e-10)  # Avoid division by zero
    return xyz[..., :2] / sum_xyz


def _cct_from_xy(xy: np.ndarray) -> np.ndarray:
    """Calculate correlated color temperature from chromaticity coordinates."""
    x, y = xy[..., 0], xy[..., 1]
    
    # Handle edge cases
    denominator = 0.1858 - y
    valid_mask = np.abs(denominator) > 1e-10
    
    n = np.zeros_like(x)
    n[valid_mask] = (x[valid_mask] - 0.3320) / denominator[valid_mask]
    
    # McCamy's approximation with bounds checking
    cct = 449 * n**3 + 3525 * n**2 + 6823.3 * n + 5520.33
    
    # Clamp to reasonable temperature range
    return np.clip(cct, 1000, 25000)


def calculate_temperature_stats(image: np.ndarray, temp_threshold: float = COLD_TEMP_THRESHOLD) -> Dict[str, float]:
    """Calculate comprehensive temperature statistics for an image."""
    if image.size == 0:
        return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
    
    img_flat = image.reshape(-1, 3).astype(np.float32)
    
    # Filter out very dark pixels
    brightness = np.mean(img_flat, axis=1)
    bright_mask = brightness > 10  # Minimum brightness threshold
    
    if not np.any(bright_mask):
        return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
    
    bright_pixels = img_flat[bright_mask]
    xyz = _rgb_to_xyz(bright_pixels)
    xy = _xyz_to_xy(xyz)
    cct = _cct_from_xy(xy)
    
    # Filter by temperature threshold
    temp_mask = cct > temp_threshold
    
    if not np.any(temp_mask):
        return {'average': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0, 'count': 0}
    
    filtered_temps = cct[temp_mask]
    
    return {
        'average': float(np.mean(filtered_temps)),
        'min': float(np.min(filtered_temps)),
        'max': float(np.max(filtered_temps)),
        'std': float(np.std(filtered_temps)),
        'count': int(np.sum(temp_mask))
    }


def calculate_average_cold_cct(image: np.ndarray) -> float:
    """Return the average cold color temperature (>6500K) for backward compatibility."""
    stats = calculate_temperature_stats(image, COLD_TEMP_THRESHOLD)
    return stats['average']


def measure_cold_temperatures(image_path: str, include_stats: bool = False) -> Dict[str, float]:
    """Measure cold color temperatures across all filter types."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')
    
    logger.info(f"Processing image: {image_path}")
    
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f'Could not load image: {image_path}')
    
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = {}
    
    for ftype in FILTER_TYPES:
        try:
            filtered = apply_colorblind_filter(image, ftype)
            if include_stats:
                results[ftype] = calculate_temperature_stats(filtered)
            else:
                results[ftype] = calculate_average_cold_cct(filtered)
        except Exception as e:
            logger.warning(f"Error processing filter {ftype}: {e}")
            results[ftype] = 0.0 if not include_stats else {'average': 0.0, 'count': 0}
    
    return results


def analyze_temperature_distribution(image_path: str) -> Dict[str, Dict[str, int]]:
    """Analyze temperature distribution across ranges."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    results = {}
    temp_ranges = {
        'very_warm': (0, 3000),
        'warm': (3000, 5000),
        'neutral': (5000, 6500),
        'cool': (6500, 10000),
        'very_cool': (10000, float('inf'))
    }
    
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, ftype)
        img_flat = filtered.reshape(-1, 3).astype(np.float32)
        
        xyz = _rgb_to_xyz(img_flat)
        xy = _xyz_to_xy(xyz)
        cct = _cct_from_xy(xy)
        
        distribution = {}
        for range_name, (min_temp, max_temp) in temp_ranges.items():
            mask = (cct >= min_temp) & (cct < max_temp)
            distribution[range_name] = int(np.sum(mask))
        
        results[ftype] = distribution
    
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure cold color temperatures for all filters')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--stats', action='store_true', help='Include detailed statistics')
    parser.add_argument('--distribution', action='store_true', help='Show temperature distribution')
    args = parser.parse_args()

    try:
        if args.distribution:
            dist = analyze_temperature_distribution(args.image)
            print("Temperature Distribution:")
            for filter_type, ranges in dist.items():
                print(f"\n{filter_type}:")
                for range_name, count in ranges.items():
                    print(f"  {range_name}: {count} pixels")
        else:
            temps = measure_cold_temperatures(args.image, args.stats)
            print("Cold Temperature Analysis:")
            for filter_type, result in temps.items():
                if args.stats and isinstance(result, dict):
                    print(f"{filter_type}:")
                    print(f"  Average: {result['average']:.2f} K")
                    print(f"  Range: {result['min']:.2f} - {result['max']:.2f} K")
                    print(f"  Std Dev: {result['std']:.2f} K")
                    print(f"  Pixel Count: {result['count']}")
                else:
                    print(f"{filter_type}: {result:.2f} K")
    
    except Exception as e:
        logger.error(f"Error: {e}")

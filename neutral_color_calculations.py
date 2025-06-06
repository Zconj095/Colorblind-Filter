from ColorBlind_Filter_Base import (
    adjust_for_protanopia,
    adjust_for_deuteranopia,
    adjust_for_tritanopia,
    adjust_for_monochrome,
    adjust_for_no_purple,
    adjust_for_neutral_difficulty,
    adjust_for_warm_color_difficulty,
    adjust_for_neutral_greyscale,
    adjust_for_warm_greyscale,
    adjust_for_complex_color_difficulty,
    adjust_for_simple_color_difficulty,
)
import numpy as np
# Mapping filter names to their corresponding functions
FILTER_FUNCTIONS = {
    'protanopia': adjust_for_protanopia,
    'deuteranopia': adjust_for_deuteranopia,
    'tritanopia': adjust_for_tritanopia,
    'monochrome': adjust_for_monochrome,
    'no_purple': adjust_for_no_purple,
    'neutral_difficulty': adjust_for_neutral_difficulty,
    'warm_color_difficulty': adjust_for_warm_color_difficulty,
    'neutral_greyscale': adjust_for_neutral_greyscale,
    'warm_greyscale': adjust_for_warm_greyscale,
    'complex_color_difficulty': adjust_for_complex_color_difficulty,
    'simple_color_difficulty': adjust_for_simple_color_difficulty,
}


def compute_neutral_response(gray_value: int, filter_func) -> np.ndarray:
    """Apply a filter to a neutral image and return the mean color."""
    gray_value = np.clip(gray_value, 0, 255)
    image = np.full((10, 10, 3), gray_value, dtype=np.float32)
    filtered = filter_func(image)
    return filtered.mean(axis=(0, 1))


def calculate_neutral_colors(gray_values=None):
    """Calculate filter responses for a list of gray values."""
    if gray_values is None:
        gray_values = [0, 64, 128, 192, 255]

    results = {}
    for filter_name, func in FILTER_FUNCTIONS.items():
        values = []
        for g in gray_values:
            avg_color = compute_neutral_response(g, func)
            values.append(avg_color)
        results[filter_name] = np.stack(values)
    return results


def main():
    results = calculate_neutral_colors()
    for filter_name, colors in results.items():
        print(f"Filter: {filter_name}")
        for i, color in enumerate(colors):
            print(f"  Gray {i}: {color}")


if __name__ == "__main__":
    main()

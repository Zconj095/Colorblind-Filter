from ColorBlind_Filter_Base import apply_colorblind_filter
import numpy as np
# Define a set of simple RGB colors
SIMPLE_COLORS = {
    'red': [255, 0, 0],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0],
    'cyan': [0, 255, 255],
    'magenta': [255, 0, 255],
    'white': [255, 255, 255],
    'black': [0, 0, 0]
}

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
    'simple_color_difficulty'
]

def apply_filter_to_colors(filter_type):
    """Apply a colorblind filter to the simple color set."""
    colors = list(SIMPLE_COLORS.values())
    image = np.array(colors, dtype=np.float32).reshape(1, len(colors), 3)
    filtered = apply_colorblind_filter(image, filter_type)
    return filtered.reshape(len(colors), 3).astype(int)

def main():
    color_names = list(SIMPLE_COLORS.keys())
    for filter_type in FILTER_TYPES:
        transformed = apply_filter_to_colors(filter_type)
        print(f"\nFilter: {filter_type}")
        for name, rgb in zip(color_names, transformed):
            print(f"  {name:7s} -> {tuple(rgb)}")

if __name__ == "__main__":
    main()

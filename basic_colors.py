import cv2
import os
import numpy as np
import warnings
from typing import Dict, Callable, Tuple, List
from sklearn.exceptions import ConvergenceWarning
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
    apply_high_contrast,
    apply_enhanced_contrast,
    apply_color_boost,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*convergence.*", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Enhanced color mapping with names and hex codes
COLOR_INFO = [
    ("Red", "#FF0000", (255, 0, 0)),
    ("Green", "#00FF00", (0, 255, 0)),
    ("Blue", "#0000FF", (0, 0, 255)),
    ("Yellow", "#FFFF00", (255, 255, 0)),
    ("Magenta", "#FF00FF", (255, 0, 255)),
    ("Cyan", "#00FFFF", (0, 255, 255)),
    ("Orange", "#FF8000", (255, 128, 0)),
    ("Purple", "#8000FF", (128, 0, 255)),
    ("White", "#FFFFFF", (255, 255, 255)),
    ("Black", "#000000", (0, 0, 0)),
]

FILTERS: Dict[str, Callable] = {
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
    'high_contrast': apply_high_contrast,
    'enhanced_contrast': lambda img: apply_enhanced_contrast(img, 1.0),
    'color_boost': lambda img: apply_color_boost(img, 1.0),
}


def create_palette(block_size: int = 100, include_labels: bool = True) -> np.ndarray:
    """Create a palette with color blocks and optional labels."""
    colors = [color for _, _, color in COLOR_INFO]
    width = block_size * len(colors)
    height = block_size + (30 if include_labels else 0)
    
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill color blocks
    for i, color in enumerate(colors):
        x_start, x_end = i * block_size, (i + 1) * block_size
        palette[:block_size, x_start:x_end] = color
    
    # Add labels if requested
    if include_labels:
        for i, (name, hex_code, _) in enumerate(COLOR_INFO):
            x_pos = i * block_size + 5
            cv2.putText(palette, name[:3], (x_pos, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return palette


def create_comparison_grid(filters_subset: List[str] = None) -> np.ndarray:
    """Create a grid showing original vs filtered versions."""
    if filters_subset is None:
        filters_subset = ['protanopia', 'deuteranopia', 'tritanopia', 'monochrome']
    
    palette = create_palette(80, False)
    rows = []
    
    # Original row
    rows.append(palette)
    
    # Filter rows
    for filter_name in filters_subset:
        if filter_name in FILTERS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filtered = FILTERS[filter_name](palette.copy())
                filtered = np.clip(filtered, 0, 255).astype(np.uint8)
                rows.append(filtered)
    
    return np.vstack(rows)


def save_individual_filters(output_dir: str = "filter_examples") -> None:
    """Save individual filter examples."""
    os.makedirs(output_dir, exist_ok=True)
    palette = create_palette()
    
    # Save original
    cv2.imwrite(f"{output_dir}/original.png", palette)
    print(f"Saved: {output_dir}/original.png")
    
    # Save filtered versions
    for name, func in FILTERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filtered = func(palette.copy())
                filtered = np.clip(filtered, 0, 255).astype(np.uint8)
                cv2.imwrite(f"{output_dir}/{name}.png", filtered)
                print(f"Saved: {output_dir}/{name}.png")
        except Exception as e:
            print(f"Error processing {name}: {e}")


def generate_report() -> None:
    """Generate a text report of available filters."""
    report_path = "filter_examples/filter_report.txt"
    with open(report_path, 'w') as f:
        f.write("Color Blindness Filter Report\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total filters available: {len(FILTERS)}\n\n")
        f.write("Color palette includes:\n")
        for name, hex_code, rgb in COLOR_INFO:
            f.write(f"  {name}: {hex_code} {rgb}\n")
        f.write("\nAvailable filters:\n")
        for i, filter_name in enumerate(FILTERS.keys(), 1):
            f.write(f"  {i:2d}. {filter_name}\n")
    print(f"Report saved: {report_path}")


def main() -> None:
    """Main execution function."""
    print("Generating color blindness filter examples...")
    
    # Save individual filters
    save_individual_filters()
    
    # Create and save comparison grid
    comparison = create_comparison_grid()
    cv2.imwrite("filter_examples/comparison_grid.png", comparison)
    print("Saved: filter_examples/comparison_grid.png")
    
    # Generate report
    generate_report()
    
    print(f"\nGenerated {len(FILTERS) + 2} images and 1 report file.")


if __name__ == "__main__":
    main()

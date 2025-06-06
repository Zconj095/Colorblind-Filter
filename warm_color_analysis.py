import numpy as np
# Try to import cupy; fall back to numpy if unavailable


# Try to import qiskit for quantum weight generation
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

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
)

# Predefined warm colors (RGB)
WARM_COLORS = {
    "red": [255, 0, 0],
    "orange": [255, 165, 0],
    "yellow": [255, 255, 0],
}

# Map filter names to functions
FILTER_FUNCTIONS = {
    "protanopia": adjust_for_protanopia,
    "deuteranopia": adjust_for_deuteranopia,
    "tritanopia": adjust_for_tritanopia,
    "monochrome": adjust_for_monochrome,
    "no_purple": adjust_for_no_purple,
    "neutral_difficulty": adjust_for_neutral_difficulty,
    "warm_color_difficulty": adjust_for_warm_color_difficulty,
    "neutral_greyscale": adjust_for_neutral_greyscale,
    "warm_greyscale": adjust_for_warm_greyscale,
    "complex_color_difficulty": adjust_for_complex_color_difficulty,
    "simple_color_difficulty": adjust_for_simple_color_difficulty,
    "high_contrast": apply_high_contrast,
    "enhanced_contrast": lambda img: apply_enhanced_contrast(img, 1.0),
}


def compute_quantum_weights(num_colors: int) -> np.ndarray:
    """Generate weights using a simple quantum circuit if qiskit is available."""
    if QISKIT_AVAILABLE:
        try:
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            backend = AerSimulator()
            compiled_circuit = transpile(qc, backend)
            job = backend.run(compiled_circuit, shots=1024)
            result = job.result()
            counts = result.get_counts()
            prob_one = counts.get("1", 0) / 1024
            return np.full(num_colors, prob_one, dtype=float)
        except Exception:
            pass
    return np.ones(num_colors, dtype=float)


def apply_filter_to_color(color: np.ndarray, filter_func) -> np.ndarray:
    """Apply a colorblind filter function to a single RGB color."""
    color_img = color.reshape(1, 1, 3).astype(np.float32)
    adjusted = filter_func(color_img)
    # Replace NaN values with the original color values
    result = adjusted.reshape(3)
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        result[nan_mask] = color[nan_mask]
    return result


def compute_warm_color_adjustments() -> dict:
    """Compute warm color transformations for each filter."""
    weights = compute_quantum_weights(len(WARM_COLORS))
    results = {}
    for filter_name, filter_func in FILTER_FUNCTIONS.items():
        transformed = {}
        for (weight, (color_name, rgb)) in zip(weights, WARM_COLORS.items()):
            color = np.array(rgb, dtype=np.float32) * weight
            adjusted = apply_filter_to_color(color, filter_func)
            transformed[color_name] = adjusted
        results[filter_name] = transformed
    return results


if __name__ == "__main__":
    data = compute_warm_color_adjustments()
    for filt, colors in data.items():
        print(f"Filter: {filt}")
        for name, value in colors.items():
            print(f"  {name}: {np.round(value, 2)}")
        print()


import numpy as np
import cupy as cp
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Import filter application from the project
from ColorBlind_Filter_Base import apply_colorblind_filter

# List of filter types defined in ColorBlind_Filter_Base.py
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


def compute_cct(image: np.ndarray) -> float:
    """Approximate correlated color temperature of an image."""
    # Transfer to GPU
    arr = cp.asarray(image, dtype=cp.float32) / 255.0

    # RGB to XYZ conversion matrix
    M = cp.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=cp.float32)

    # Linearize RGB
    linear = cp.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

    # Convert to XYZ
    xyz = cp.tensordot(linear, M.T, axes=([2], [0]))
    X, Y, Z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    denom = X + Y + Z + 1e-7
    x = X / denom
    y = Y / denom

    n = (x - 0.3320) / (0.1858 - y)
    cct = -449.0 * n ** 3 + 3525.0 * n ** 2 - 6823.3 * n + 5520.33
    return float(cct.mean())


def quantum_noise_factor() -> float:
    """Sample a small quantum circuit to provide a noise factor."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    simulator = AerSimulator()
    result = simulator.run(qc).result()
    counts = result.get_counts()
    shots = sum(counts.values())
    p0 = counts.get('0', 0) / shots
    return abs(p0 - 0.5)


def hot_temperature_change(image: np.ndarray, filter_type: str) -> float:
    """Compute temperature change after applying a specific colorblind filter."""
    filtered = apply_colorblind_filter(image, filter_type)
    original_temp = compute_cct(image)
    filtered_temp = compute_cct(filtered)
    noise = quantum_noise_factor()
    return (filtered_temp - original_temp) * (1 + noise)


def main():
    image = np.random.rand(100, 100, 3).astype(np.float32) * 255.0
    results = {}
    for ft in FILTER_TYPES:
        try:
            change = hot_temperature_change(image, ft)
            results[ft] = change
        except Exception as e:
            results[ft] = f"Error: {e}"

    for ft, val in results.items():
        print(f"{ft}: {val}")


if __name__ == "__main__":
    main()

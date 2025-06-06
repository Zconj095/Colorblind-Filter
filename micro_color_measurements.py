import cupy as cp
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from ColorBlind_Filter_Base import apply_colorblind_filter
import numpy as np

def quantum_probability(value: float, shots: int = 1024) -> float:
    """Return probability of measuring |1> for a rotation encoding value."""
    qc = QuantumCircuit(1, 1)
    angle = float(np.clip(value, 0, np.pi))
    qc.ry(angle, 0)
    qc.measure(0, 0)
    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts(qc)
    return counts.get('1', 0) / shots


def micro_color_measurements(image: np.ndarray) -> dict:
    """Compute micro color metrics for all filters in ColorBlind_Filter_Base."""
    filter_types = [
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

    cp_img = cp.asarray(image, dtype=cp.float32)
    results = {}

    for ftype in filter_types:
        filtered_np = apply_colorblind_filter(image, ftype)
        cp_filtered = cp.asarray(filtered_np, dtype=cp.float32)

        diff = cp_filtered - cp_img
        diff_norm = cp.linalg.norm(diff) / diff.size
        mean_color = cp.mean(cp_filtered, axis=(0, 1))

        quantum_score = quantum_probability(float(diff_norm))

        results[ftype] = {
            'mean_color': cp.asnumpy(mean_color),
            'diff_norm': float(diff_norm),
            'quantum_score': quantum_score,
        }

    return results


if __name__ == "__main__":
    # Example with random image
    np.random.seed(0)
    dummy_image = (np.random.rand(64, 64, 3) * 255).astype(np.float32)
    metrics = micro_color_measurements(dummy_image)
    for k, v in metrics.items():
        print(k, v)

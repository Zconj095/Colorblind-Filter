import cupy as cp
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import ColorBlind_Filter_Base as cb_base


def quantum_warm_intensity():
    """Generate a warm intensity factor using a simple quantum circuit."""
    simulator = AerSimulator()
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(np.pi / 4, 0)
    qc.measure(0, 0)
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled).result()
    counts = result.get_counts()
    prob1 = counts.get('1', 0) / sum(counts.values())
    return 0.9 + 0.2 * prob1


def apply_warm_tone(image, intensity=1.0):
    """Apply a warm tone to an image using CuPy for acceleration."""
    cp_img = cp.asarray(image, dtype=cp.float32)
    warm_matrix = cp.array(
        [[1.1, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 0.9]],
        dtype=cp.float32,
    )
    warm_img = cp.tensordot(cp_img, warm_matrix, axes=([2], [1]))
    warm_img *= intensity
    warm_img = cp.clip(warm_img, 0, 255)
    return cp.asnumpy(warm_img).astype(np.uint8)


def apply_warm_filters(image):
    """Apply all colorblind filters with an additional warm tone."""
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
    results = {}
    intensity = quantum_warm_intensity()
    for f in filter_types:
        filtered = cb_base.apply_colorblind_filter(image, f)
        warmed = apply_warm_tone(filtered, intensity)
        results[f] = warmed
    return results


if __name__ == "__main__":
    test_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    outputs = apply_warm_filters(test_image)
    for name, img in outputs.items():
        print(f"Applied warm filter for {name}, result shape: {img.shape}")


import cupy as cp
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
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
    'color_boost'
]

def quantum_scale_factors():
    """Create a simple quantum circuit to derive color scaling factors."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.save_statevector()
    backend = AerSimulator(method='statevector')
    result = backend.run(transpile(qc, backend)).result()
    state = result.get_statevector()
    # Use absolute values of amplitudes to create RGB scale factors
    amplitudes = np.abs(state)
    # Take first 3 non-zero amplitudes or pad with ones if needed
    rgb_factors = amplitudes[:3] if len(amplitudes) >= 3 else np.ones(3)
    return cp.asarray(rgb_factors)


def compute_all_filters(image: np.ndarray):
    """Compute pixel colors for all filters."""
    scale = quantum_scale_factors()
    results = {}
    for f_type in FILTER_TYPES:
        filtered = cp.asarray(apply_colorblind_filter(image.copy(), f_type))
        results[f_type] = cp.clip(filtered * scale, 0, 255).astype(cp.uint8)
    return results


def example_usage():
    img = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
    results = compute_all_filters(img)
    for key, val in results.items():
        print(key, val[0, 0])


if __name__ == '__main__':
    example_usage()

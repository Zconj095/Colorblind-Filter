import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except Exception:
    QuantumCircuit = None
    AerSimulator = None

from ColorBlind_Filter_Base import apply_colorblind_filter, adjust_for_complex_color_difficulty

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
    'simple_color_difficulty',
    'high_contrast',
    'enhanced_contrast',
    'color_boost'
]

def quantum_color_matrix(theta=0.5):
    """Return a 3x3 color matrix using a simple quantum circuit."""
    if QuantumCircuit is None or AerSimulator is None:
        return np.eye(3)
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.save_statevector()
    sim = AerSimulator()
    result = sim.run(qc).result()
    vec = result.get_statevector(qc)
    amp = np.abs(vec)**2
    r = amp[0]
    g = amp[1]
    b = 1 - (r + g)
    return np.array([
        [r, 0, 0],
        [0, g, 0],
        [0, 0, b]
    ])

def apply_complex_colors(image, intensity=1.0, theta=0.5):
    """Apply complex color adjustment to all filters."""
    cp_img = cp.asarray(image) if cp is not None else image
    results = {}
    q_matrix = quantum_color_matrix(theta)
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(cp.asnumpy(cp_img) if cp is not None else cp_img, ftype)
        complex_img = adjust_for_complex_color_difficulty(filtered)
        complex_img = np.tensordot(complex_img, q_matrix, axes=([2], [1]))
        if cp is not None:
            results[ftype] = cp.asarray(complex_img)
        else:
            results[ftype] = complex_img
    return results

if __name__ == '__main__':
    img = np.random.rand(64, 64, 3) * 255
    outputs = apply_complex_colors(img)
    for k, v in outputs.items():
        print(k, v.shape)

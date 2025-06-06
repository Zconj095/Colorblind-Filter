import numpy as np

try:
    import cupy as cp
except Exception:  # fallback to numpy if cupy unavailable
    cp = np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except Exception:
    QuantumCircuit = None
    AerSimulator = None
    execute = None
import argparse
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

def quantum_scale():
    """Generate a scaling factor using a quantum circuit if available."""
    if QuantumCircuit and AerSimulator:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        backend = AerSimulator()
        result = backend.run(qc).result()
        counts = result.get_counts()
        return 1.1 if counts.get('1', 0) else 0.9
    return 1.0

def compute_sharpness(array):
    """Compute average gradient magnitude as sharpness measure."""
    g_x = cp.diff(array, axis=1, append=array[:, -1:, :])
    g_y = cp.diff(array, axis=0, append=array[-1:, :, :])
    grad_mag = cp.sqrt(g_x ** 2 + g_y ** 2)
    return float(cp.mean(grad_mag))

def analyze(image):
    """Apply each colorblind filter and compute sharpness."""
    scores = {}
    for f_type in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, f_type)
        cp_filtered = cp.asarray(filtered.astype(np.float32))
        sharpness = compute_sharpness(cp_filtered)
        scores[f_type] = sharpness * quantum_scale()
    return scores

def main():
    parser = argparse.ArgumentParser(
        description='Compute color sharpness for each colorblind filter.'
    )
    parser.add_argument('image', help='Path to input image')
    args = parser.parse_args()

    import imageio.v2 as imageio
    image = imageio.imread(args.image)
    scores = analyze(image)
    for f, s in scores.items():
        print(f"{f}: {s:.4f}")

if __name__ == '__main__':
    main()

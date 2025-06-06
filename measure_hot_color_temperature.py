import cupy as cp
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
import numpy as np
from ColorBlind_Filter_Base import apply_colorblind_filter

FILTER_TYPES = [
    'protanopia', 'deuteranopia', 'tritanopia', 'monochrome',
    'no_purple', 'neutral_difficulty', 'warm_color_difficulty',
    'neutral_greyscale', 'warm_greyscale', 'complex_color_difficulty',
    'simple_color_difficulty', 'high_contrast', 'enhanced_contrast',
    'color_boost'
]

def quantum_random_scalar():
    """Generate a small random scalar using a quantum circuit."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    backend = AerSimulator()
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    return counts.get('1', 0) / 1.0

def compute_hot_temperature(image):
    """Approximate hot color temperature for an RGB image using GPU arrays."""
    gpu_img = cp.asarray(image, dtype=cp.float32) / 255.0
    red_mean = cp.mean(gpu_img[:, :, 0])
    blue_mean = cp.mean(gpu_img[:, :, 2])
    hot_index = red_mean - blue_mean
    noise = quantum_random_scalar() * 0.01
    return float((hot_index + noise).get())

def measure_all_filters(image):
    """Measure hot color temperature across all colorblind filters."""
    results = {}
    for ftype in FILTER_TYPES:
        filtered = apply_colorblind_filter(image, ftype)
        temperature = compute_hot_temperature(filtered)
        results[ftype] = temperature
    return results

if __name__ == '__main__':
    # Example usage with a random image
    img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    temps = measure_all_filters(img)
    for f, t in temps.items():
        print(f"{f}: {t:.4f}")


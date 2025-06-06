import numpy as np
try:
    import cupy as cp
    xp = cp
except Exception:
    cp = None
    xp = np

from ColorBlind_Filter_Base import apply_colorblind_filter

# Available filter types from the base filter module
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

def compute_color_density(image):
    """Return normalized histogram of pixel intensities."""
    flat = xp.asarray(image).ravel()
    hist = xp.histogram(flat, bins=256, range=(0, 255))[0]
    density = hist / hist.sum()
    return xp.asnumpy(density)

def compute_quantum_density(values):
    """Example quantum-based mean using qiskit_aer."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        from qiskit_aer import AerSimulator
    except Exception as exc:
        raise ImportError("qiskit_aer is required for quantum density") from exc

    # Normalize values to use as amplitudes
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    n = int(np.ceil(np.log2(values.size)))
    size = 2 ** n
    amplitudes = np.zeros(size, dtype=complex)
    amplitudes[: values.size] = values / np.linalg.norm(values)

    qc = QuantumCircuit(n)
    qc.initialize(amplitudes, range(n))
    simulator = AerSimulator()
    result = simulator.run(qc).result()
    state = result.get_statevector()
    probs = np.abs(state) ** 2
    mean_idx = np.dot(probs, np.arange(size))
    return mean_idx * (values.max() / values.size)

def analyze_image(image):
    results = {}
    for filter_name in FILTER_TYPES:
        filtered = apply_colorblind_filter(image.copy(), filter_name)
        density = compute_color_density(filtered)
        results[filter_name] = density
    return results

if __name__ == "__main__":
    # Example usage with a random image
    rng = np.random.default_rng(42)
    sample_image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    densities = analyze_image(sample_image)
    for name, dens in densities.items():
        print(name, dens[:10])  # print first 10 bins of histogram

import cupy as cp
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from ColorBlind_Filter_Base import apply_colorblind_filter, analyze_color_distribution

FILTERS = [
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

def quantum_seed():
    """Generate a small quantum-based seed using qiskit_aer."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    backend = AerSimulator()
    job = backend.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    return int(list(counts.keys())[0], 2)

def micro_measurements(image):
    """Return color distribution metrics for each filter."""
    results = {}
    img_cp = cp.asarray(image)
    for flt in FILTERS:
        filtered = apply_colorblind_filter(cp.asnumpy(img_cp), flt)
        metrics = analyze_color_distribution(filtered.astype('float32'))
        results[flt] = metrics
    return results

if __name__ == "__main__":
    # Example using a random image seeded by quantum randomness
    seed = quantum_seed() or 1
    cp.random.seed(seed)
    dummy_image = (cp.random.rand(64, 64, 3) * 255).astype('uint8')
    measurements = micro_measurements(dummy_image)
    for name, metrics in measurements.items():
        print(name, metrics)

"""Utility to compute the number of unique color shades produced by each
colorblind filter defined in ``ColorBlind_Filter_Base.py``.

This script loads an image, applies each filter, and counts the
unique RGB shades in the result. It optionally uses ``cupy`` for GPU
acceleration and demonstrates a simple quantum random seed using
``qiskit_aer`` if available.
"""

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - cupy may not be available
    cp = None

try:
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit, transpile
except Exception:  # pragma: no cover - qiskit may not be available
    AerSimulator = None

from ColorBlind_Filter_Base import apply_colorblind_filter

# Filter types from ColorBlind_Filter_Base.apply_colorblind_filter
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
    'color_boost',
]


def count_unique_shades(array) -> int:
    """Return the number of unique RGB colors in the provided image array."""
    if cp is not None and isinstance(array, cp.ndarray):
        # Convert back to numpy for unique operation due to CuPy performance issues
        array = cp.asnumpy(array)
    
    flat = array.reshape(-1, array.shape[-1])
    return int(np.unique(flat, axis=0).shape[0])


def quantum_random_seed() -> Optional[int]:
    """Generate a random seed via a simple quantum circuit if qiskit is available."""
    if AerSimulator is None:
        return None

    backend = AerSimulator()
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    transpiled_qc = transpile(qc, backend)
    job = backend.run(transpiled_qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    bit = max(counts, key=counts.get)
    return int(bit, 2)


def load_image(path: Path) -> np.ndarray:
    """Load an image file as a NumPy array in RGB format."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count unique shades across colorblind filters")
    parser.add_argument('image', type=Path, help='Path to the input image file')
    args = parser.parse_args()

    img_np = load_image(args.image)

    seed = quantum_random_seed()
    if seed is not None:
        print(f"Quantum random seed: {seed}")

    for filt in FILTER_TYPES:
        filtered = apply_colorblind_filter(img_np.copy(), filt)
        filtered_gpu = cp.asarray(filtered) if cp is not None else filtered
        shade_count = count_unique_shades(filtered_gpu)
        print(f"{filt}: {shade_count} unique shades")


if __name__ == '__main__':
    main()

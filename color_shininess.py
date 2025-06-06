from ColorBlind_Filter_Base import apply_colorblind_filter
import numpy as np
try:
    import cupy as cp
except Exception:  # fallback to numpy if cupy not available
    cp = None
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
except Exception:
    AerSimulator = None
    QuantumCircuit = None

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

def quantum_shininess_factor(level: float) -> float:
    """Compute a shininess factor using a simple quantum circuit."""
    if QuantumCircuit is None or AerSimulator is None:
        return 1.0 + level * 0.1
    circ = QuantumCircuit(1)
    circ.ry(level, 0)
    circ.save_statevector()
    backend = AerSimulator(method="statevector")
    result = backend.run(circ).result()
    state = result.get_statevector(circ)
    probability_one = abs(state[1]) ** 2
    return 1.0 + probability_one * level

def apply_shininess(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply shininess effect using cupy if available."""
    if cp is None:
        img = image.astype(np.float32) / 255.0
        shiny = np.clip(img * factor, 0, 1)
        return (shiny * 255).astype(np.uint8)
    cp_img = cp.asarray(image, dtype=cp.float32) / 255.0
    shiny = cp.clip(cp_img * factor, 0, 1)
    return cp.asnumpy((shiny * 255).astype(cp.uint8))

def apply_colorblind_shiny_filter(image: np.ndarray, filter_type: str, shininess_level: float = 1.0, intensity: float = 1.0, adaptive: bool = False) -> np.ndarray:
    """Apply colorblind filter and then enhance shininess."""
    filtered = apply_colorblind_filter(image, filter_type, intensity=intensity, adaptive=adaptive)
    factor = quantum_shininess_factor(shininess_level)
    return apply_shininess(filtered, factor)

def apply_shininess_across_filters(image: np.ndarray, shininess_level: float = 1.0) -> dict:
    """Return a dictionary of all filters with shininess applied."""
    results = {}
    for ft in FILTER_TYPES:
        results[ft] = apply_colorblind_shiny_filter(image, ft, shininess_level)
    return results

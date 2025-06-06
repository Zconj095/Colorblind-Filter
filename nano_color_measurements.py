import numpy as np
try:
    import cupy as cp
except ImportError:  # fallback if cupy is not installed
    cp = np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
except ImportError:
    Aer = None
    QuantumCircuit = None
    transpile = None

from ColorBlind_Filter_Base import apply_colorblind_filter


# Visible spectrum range in nanometers
WAVELENGTHS_NM = cp.array([400, 450, 500, 550, 600, 650, 700])


def wavelength_to_rgb(wavelength):
    """Approximate conversion from wavelength to RGB."""
    w = wavelength
    if isinstance(w, cp.ndarray):
        w = w.tolist()
    R = G = B = 0.0
    if 380 <= w < 440:
        R = -(w - 440) / (440 - 380)
        B = 1.0
    elif 440 <= w < 490:
        G = (w - 440) / (490 - 440)
        B = 1.0
    elif 490 <= w < 510:
        G = 1.0
        B = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        R = (w - 510) / (580 - 510)
        G = 1.0
    elif 580 <= w < 645:
        R = 1.0
        G = -(w - 645) / (645 - 580)
    elif 645 <= w <= 780:
        R = 1.0
    factor = 0.8
    rgb = cp.array([R, G, B]) * factor * 255
    return rgb.astype(cp.uint8)


def rgb_to_wavelength(rgb):
    """Approximate conversion from RGB to wavelength using hue mapping."""
    if isinstance(rgb, cp.ndarray):
        rgb = cp.asnumpy(rgb).astype(float) / 255.0
    else:
        rgb = np.asarray(rgb).astype(float) / 255.0
    r, g, b = rgb
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx == mn:
        hue = 0
    elif mx == r:
        hue = (60 * ((g - b) / (mx - mn)) + 360) % 360
    elif mx == g:
        hue = (60 * ((b - r) / (mx - mn)) + 120)
    else:
        hue = (60 * ((r - g) / (mx - mn)) + 240)
    # Map hue [0, 360] to wavelength [700, 400]
    wavelength = 700 - (hue / 360) * 300
    return wavelength


def quantum_random_index(num):
    """Return a pseudo-random index using a small quantum circuit."""
    if AerSimulator is None:
        return 0
    n = max(1, int(cp.ceil(cp.log2(num))))
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    backend = AerSimulator()
    job = backend.run(transpile(qc, backend), shots=1)
    result = job.result()
    counts = result.get_counts()
    bitstring = list(counts.keys())[0]
    return int(bitstring, 2) % num


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


def measure_filter_wavelengths():
    palette = cp.stack([wavelength_to_rgb(w) for w in WAVELENGTHS_NM])
    palette_img = cp.reshape(palette, (len(WAVELENGTHS_NM), 1, 3))
    palette_np = cp.asnumpy(palette_img)

    measurements = {}
    for f in FILTER_TYPES:
        filtered = apply_colorblind_filter(palette_np, f)
        filtered_cp = cp.asarray(filtered)
        wavelengths = [float(rgb_to_wavelength(filtered_cp[i, 0])) for i in range(len(WAVELENGTHS_NM))]
        measurements[f] = wavelengths
    return measurements


def main():
    idx = quantum_random_index(len(FILTER_TYPES))
    measurements = measure_filter_wavelengths()
    selected_filter = FILTER_TYPES[idx]
    print(f"Random filter selected via quantum circuit: {selected_filter}\n")
    for f, wl in measurements.items():
        print(f"{f}: {wl}")


if __name__ == "__main__":
    main()

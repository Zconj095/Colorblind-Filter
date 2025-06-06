import numpy as np
# Transformation matrices defined in ColorBlind_Filter_Base.py
TRANSFORMATION_MATRICES = {
    'protanopia': np.array([
        [0.567, 0.433, 0],
        [0.558, 0.442, 0],
        [0,     0.242, 0.758]
    ]),
    'deuteranopia': np.array([
        [0.625, 0.375, 0],
        [0.70,  0.30,  0],
        [0,     0.30,  0.70]
    ]),
    'tritanopia': np.array([
        [0.95, 0.05,  0],
        [0,    0.433, 0.567],
        [0,    0.475, 0.525]
    ]),
}

# Base RGB wavelengths (nm) for red, green, and blue
BASE_WAVELENGTHS_NM = np.array([650.0, 510.0, 475.0])


def compute_wavelengths(matrix: np.ndarray) -> np.ndarray:
    """Apply the transformation matrix to the base wavelengths."""
    return matrix.dot(BASE_WAVELENGTHS_NM)


def calculate_all():
    results = {}
    for name, matrix in TRANSFORMATION_MATRICES.items():
        results[name] = compute_wavelengths(matrix)
    return results


def main():
    results = calculate_all()
    for name, w in results.items():
        print(f"{name}: R={w[0]:.2f} nm, G={w[1]:.2f} nm, B={w[2]:.2f} nm")


if __name__ == "__main__":
    main()

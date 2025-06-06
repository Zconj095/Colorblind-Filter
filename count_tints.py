from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from ColorBlind_Filter_Base import apply_colorblind_filter
import cupy as cp

def quantum_random_bytes(n):
    """Generate n random bytes using a simple quantum circuit."""
    simulator = AerSimulator()
    qc = QuantumCircuit(8, 8)
    qc.h(range(8))
    qc.measure(range(8), range(8))
    job = simulator.run(qc, shots=n)
    result = job.result()
    counts = result.get_counts()
    out = []
    for bits, count in counts.items():
        value = int(bits, 2)
        out.extend([value] * count)
        if len(out) >= n:
            break
    return out[:n]


def generate_palette(num_colors=256):
    """Generate a palette of random colors using quantum randomness."""
    values = quantum_random_bytes(num_colors * 3)
    arr = cp.array(values, dtype=cp.uint8).reshape(num_colors, 3)
    return arr


def count_tints_for_filter(palette, filter_name):
    """Apply the filter and count unique resulting tints."""
    # apply_colorblind_filter expects a numpy image
    img = palette.reshape(-1, 1, 3)
    adjusted = apply_colorblind_filter(cp.asnumpy(img), filter_name)
    adjusted_cp = cp.asarray(adjusted.reshape(-1, 3))
    unique_colors = cp.unique(adjusted_cp, axis=0)
    return int(unique_colors.shape[0])


def get_filter_names():
    """Extract filter names from ColorBlind_Filter_Base.py."""
    names = []
    with open('ColorBlind_Filter_Base.py', 'r') as f:
        in_dict = False
        for line in f:
            if 'filter_functions' in line:
                in_dict = True
                continue
            if in_dict:
                if '}' in line:
                    break
                parts = line.split(':')
                if len(parts) > 1:
                    name = parts[0].strip().strip("'")
                    if name:
                        names.append(name)
    return names


def main():
    palette = generate_palette()
    filters = get_filter_names()
    results = {}
    for f in filters:
        results[f] = count_tints_for_filter(palette, f)
    for k, v in results.items():
        print(f"{k}: {v} unique tints")


if __name__ == '__main__':
    main()

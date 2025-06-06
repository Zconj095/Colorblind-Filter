import matplotlib.colors
import logging
from typing import Dict, Tuple, Optional, Union
import time

# Use cupy if available; fallback to numpy
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False

# Try to import qiskit for demonstration purposes
try:
    from qiskit import QuantumCircuit, Aer, execute
    QISKIT_AVAILABLE = True
except Exception:
    QuantumCircuit = None
    QISKIT_AVAILABLE = False

from ColorBlind_Filter_Base import apply_colorblind_filter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hard-coded filter types extracted from ColorBlind_Filter_Base.py
FILTER_TYPES = [
    'protanopia', 'deuteranopia', 'tritanopia', 'monochrome', 'no_purple',
    'neutral_difficulty', 'warm_color_difficulty', 'neutral_greyscale',
    'warm_greyscale', 'complex_color_difficulty', 'simple_color_difficulty',
    'high_contrast', 'enhanced_contrast', 'color_boost'
]

class ColorAnalyzer:
    """Enhanced color blindness analysis with performance monitoring."""
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.qiskit_available = QISKIT_AVAILABLE
        logger.info(f"GPU acceleration: {'Available' if self.gpu_available else 'Not available'}")
        logger.info(f"Qiskit: {'Available' if self.qiskit_available else 'Not available'}")

    def generate_palette(self, num_colors: int = 360, saturation: float = 1.0, 
                        brightness: float = 1.0) -> cp.ndarray:
        """Generate an RGB palette covering the hue spectrum with configurable parameters."""
        try:
            hues = cp.linspace(0, 1, num_colors, endpoint=False)
            hsv = cp.stack([
                hues, 
                cp.full_like(hues, saturation), 
                cp.full_like(hues, brightness)
            ], axis=1)
            
            rgb = matplotlib.colors.hsv_to_rgb(cp.asnumpy(hsv))
            return (rgb * 255).astype('uint8')[cp.newaxis, :]
        except Exception as e:
            logger.error(f"Error generating palette: {e}")
            raise

    def count_unique_hues(self, image: cp.ndarray, precision: int = 360) -> int:
        """Count unique hue values in an RGB image with configurable precision."""
        try:
            hsv = matplotlib.colors.rgb_to_hsv(cp.asnumpy(image) / 255.0)
            hues = cp.asarray(hsv[..., 0])
            # Remove NaN values (can occur with grayscale pixels)
            valid_hues = hues[~cp.isnan(hues)]
            if valid_hues.size == 0:
                return 0
            unique_hues = cp.unique(cp.rint(valid_hues * precision))
            return int(unique_hues.size)
        except Exception as e:
            logger.error(f"Error counting unique hues: {e}")
            return 0

    def analyze_color_distribution(self, image: cp.ndarray) -> Dict[str, float]:
        """Analyze color distribution statistics."""
        try:
            hsv = matplotlib.colors.rgb_to_hsv(cp.asnumpy(image) / 255.0)
            hues = cp.asarray(hsv[..., 0])
            sats = cp.asarray(hsv[..., 1])
            vals = cp.asarray(hsv[..., 2])
            
            # Remove NaN values
            valid_mask = ~cp.isnan(hues)
            if cp.sum(valid_mask) == 0:
                return {'hue_range': 0, 'avg_saturation': 0, 'avg_brightness': 0}
            
            valid_hues = hues[valid_mask]
            valid_sats = sats[valid_mask]
            valid_vals = vals[valid_mask]
            
            return {
                'hue_range': float(cp.max(valid_hues) - cp.min(valid_hues)),
                'avg_saturation': float(cp.mean(valid_sats)),
                'avg_brightness': float(cp.mean(valid_vals)),
                'saturation_std': float(cp.std(valid_sats)),
                'brightness_std': float(cp.std(valid_vals))
            }
        except Exception as e:
            logger.error(f"Error analyzing color distribution: {e}")
            return {}

    def quantum_seed(self) -> Optional[int]:
        """Generate a quantum random seed if Qiskit is available."""
        if not self.qiskit_available:
            return None
        
        try:
            qc = QuantumCircuit(3, 3)  # Use 3 qubits for better randomness
            for i in range(3):
                qc.h(i)
                qc.measure(i, i)
            
            backend = Aer.get_backend('qasm_simulator')
            result = execute(qc, backend, shots=1).result()
            counts = result.get_counts()
            seed = int(list(counts.keys())[0], 2)  # Convert binary to int
            logger.info(f"Generated quantum seed: {seed}")
            return seed
        except Exception as e:
            logger.warning(f"Quantum seed generation failed: {e}")
            return None

    def benchmark_filters(self, palette: cp.ndarray, iterations: int = 3) -> Dict[str, Dict[str, Union[int, float]]]:
        """Benchmark all filters with performance metrics."""
        results = {}
        
        for filter_name in FILTER_TYPES:
            times = []
            hue_counts = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                try:
                    filtered = apply_colorblind_filter(cp.asnumpy(palette.copy()), filter_name)
                    filtered_cp = cp.asarray(filtered)
                    hue_count = self.count_unique_hues(filtered_cp)
                    end_time = time.perf_counter()
                    
                    times.append(end_time - start_time)
                    hue_counts.append(hue_count)
                except Exception as e:
                    logger.error(f"Error processing filter {filter_name}: {e}")
                    hue_counts.append(0)
                    times.append(0)
            
            # Calculate statistics
            avg_time = sum(times) / len(times) if times else 0
            avg_hues = sum(hue_counts) / len(hue_counts) if hue_counts else 0
            
            results[filter_name] = {
                'unique_hues': int(avg_hues),
                'avg_processing_time': avg_time,
                'hue_reduction_ratio': avg_hues / 360.0 if avg_hues > 0 else 0
            }
        
        return results

def main():
    """Enhanced main function with comprehensive analysis."""
    analyzer = ColorAnalyzer()
    
    # Generate quantum seed if available
    quantum_seed = analyzer.quantum_seed()
    if quantum_seed:
        cp.random.seed(quantum_seed)
    
    # Generate palette with different configurations
    configs = [
        {'num_colors': 360, 'saturation': 1.0, 'brightness': 1.0, 'name': 'Full Spectrum'},
        {'num_colors': 180, 'saturation': 0.8, 'brightness': 0.9, 'name': 'Muted Colors'},
        {'num_colors': 720, 'saturation': 1.0, 'brightness': 1.0, 'name': 'High Resolution'}
    ]
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Analysis for: {config['name']}")
        print(f"{'='*50}")
        
        palette = analyzer.generate_palette(
            num_colors=config['num_colors'],
            saturation=config['saturation'],
            brightness=config['brightness']
        )
        
        # Analyze original palette
        original_stats = analyzer.analyze_color_distribution(palette)
        print(f"Original palette stats: {original_stats}")
        
        # Benchmark all filters
        results = analyzer.benchmark_filters(palette)
        
        # Display results
        print(f"\n{'Filter':<25} {'Unique Hues':<15} {'Reduction %':<15} {'Time (ms)':<15}")
        print("-" * 70)
        
        for name, metrics in results.items():
            reduction_pct = (1 - metrics['hue_reduction_ratio']) * 100
            time_ms = metrics['avg_processing_time'] * 1000
            print(f"{name:<25} {metrics['unique_hues']:<15} {reduction_pct:<14.1f}% {time_ms:<14.2f}ms")

if __name__ == "__main__":
    main()

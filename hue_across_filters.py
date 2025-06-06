import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ModuleNotFoundError:
    cp = None
    CUPY_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ModuleNotFoundError:
    QuantumCircuit = None
    AerSimulator = None
    QISKIT_AVAILABLE = False

from ColorBlind_Filter_Base import apply_colorblind_filter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HueProcessor:
    """Enhanced hue processing with caching and batch operations."""
    
    def __init__(self, use_gpu: bool = True, cache_quantum: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.cache_quantum = cache_quantum
        self._quantum_cache = {}
        
        if self.use_gpu:
            logger.info("Using GPU acceleration with CuPy")
        else:
            logger.info("Using CPU processing with NumPy")
    
    def _get_array_backend(self):
        """Get the appropriate array backend (CuPy or NumPy)."""
        return cp if self.use_gpu else np
    
    def _quantum_hue_angle(self, seed: Optional[int] = None) -> float:
        """Generate quantum-derived hue rotation angle with caching."""
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available, using fallback angle")
            np.random.seed(seed)
            return float(np.random.uniform(0, 360))
        
        cache_key = f"quantum_{seed}" if seed is not None else "quantum_default"
        
        if self.cache_quantum and cache_key in self._quantum_cache:
            return self._quantum_cache[cache_key]
        
        try:
            qc = QuantumCircuit(2)  # Use 2 qubits for more variation
            if seed is not None:
                np.random.seed(seed)
            
            # Apply random rotations for more diverse angles
            qc.h(0)
            qc.h(1)
            qc.cx(0, 1)
            qc.ry(np.random.uniform(0, 2*np.pi), 0)
            qc.save_statevector()
            
            sim = AerSimulator()
            result = sim.run(qc, shots=1).result()
            state = result.data(0)["statevector"]
            
            # Use superposition amplitudes for angle calculation
            prob_sum = sum(abs(amplitude) ** 2 for amplitude in state)
            angle = float((prob_sum * 360.0) % 360.0)
            
            if self.cache_quantum:
                self._quantum_cache[cache_key] = angle
            
            return angle
            
        except Exception as e:
            logger.error(f"Quantum circuit failed: {e}, using fallback")
            np.random.seed(seed)
            return float(np.random.uniform(0, 360))
    
    def _rgb_to_hsv_optimized(self, img: np.ndarray) -> np.ndarray:
        """Optimized RGB to HSV conversion with error handling."""
        xp = self._get_array_backend()
        
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        img_array = xp.asarray(img, dtype=xp.float32) / 255.0
        r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
        
        maxc = xp.maximum(xp.maximum(r, g), b)
        minc = xp.minimum(xp.minimum(r, g), b)
        diff = maxc - minc
        
        # Avoid division by zero
        safe_diff = xp.where(diff == 0, 1e-10, diff)
        safe_maxc = xp.where(maxc == 0, 1e-10, maxc)
        
        # Vectorized hue calculation
        h = xp.zeros_like(maxc)
        h = xp.where((maxc == r) & (diff != 0), ((g - b) / safe_diff) % 6, h)
        h = xp.where((maxc == g) & (diff != 0), ((b - r) / safe_diff) + 2, h)
        h = xp.where((maxc == b) & (diff != 0), ((r - g) / safe_diff) + 4, h)
        h = (h / 6.0) % 1.0
        
        s = xp.where(maxc == 0, 0.0, diff / safe_maxc)
        v = maxc
        
        return xp.stack([h, s, v], axis=-1)
    
    def _hsv_to_rgb_optimized(self, hsv: np.ndarray) -> np.ndarray:
        """Optimized HSV to RGB conversion."""
        xp = self._get_array_backend()
        
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h % 1.0) * 6.0
        i = xp.floor(h).astype(xp.int32)
        f = h - i
        
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        # Vectorized RGB calculation
        rgb = xp.zeros(hsv.shape, dtype=xp.float32)
        
        conditions = [
            i % 6 == 0, i % 6 == 1, i % 6 == 2,
            i % 6 == 3, i % 6 == 4, i % 6 == 5
        ]
        
        choices_r = [v, q, p, p, t, v]
        choices_g = [t, v, v, q, p, p]
        choices_b = [p, p, t, v, v, q]
        
        rgb[..., 0] = xp.select(conditions, choices_r)
        rgb[..., 1] = xp.select(conditions, choices_g)
        rgb[..., 2] = xp.select(conditions, choices_b)
        
        result = xp.clip(rgb * 255.0, 0, 255).astype(xp.uint8)
        
        if self.use_gpu:
            return cp.asnumpy(result)
        return result
    
    def apply_hue_shift(self, image: np.ndarray, degrees: float) -> np.ndarray:
        """Apply hue shift with input validation."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if image.shape[-1] != 3:
            raise ValueError("Image must have 3 color channels (RGB)")
        
        if not 0 <= degrees <= 360:
            degrees = degrees % 360
        
        hsv = self._rgb_to_hsv_optimized(image)
        hsv[..., 0] = (hsv[..., 0] + degrees / 360.0) % 1.0
        return self._hsv_to_rgb_optimized(hsv)
    
    def apply_hue_across_filters_batch(
        self, 
        image: np.ndarray, 
        filter_types: List[str],
        custom_angles: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Enhanced batch processing with custom angles and better error handling."""
        
        if custom_angles is None:
            angle = self._quantum_hue_angle(seed)
            custom_angles = {f: angle for f in filter_types}
        
        results = {}
        failed_filters = []
        
        for filter_type in filter_types:
            try:
                # Apply color blind filter
                filtered = apply_colorblind_filter(image, filter_type)
                
                # Apply hue shift
                angle = custom_angles.get(filter_type, 0)
                shifted = self.apply_hue_shift(filtered, angle)
                
                results[filter_type] = shifted
                logger.debug(f"Successfully processed filter: {filter_type}")
                
            except Exception as e:
                logger.error(f"Failed to process filter {filter_type}: {e}")
                failed_filters.append(filter_type)
                # Store original image as fallback
                results[filter_type] = image.copy()
        
        if failed_filters:
            logger.warning(f"Failed filters: {failed_filters}")
        
        return results
    
    def save_results(self, results: Dict[str, np.ndarray], output_dir: str):
        """Save processed images to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            import cv2
            for filter_name, image in results.items():
                filename = output_path / f"{filter_name}_hue_shifted.png"
                cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved {filename}")
        except ImportError:
            logger.warning("OpenCV not available for saving images")


def create_demo_image(size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Create a more interesting demo image with gradients."""
    h, w = size
    demo = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create color gradients
    for i in range(h):
        for j in range(w):
            demo[i, j, 0] = int(255 * i / h)  # Red gradient
            demo[i, j, 1] = int(255 * j / w)  # Green gradient
            demo[i, j, 2] = int(255 * ((i + j) % (h + w)) / (h + w))  # Blue pattern
    
    return demo


if __name__ == "__main__":
    # Enhanced demo with more comprehensive testing
    processor = HueProcessor(use_gpu=True, cache_quantum=True)
    
    demo_image = create_demo_image((128, 128))
    
    filters = [
        'protanopia', 'deuteranopia', 'tritanopia',
        'monochrome', 'no_purple', 'neutral_difficulty',
        'warm_color_difficulty', 'neutral_greyscale', 'warm_greyscale'
    ]
    
    # Test with quantum-derived angles
    logger.info("Processing with quantum-derived angles...")
    shifted_images = processor.apply_hue_across_filters_batch(demo_image, filters, seed=42)
    
    # Test with custom angles
    custom_angles = {f: i * 40 for i, f in enumerate(filters)}
    logger.info("Processing with custom angles...")
    custom_shifted = processor.apply_hue_across_filters_batch(demo_image, filters, custom_angles)
    
    # Display results
    for filter_name, image in shifted_images.items():
        print(f"Filter {filter_name} -> shape {image.shape}, dtype {image.dtype}")
    
    # Optionally save results
    processor.save_results(shifted_images, "output_quantum")
    processor.save_results(custom_shifted, "output_custom")
    
    logger.info("Processing complete!")

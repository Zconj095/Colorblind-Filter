import numpy as np
import cupy as cp
import warnings
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from ColorBlind_Filter_Base import apply_colorblind_filter
from PIL import Image

# Suppress sklearn convergence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ColorShadingProcessor:
    """Enhanced color shading processor with quantum randomization."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.simulator = AerSimulator()
        
    def quantum_shading_factor(self, shots: int = 10) -> float:
        """Generate a shading factor using quantum randomization with multiple shots."""
        qc = QuantumCircuit(2, 2)  # Use 2 qubits for better randomness
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)  # Entangle qubits
        qc.measure_all()
        
        transpiled_qc = transpile(qc, self.simulator)
        result = self.simulator.run(transpiled_qc, shots=shots).result()
        counts = result.get_counts()
        
        total_shots = sum(counts.values())
        ones_count = sum(count for state, count in counts.items() if state.count('1') >= 1)
        factor = 0.7 + (ones_count / total_shots) * 0.4  # Range: 0.7 to 1.1
        return factor

    def apply_adaptive_shading(self, image: np.ndarray, factor: float, 
                               preserve_brightness: bool = True) -> np.ndarray:
        """Apply adaptive shading with optional brightness preservation."""
        if self.use_gpu:
            img_gpu = cp.asarray(image, dtype=cp.float32)
            
            if preserve_brightness:
                mean_brightness = cp.mean(img_gpu, axis=(0, 1), keepdims=True)
                shaded = img_gpu * factor
                brightness_ratio = mean_brightness / (cp.mean(shaded, axis=(0, 1), keepdims=True) + 1e-8)
                shaded *= brightness_ratio
            else:
                shaded = img_gpu * factor
                
            result = cp.clip(shaded, 0, 255)
            return cp.asnumpy(result).astype(np.uint8)
        else:
            img_float = image.astype(np.float32)
            if preserve_brightness:
                mean_brightness = np.mean(img_float, axis=(0, 1), keepdims=True)
                shaded = img_float * factor
                brightness_ratio = mean_brightness / (np.mean(shaded, axis=(0, 1), keepdims=True) + 1e-8)
                shaded *= brightness_ratio
            else:
                shaded = img_float * factor
            return np.clip(shaded, 0, 255).astype(np.uint8)

    def apply_gradient_shading(self, image: np.ndarray, 
                               start_factor: float = 0.8, end_factor: float = 1.2,
                               direction: str = 'horizontal') -> np.ndarray:
        """Apply gradient shading across the image."""
        h, w = image.shape[:2]
        
        if direction == 'horizontal':
            gradient = np.linspace(start_factor, end_factor, w)
            gradient = np.tile(gradient, (h, 1))
        elif direction == 'vertical':
            gradient = np.linspace(start_factor, end_factor, h)
            gradient = np.tile(gradient.reshape(-1, 1), (1, w))
        else:  # radial
            center_x, center_y = w // 2, h // 2
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            gradient = start_factor + (end_factor - start_factor) * (distance / max_distance)
        
        if len(image.shape) == 3:
            gradient = np.expand_dims(gradient, axis=2)
            
        return self.apply_adaptive_shading(image, gradient, preserve_brightness=False)

    def process_filters_with_variants(self, image: np.ndarray, 
                                      base_factor: float = 0.9) -> Dict[str, np.ndarray]:
        """Process all filters with multiple shading variants."""
        filter_names = [
            'protanopia', 'deuteranopia', 'tritanopia', 'monochrome',
            'no_purple', 'neutral_difficulty', 'warm_color_difficulty',
            'neutral_greyscale', 'warm_greyscale', 'complex_color_difficulty',
            'simple_color_difficulty'
        ]
        
        outputs = {}

        for name in filter_names:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Convert from NumPy to PIL because ColorBlind_Filter_Base might need PIL
                    image_uint8 = image.astype(np.uint8)
                    pil_image = Image.fromarray(image_uint8, 'RGB')
                    
                    filtered_pil = apply_colorblind_filter(pil_image, name)
                    filtered = np.array(filtered_pil, dtype=np.uint8)  # convert back to NumPy

                # Standard shading
                shaded = self.apply_adaptive_shading(filtered, base_factor)
                outputs[f'{name}_standard'] = shaded
                
                # Quantum randomized shading
                quantum_factor = self.quantum_shading_factor()
                quantum_shaded = self.apply_adaptive_shading(filtered, quantum_factor)
                outputs[f'{name}_quantum'] = quantum_shaded
                
                # Gradient shading
                gradient_shaded = self.apply_gradient_shading(filtered)
                outputs[f'{name}_gradient'] = gradient_shaded
                
            except Exception as e:
                print(f"Warning: Failed to process filter '{name}': {e}")
                continue
        
        return outputs

    def save_results(self, results: Dict[str, np.ndarray], 
                     output_dir: str = "output", prefix: str = "") -> None:
        """Save all processed images to the specified directory."""
        import cv2
        Path(output_dir).mkdir(exist_ok=True)
        
        for name, out_img in results.items():
            filename = f"{prefix}{name}.png" if prefix else f"{name}.png"
            filepath = Path(output_dir) / filename
            cv2.imwrite(str(filepath), out_img)
            print(f"Saved {filepath}")


def main():
    """Enhanced main function with argument parsing."""
    if len(sys.argv) < 2:
        print('Usage: python apply_color_shading.py <image_path> [output_dir] [--cpu]')
        print('  image_path: Path to input image')
        print('  output_dir: Output directory (default: output)')
        print('  --cpu: Force CPU processing (disable GPU)')
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "output"
    use_gpu = '--cpu' not in sys.argv
    
    # Validate input
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image not found: {image_path}')
    
    # Load image
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Unable to load image: {image_path}')
    
    print(f"Processing image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using {'GPU' if use_gpu else 'CPU'} processing")
    
    # Initialize processor
    processor = ColorShadingProcessor(use_gpu=use_gpu)
    
    # Process with enhanced features
    results = processor.process_filters_with_variants(img)
    
    # Save results
    image_name = Path(image_path).stem
    processor.save_results(results, output_dir, f"{image_name}_")
    
    print(f"\nProcessing complete! Generated {len(results)} images in '{output_dir}' directory.")


if __name__ == '__main__':
    main()

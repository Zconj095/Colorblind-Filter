import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy detected - GPU acceleration enabled")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.info("CuPy not available - using CPU computation")

try:
    from qiskit import Aer, QuantumCircuit, execute
    QUANTUM_AVAILABLE = True
    logger.info("Qiskit detected - quantum metrics enabled")
except ImportError:
    Aer = None
    QUANTUM_AVAILABLE = False
    logger.info("Qiskit not available - quantum metrics disabled")

# Enhanced transformation matrices with metadata
transformation_matrices = {
    'protanopia': {
        'matrix': np.array([
            [0.567, 0.433, 0.0],
            [0.558, 0.442, 0.0],
            [0.0,   0.242, 0.758]
        ]),
        'description': 'Red color blindness - difficulty distinguishing red/green',
        'affected_population': '1-2% of males'
    },
    'deuteranopia': {
        'matrix': np.array([
            [0.625, 0.375, 0.0],
            [0.70,  0.30,  0.0],
            [0.0,   0.30,  0.70]
        ]),
        'description': 'Green color blindness - most common form',
        'affected_population': '6% of males, 0.4% of females'
    },
    'tritanopia': {
        'matrix': np.array([
            [0.95, 0.05, 0.0],
            [0.0,  0.433, 0.567],
            [0.0,  0.475, 0.525]
        ]),
        'description': 'Blue color blindness - rare form',
        'affected_population': '0.01% of population'
    }
}

class ColorBlindnessAnalyzer:
    """Enhanced analyzer for color blindness transformations."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.lib = cp if self.use_gpu else np
        
    def compute_dimensional_colors(self, matrix: np.ndarray) -> np.ndarray:
        """Compute transformed basis colors with improved efficiency."""
        try:
            basis = self.lib.eye(3, dtype=np.float32)
            mat = self.lib.asarray(matrix, dtype=np.float32)
            transformed = self.lib.dot(basis, mat.T)
            
            if self.use_gpu:
                return cp.asnumpy(transformed)
            return transformed
        except Exception as e:
            logger.error(f"Error in dimensional color computation: {e}")
            return np.zeros((3, 3))
    
    def compute_color_confusion_matrix(self, matrix: np.ndarray) -> Dict[str, float]:
        """Analyze color confusion patterns."""
        # Standard RGB colors
        colors = {
            'red': [1, 0, 0],
            'green': [0, 1, 0],
            'blue': [0, 0, 1],
            'yellow': [1, 1, 0],
            'cyan': [0, 1, 1],
            'magenta': [1, 0, 1]
        }
        
        confusion_scores = {}
        for name, color in colors.items():
            original = np.array(color)
            transformed = np.dot(matrix, original)
            # Calculate perceptual difference
            diff = np.linalg.norm(original - transformed)
            confusion_scores[name] = float(diff)
            
        return confusion_scores
    
    def quantum_metric(self, matrix: np.ndarray) -> Optional[Dict[str, float]]:
        """Enhanced quantum-derived metrics."""
        if not QUANTUM_AVAILABLE:
            return None
            
        try:
            # Multiple quantum metrics
            trace_angle = float(np.trace(matrix)) % (2 * np.pi)
            det_angle = float(np.linalg.det(matrix)) % (2 * np.pi)
            
            metrics = {}
            
            # Trace-based quantum state
            qc1 = QuantumCircuit(1)
            qc1.ry(trace_angle, 0)
            backend = Aer.get_backend('statevector_simulator')
            result1 = execute(qc1, backend).result()
            state1 = result1.get_statevector()
            metrics['trace_amplitude'] = abs(state1[1])
            
            # Determinant-based quantum state
            qc2 = QuantumCircuit(1)
            qc2.ry(det_angle, 0)
            result2 = execute(qc2, backend).result()
            state2 = result2.get_statevector()
            metrics['det_amplitude'] = abs(state2[1])
            
            # Quantum entanglement-inspired metric
            eigenvals = np.linalg.eigvals(matrix)
            entropy = -np.sum([val * np.log(val + 1e-10) for val in eigenvals if val > 0])
            metrics['matrix_entropy'] = float(entropy)
            
            return metrics
        except Exception as e:
            logger.error(f"Error in quantum metric computation: {e}")
            return None
    
    def analyze_matrix_properties(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Comprehensive matrix analysis."""
        properties = {}
        
        # Basic properties
        properties['determinant'] = float(np.linalg.det(matrix))
        properties['trace'] = float(np.trace(matrix))
        properties['rank'] = int(np.linalg.matrix_rank(matrix))
        properties['condition_number'] = float(np.linalg.cond(matrix))
        
        # Eigenanalysis
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        properties['eigenvalues'] = eigenvals.real.tolist()
        properties['max_eigenvalue'] = float(np.max(eigenvals.real))
        properties['spectral_radius'] = float(np.max(np.abs(eigenvals)))
        
        # Matrix norms
        properties['frobenius_norm'] = float(np.linalg.norm(matrix, 'fro'))
        properties['spectral_norm'] = float(np.linalg.norm(matrix, 2))
        
        return properties
    
    def visualize_color_space_transformation(self, name: str, matrix: np.ndarray):
        """Create visualization of color space transformation."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original color space
            colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1]])
            ax1.scatter(colors[:, 0], colors[:, 1], c=colors, s=100)
            ax1.set_title('Original Color Space')
            ax1.set_xlabel('Red')
            ax1.set_ylabel('Green')
            ax1.grid(True, alpha=0.3)
            
            # Transformed color space
            transformed_colors = np.array([np.dot(matrix, color) for color in colors])
            ax2.scatter(transformed_colors[:, 0], transformed_colors[:, 1], 
                       c=colors, s=100)
            ax2.set_title(f'Transformed Color Space ({name})')
            ax2.set_xlabel('Red')
            ax2.set_ylabel('Green')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'color_transformation_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

def generate_comprehensive_report():
    """Generate a comprehensive analysis report."""
    analyzer = ColorBlindnessAnalyzer()
    
    print("=" * 60)
    print("COMPREHENSIVE COLOR BLINDNESS ANALYSIS REPORT")
    print("=" * 60)
    print(f"GPU Acceleration: {'Enabled' if analyzer.use_gpu else 'Disabled'}")
    print(f"Quantum Metrics: {'Enabled' if QUANTUM_AVAILABLE else 'Disabled'}")
    print("=" * 60)
    
    for name, data in transformation_matrices.items():
        matrix = data['matrix']
        
        print(f"\n📊 FILTER: {name.upper()}")
        print(f"Description: {data['description']}")
        print(f"Affected Population: {data['affected_population']}")
        print("-" * 40)
        
        # Dimensional analysis
        dims = analyzer.compute_dimensional_colors(matrix)
        print("Dimensional Colors (RGB basis transformation):")
        print(np.round(dims, 4))
        
        # Confusion analysis
        confusion = analyzer.compute_color_confusion_matrix(matrix)
        print("\nColor Confusion Scores:")
        for color, score in confusion.items():
            print(f"  {color:8}: {score:.4f}")
        
        # Matrix properties
        props = analyzer.analyze_matrix_properties(matrix)
        print(f"\nMatrix Properties:")
        print(f"  Determinant: {props['determinant']:.4f}")
        print(f"  Condition Number: {props['condition_number']:.4f}")
        print(f"  Spectral Radius: {props['spectral_radius']:.4f}")
        
        # Quantum metrics
        qmetrics = analyzer.quantum_metric(matrix)
        if qmetrics:
            print("\nQuantum-Derived Metrics:")
            for metric, value in qmetrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Visualization
        analyzer.visualize_color_space_transformation(name, matrix)
        print("=" * 60)

if __name__ == "__main__":
    generate_comprehensive_report()

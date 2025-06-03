from ColorBlind_Filter_Base import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
import tempfile
import os
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import RBFInterpolator
import warnings
import psutil
import platform
import time
import gc
import weakref
import ctypes
import threading
import torch

def apply_bigdata_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size):
    """
    Big data bilateral filter implementation for very large images and kernels.
    Uses distributed processing, memory mapping, and advanced caching strategies.
    
    :param image: Input image array (float32, 0-255 range) - can be very large
    :param spatial_sigma: Spatial standard deviation for Gaussian weights
    :param intensity_sigma: Intensity standard deviation for edge preservation
    :param kernel_size: Size of the filter kernel (can be very large)
    :return: High-quality bilateral filtered image optimized for big data processing
    """
    if image is None or len(image.shape) != 3:
        return image
    
    h, w, c = image.shape
    
    # Big data threshold detection
    is_big_data = (h * w > 10**7) or (kernel_size > 50)
    
    if is_big_data:
        # Use distributed processing for big data scenarios
        return apply_distributed_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size)
    
    # Standard processing for smaller images
    def apply_distributed_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size):
        """
        Distributed bilateral filter implementation for large-scale image processing.
        Uses multi-threading, block processing, and optimized memory management.
        """
        
        h, w, c = image.shape
        filtered_image = np.zeros_like(image, dtype=np.float32)
        
        # Calculate optimal block size based on available memory
        block_size = min(512, max(64, int(np.sqrt(50000000 / (kernel_size * kernel_size)))))
        
        def process_block(args):
            y_start, y_end, x_start, x_end = args
            
            # Extract block with padding
            pad_size = kernel_size // 2
            y_pad_start = max(0, y_start - pad_size)
            y_pad_end = min(h, y_end + pad_size)
            x_pad_start = max(0, x_start - pad_size)
            x_pad_end = min(w, x_end + pad_size)
            
            block = image[y_pad_start:y_pad_end, x_pad_start:x_pad_end].copy()
            
            # Apply bilateral filter to the block
            filtered_block = np.zeros_like(block)
            for ch in range(c):
                filtered_block[:, :, ch] = cv2.bilateralFilter(
                    block[:, :, ch].astype(np.uint8),
                    kernel_size,
                    intensity_sigma,
                    spatial_sigma
                ).astype(np.float32)
            
            # Extract the actual region (remove padding)
            actual_y_start = y_start - y_pad_start
            actual_y_end = actual_y_start + (y_end - y_start)
            actual_x_start = x_start - x_pad_start
            actual_x_end = actual_x_start + (x_end - x_start)
            
            return (y_start, y_end, x_start, x_end, 
                   filtered_block[actual_y_start:actual_y_end, actual_x_start:actual_x_end])
        
        # Generate block coordinates
        blocks = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                blocks.append((y, y_end, x, x_end))
        
        # Process blocks in parallel
        max_workers = min(4, len(blocks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_block, blocks))
        
        # Combine results
        for y_start, y_end, x_start, x_end, block_result in results:
            filtered_image[y_start:y_end, x_start:x_end] = block_result
        
        return filtered_image

    def create_memory_mapped_array(shape):
        """
        Create a memory-mapped array for efficient handling of large image data.
        Uses temporary files and numpy memmap for optimal memory usage.
        """
        
        # Calculate memory requirements
        dtype = np.float32
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        size_mb = size_bytes / (1024 * 1024)
        
        # Use memory mapping for arrays larger than 100MB
        if size_mb > 100:
            # Create temporary file for memory mapping
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.close()
            
            # Create memory-mapped array
            mmap_array = np.memmap(
                temp_file.name,
                dtype=dtype,
                mode='w+',
                shape=shape
            )
            
            # Initialize with zeros
            mmap_array[:] = 0
            
            # Store cleanup info as array attribute
            mmap_array._temp_file_path = temp_file.name
            mmap_array._cleanup_required = True
            
            return mmap_array
        else:
            # Use regular numpy array for smaller data
            regular_array = np.zeros(shape, dtype=dtype)
            regular_array._cleanup_required = False
            return regular_array

    def create_streaming_padded_generator(image, pad_size):
        """
        Create a memory-efficient streaming generator for padded image chunks.
        Yields image chunks with appropriate padding for bilateral filtering.
        
        :param image: Input image array
        :param pad_size: Padding size for kernel operations
        :yield: Padded image chunks for processing
        """
        h, w, c = image.shape
        
        # Calculate optimal chunk size based on memory constraints
        max_memory_mb = 500  # Maximum memory per chunk in MB
        bytes_per_pixel = 4 * c  # float32 * channels
        max_pixels = (max_memory_mb * 1024 * 1024) // bytes_per_pixel
        
        # Determine chunk height that fits memory constraints
        chunk_height = min(h, max(32, int(np.sqrt(max_pixels / w))))
        
        for chunk_start in range(0, h, chunk_height):
            chunk_end = min(chunk_start + chunk_height, h)
            
            # Calculate padded boundaries
            padded_start = max(0, chunk_start - pad_size)
            padded_end = min(h, chunk_end + pad_size)
            
            # Extract chunk with padding
            padded_chunk = image[padded_start:padded_end].copy()
            
            # Add reflection padding if needed at boundaries
            if padded_start == 0 and chunk_start > 0:
                # Top boundary - reflect padding
                top_pad = chunk_start - padded_start
                if top_pad > 0:
                    reflected_top = np.flip(padded_chunk[:min(top_pad, padded_chunk.shape[0])], axis=0)
                    padded_chunk = np.concatenate([reflected_top, padded_chunk], axis=0)
            
            if padded_end == h and chunk_end < h:
                # Bottom boundary - reflect padding
                bottom_pad = padded_end - chunk_end
                if bottom_pad > 0:
                    reflected_bottom = np.flip(padded_chunk[-min(bottom_pad, padded_chunk.shape[0]):], axis=0)
                    padded_chunk = np.concatenate([padded_chunk, reflected_bottom], axis=0)
            
            # Add metadata for chunk processing
            chunk_info = {
                'data': padded_chunk,
                'original_start': chunk_start,
                'original_end': chunk_end,
                'padded_start': padded_start,
                'padded_end': padded_end,
                'top_pad': max(0, chunk_start - padded_start),
                'bottom_pad': max(0, padded_end - chunk_end)
            }
            
            yield chunk_info
            
            # Memory cleanup
            del padded_chunk

    def create_compressed_weight_representation(kernel_size, spatial_sigma):
        """
        Create a compressed and optimized weight representation for bilateral filtering.
        Uses sparse matrices, symmetry optimization, and memory-efficient storage.
        
        :param kernel_size: Size of the filter kernel
        :param spatial_sigma: Spatial standard deviation for Gaussian weights
        :return: Compressed weight representation optimized for large-scale processing
        """
        import scipy.sparse as sp
        
        # Calculate half kernel size for symmetry optimization
        half_kernel = kernel_size // 2
        
        # Create coordinate grid for weight calculation
        y_coords, x_coords = np.meshgrid(
            np.arange(-half_kernel, half_kernel + 1),
            np.arange(-half_kernel, half_kernel + 1),
            indexing='ij'
        )
        
        # Calculate spatial distances
        spatial_distances = np.sqrt(x_coords**2 + y_coords**2)
        
        # Compute Gaussian spatial weights
        spatial_weights = np.exp(-(spatial_distances**2) / (2 * spatial_sigma**2))
        
        # Apply distance-based sparsity threshold
        distance_threshold = 3 * spatial_sigma  # 3-sigma rule
        mask = spatial_distances <= distance_threshold
        spatial_weights[~mask] = 0
        
        # Create compressed sparse representation
        sparse_weights = sp.csr_matrix(spatial_weights.flatten())
        
        # Store coordinate mappings for efficient access
        valid_coords = np.column_stack((y_coords[mask], x_coords[mask]))
        flat_indices = np.where(mask.flatten())[0]
        
        # Optimize for symmetric patterns
        center_idx = len(valid_coords) // 2
        is_symmetric = np.allclose(spatial_weights, np.flip(spatial_weights))
        
        # Create lookup tables for fast weight access
        coord_to_weight = {}
        for i, (y, x) in enumerate(valid_coords):
            coord_to_weight[(y, x)] = sparse_weights.data[i] if i < len(sparse_weights.data) else 0
        
        # Pre-compute frequently used weight combinations
        weight_cache = {}
        cache_size = min(1000, len(valid_coords))
        for i in range(cache_size):
            if i < len(valid_coords):
                y, x = valid_coords[i]
                weight_cache[f"{y}_{x}"] = coord_to_weight.get((y, x), 0)
        
        # Memory-efficient storage structure
        compressed_representation = {
            'sparse_weights': sparse_weights,
            'valid_coords': valid_coords,
            'flat_indices': flat_indices,
            'coord_to_weight': coord_to_weight,
            'weight_cache': weight_cache,
            'kernel_size': kernel_size,
            'half_kernel': half_kernel,
            'spatial_sigma': spatial_sigma,
            'is_symmetric': is_symmetric,
            'distance_threshold': distance_threshold,
            'total_valid_weights': np.sum(mask),
            'compression_ratio': (kernel_size**2) / np.sum(mask),
            'memory_footprint': sparse_weights.data.nbytes + valid_coords.nbytes
        }
        
        # Add vectorized weight lookup function
        def get_weight_vectorized(y_offsets, x_offsets):
            """Vectorized weight lookup for batch processing."""
            coords = np.column_stack((y_offsets.flatten(), x_offsets.flatten()))
            weights = np.array([coord_to_weight.get((y, x), 0) for y, x in coords])
            return weights.reshape(y_offsets.shape)
        
        compressed_representation['get_weight_vectorized'] = get_weight_vectorized
        
        # Add fast neighbor lookup for streaming processing
        def get_neighbor_weights(center_y, center_x, max_neighbors=None):
            """Get weights for neighbors around a center point."""
            if max_neighbors is None:
                max_neighbors = len(valid_coords)
            
            neighbors = []
            for i, (dy, dx) in enumerate(valid_coords[:max_neighbors]):
                neighbor_y, neighbor_x = center_y + dy, center_x + dx
                weight = coord_to_weight.get((dy, dx), 0)
                if weight > 0:
                    neighbors.append((neighbor_y, neighbor_x, weight))
            
            return neighbors
        
        compressed_representation['get_neighbor_weights'] = get_neighbor_weights
        
        return compressed_representation

    def create_ml_intensity_approximator(image, intensity_sigma):
        """
        Create a machine learning-based intensity approximator for efficient bilateral filtering.
        Uses statistical analysis, clustering, and adaptive thresholding for optimal performance.
        
        :param image: Input image array (float32, 0-255 range)
        :param intensity_sigma: Intensity standard deviation for edge preservation
        :return: ML-based intensity approximator with fast lookup capabilities
        """
        warnings.filterwarnings('ignore', category=UserWarning)
        
        h, w, c = image.shape
        
        # Sample subset of pixels for ML training (memory efficient)
        sample_rate = min(1.0, 10000 / (h * w))  # Max 10k samples
        sample_indices = np.random.choice(h * w, 
                                        size=int(h * w * sample_rate), 
                                        replace=False)
        
        # Extract pixel features for ML analysis
        image_flat = image.reshape(-1, c)
        sample_pixels = image_flat[sample_indices]
        
        # Calculate intensity features
        intensity_features = np.column_stack([
            np.mean(sample_pixels, axis=1),  # Mean intensity
            np.std(sample_pixels, axis=1),   # Intensity variation
            np.max(sample_pixels, axis=1) - np.min(sample_pixels, axis=1),  # Range
            np.median(sample_pixels, axis=1)  # Median intensity
        ])
        
        # Normalize features for ML processing
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(intensity_features)
        
        # Cluster pixels into intensity groups
        n_clusters = min(32, max(8, int(np.sqrt(len(sample_pixels)))))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_features)
        
        # Calculate optimal intensity weights for each cluster
        cluster_weights = {}
        cluster_centers = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_pixels = sample_pixels[cluster_mask]
                cluster_intensities = intensity_features[cluster_mask, 0]  # Mean intensities
                
                # Calculate adaptive intensity threshold for this cluster
                intensity_std = np.std(cluster_intensities)
                adaptive_sigma = max(intensity_sigma * 0.5, 
                                   min(intensity_sigma * 2.0, intensity_std))
                
                # Store cluster characteristics
                cluster_centers[cluster_id] = {
                    'mean_rgb': np.mean(cluster_pixels, axis=0),
                    'std_rgb': np.std(cluster_pixels, axis=0),
                    'intensity_range': [np.min(cluster_intensities), np.max(cluster_intensities)],
                    'adaptive_sigma': adaptive_sigma,
                    'pixel_count': np.sum(cluster_mask)
                }
                
                # Pre-compute weight function for this cluster
                def create_cluster_weight_func(center_rgb, adaptive_sig):
                    def weight_func(pixel_diff):
                        return np.exp(-(pixel_diff**2) / (2 * adaptive_sig**2))
                    return weight_func
                
                cluster_weights[cluster_id] = create_cluster_weight_func(
                    cluster_centers[cluster_id]['mean_rgb'],
                    adaptive_sigma
                )
        
        # Create fast lookup tables using RBF interpolation
        if len(sample_pixels) > 100:
            # Create intensity-based lookup system
            intensity_values = intensity_features[:, 0]  # Mean intensities
            weight_values = np.array([
                cluster_weights.get(label, lambda x: np.exp(-(x**2) / (2 * intensity_sigma**2)))
                for label in cluster_labels
            ])
            
            # Build RBF interpolator for smooth weight transitions
            try:
                unique_intensities = np.unique(intensity_values)
                if len(unique_intensities) > 3:
                    rbf_interpolator = RBFInterpolator(
                        intensity_values.reshape(-1, 1),
                        np.arange(len(intensity_values)),
                        kernel='gaussian',
                        smoothing=0.1
                    )
                else:
                    rbf_interpolator = None
            except:
                rbf_interpolator = None
        else:
            rbf_interpolator = None
        
        # Create edge-aware intensity analysis
        def analyze_local_edge_strength(pixel_window):
            """Analyze edge strength in local pixel neighborhood."""
            if pixel_window.size < 9:  # Minimum 3x3 window
                return 1.0
            
            # Calculate gradient magnitude
            grad_x = np.gradient(pixel_window, axis=1)
            grad_y = np.gradient(pixel_window, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Edge strength metric
            edge_strength = np.mean(gradient_magnitude) / (np.std(pixel_window) + 1e-8)
            return np.clip(edge_strength, 0.1, 5.0)
        
        # Pre-compute statistical thresholds
        intensity_percentiles = np.percentile(intensity_features[:, 0], [10, 25, 50, 75, 90])
        global_intensity_std = np.std(intensity_features[:, 0])
        
        # Create fast classification system
        def classify_pixel_type(pixel_rgb, local_variance=None):
            """Classify pixel into processing categories for optimized filtering."""
            pixel_intensity = np.mean(pixel_rgb)
            
            # Determine pixel category
            if pixel_intensity < intensity_percentiles[0]:
                return 'dark', intensity_sigma * 0.7
            elif pixel_intensity > intensity_percentiles[4]:
                return 'bright', intensity_sigma * 0.8
            elif local_variance is not None and local_variance > global_intensity_std:
                return 'high_variance', intensity_sigma * 1.2
            else:
                return 'normal', intensity_sigma
        
        # Advanced intensity approximator class
        class MLIntensityApproximator:
            def __init__(self):
                self.cluster_centers = cluster_centers
                self.cluster_weights = cluster_weights
                self.kmeans_model = kmeans
                self.scaler = scaler
                self.rbf_interpolator = rbf_interpolator
                self.intensity_percentiles = intensity_percentiles
                self.global_std = global_intensity_std
                self.base_sigma = intensity_sigma
                
                # Create lookup table for common intensity differences
                self.weight_cache = {}
                common_diffs = np.linspace(0, 255, 256)
                for diff in common_diffs:
                    self.weight_cache[int(diff)] = np.exp(-(diff**2) / (2 * intensity_sigma**2))
            
            def get_intensity_weight(self, pixel1, pixel2, local_context=None):
                """Get optimized intensity weight between two pixels."""
                # Calculate pixel difference
                pixel_diff = np.linalg.norm(pixel1 - pixel2)
                
                # Fast lookup for common differences
                diff_int = int(np.clip(pixel_diff, 0, 255))
                if diff_int in self.weight_cache:
                    base_weight = self.weight_cache[diff_int]
                else:
                    base_weight = np.exp(-(pixel_diff**2) / (2 * self.base_sigma**2))
                
                # Apply context-aware adjustments
                if local_context is not None:
                    edge_factor = analyze_local_edge_strength(local_context)
                    adaptive_sigma = self.base_sigma * (1.0 + 0.3 * (edge_factor - 1.0))
                    base_weight = np.exp(-(pixel_diff**2) / (2 * adaptive_sigma**2))
                
                return np.clip(base_weight, 0.0, 1.0)
            
            def get_adaptive_sigma(self, pixel_rgb, neighborhood=None):
                """Get adaptive sigma value based on pixel characteristics."""
                pixel_intensity = np.mean(pixel_rgb)
                
                # Classify pixel and get adaptive sigma
                pixel_type, adaptive_sigma = classify_pixel_type(
                    pixel_rgb, 
                    np.var(neighborhood) if neighborhood is not None else None
                )
                
                return adaptive_sigma
            
            def batch_compute_weights(self, center_pixel, neighbor_pixels):
                """Efficiently compute weights for multiple neighbor pixels."""
                center_intensity = np.mean(center_pixel)
                neighbor_intensities = np.mean(neighbor_pixels, axis=1)
                
                # Vectorized difference calculation
                intensity_diffs = np.abs(neighbor_intensities - center_intensity)
                
                # Use cluster-based adaptive processing
                try:
                    # Predict cluster for center pixel
                    center_features = np.array([[
                        center_intensity,
                        np.std(center_pixel),
                        np.max(center_pixel) - np.min(center_pixel),
                        np.median(center_pixel)
                    ]])
                    center_features_norm = self.scaler.transform(center_features)
                    cluster_id = self.kmeans_model.predict(center_features_norm)[0]
                    
                    # Use cluster-specific sigma
                    if cluster_id in self.cluster_centers:
                        adaptive_sigma = self.cluster_centers[cluster_id]['adaptive_sigma']
                    else:
                        adaptive_sigma = self.base_sigma
                        
                except:
                    adaptive_sigma = self.base_sigma
                
                # Vectorized weight computation
                weights = np.exp(-(intensity_diffs**2) / (2 * adaptive_sigma**2))
                
                return np.clip(weights, 0.0, 1.0)
            
            def get_memory_footprint(self):
                """Get memory usage statistics."""
                total_size = 0
                total_size += len(self.weight_cache) * 16  # Cache entries
                total_size += len(self.cluster_centers) * 200  # Cluster data
                total_size += self.intensity_percentiles.nbytes
                return total_size
        
        # Create and return the approximator instance
        approximator = MLIntensityApproximator()
        
        # Add performance monitoring
        approximator.stats = {
            'n_clusters': n_clusters,
            'sample_rate': sample_rate,
            'cache_size': len(approximator.weight_cache),
            'memory_footprint_mb': approximator.get_memory_footprint() / (1024 * 1024),
            'has_rbf': rbf_interpolator is not None
        }
        
        return approximator

    def calculate_optimal_chunk_size(h, w, kernel_size):
        """
        Calculate optimal chunk size for bilateral filtering based on image dimensions,
        kernel size, available memory, and hardware capabilities.
        
        :param h: Image height
        :param w: Image width
        :param kernel_size: Filter kernel size
        :return: Optimal chunk size for processing
        """
        
        # Get system memory information
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        total_memory_gb = memory_info.total / (1024**3)
        
        # Get CPU information for parallel processing optimization
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        logical_cpu_count = psutil.cpu_count(logical=True)
        
        # Base memory requirements per pixel (float32 * channels + overhead)
        bytes_per_pixel = 4 * 3  # RGB float32
        kernel_memory_factor = kernel_size * kernel_size * 0.1  # Kernel processing overhead
        
        # Calculate memory constraints
        max_memory_per_chunk_gb = min(
            available_memory_gb * 0.3,  # Use max 30% of available memory
            2.0  # Cap at 2GB per chunk for stability
        )
        max_pixels_per_chunk = int((max_memory_per_chunk_gb * 1024**3) / 
                                  (bytes_per_pixel * (1 + kernel_memory_factor)))
        
        # Calculate processing efficiency factors
        def calculate_cache_efficiency(chunk_size):
            """Estimate L3 cache efficiency based on chunk size."""
            # Typical L3 cache sizes: 8-32MB
            l3_cache_mb = 16  # Conservative estimate
            chunk_memory_mb = (chunk_size * w * bytes_per_pixel) / (1024**2)
            
            if chunk_memory_mb <= l3_cache_mb * 0.5:
                return 1.0  # Excellent cache efficiency
            elif chunk_memory_mb <= l3_cache_mb:
                return 0.8  # Good cache efficiency
            elif chunk_memory_mb <= l3_cache_mb * 2:
                return 0.6  # Moderate cache efficiency
            else:
                return 0.3  # Poor cache efficiency
        
        def calculate_kernel_efficiency(chunk_size, kernel_size):
            """Calculate efficiency based on kernel overlap."""
            overlap_factor = (kernel_size - 1) / chunk_size
            return max(0.5, 1.0 - overlap_factor * 0.5)
        
        def calculate_parallelization_efficiency(chunk_size):
            """Estimate parallel processing efficiency."""
            total_chunks = max(1, h // chunk_size)
            
            if total_chunks < cpu_count:
                return 0.7  # Under-utilization
            elif total_chunks <= cpu_count * 2:
                return 1.0  # Optimal utilization
            elif total_chunks <= cpu_count * 4:
                return 0.9  # Good utilization with some overhead
            else:
                return 0.8  # High overhead from too many chunks
        
        # Adaptive chunk size calculation based on image characteristics
        def calculate_adaptive_chunk_size():
            """Calculate chunk size adapted to image and system characteristics."""
            
            # Base chunk size from memory constraints
            base_chunk_size = min(h, int(np.sqrt(max_pixels_per_chunk / w)))
            
            # Adjust for kernel size - ensure chunks are significantly larger than kernel
            min_chunk_for_kernel = max(kernel_size * 4, 64)
            kernel_adjusted_size = max(base_chunk_size, min_chunk_for_kernel)
            
            # Test different chunk sizes for optimal efficiency
            candidate_sizes = []
            
            # Generate candidate chunk sizes
            for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                candidate_size = int(kernel_adjusted_size * multiplier)
                if candidate_size <= h and candidate_size >= min_chunk_for_kernel:
                    candidate_sizes.append(candidate_size)
            
            # Add power-of-2 sizes for memory alignment
            power_of_2_sizes = [64, 128, 256, 512, 1024]
            for size in power_of_2_sizes:
                if min_chunk_for_kernel <= size <= h:
                    candidate_sizes.append(size)
            
            # Remove duplicates and sort
            candidate_sizes = sorted(list(set(candidate_sizes)))
            
            # Evaluate each candidate
            best_score = 0
            best_chunk_size = candidate_sizes[0] if candidate_sizes else min_chunk_for_kernel
            
            for chunk_size in candidate_sizes:
                # Calculate efficiency scores
                cache_eff = calculate_cache_efficiency(chunk_size)
                kernel_eff = calculate_kernel_efficiency(chunk_size, kernel_size)
                parallel_eff = calculate_parallelization_efficiency(chunk_size)
                
                # Memory utilization score
                chunk_memory_gb = (chunk_size * w * bytes_per_pixel) / (1024**3)
                memory_util = min(1.0, chunk_memory_gb / max_memory_per_chunk_gb)
                memory_score = memory_util if memory_util <= 0.8 else (1.6 - memory_util)
                
                # Composite score with weighted factors
                total_score = (
                    cache_eff * 0.3 +
                    kernel_eff * 0.25 +
                    parallel_eff * 0.25 +
                    memory_score * 0.2
                )
                
                if total_score > best_score:
                    best_score = total_score
                    best_chunk_size = chunk_size
            
            return best_chunk_size
        
        # Calculate the optimal chunk size
        optimal_chunk_size = calculate_adaptive_chunk_size()
        
        # Apply system-specific optimizations
        if platform.system() == "Windows":
            # Windows tends to have higher memory overhead
            optimal_chunk_size = int(optimal_chunk_size * 0.9)
        elif platform.system() == "Darwin":  # macOS
            # macOS has efficient memory management
            optimal_chunk_size = int(optimal_chunk_size * 1.1)
        
        # Apply final constraints and safety margins
        final_chunk_size = max(
            kernel_size * 2,  # Minimum for kernel operation
            min(
                optimal_chunk_size,
                h // 2,  # Don't use chunks larger than half the image
                2048  # Maximum chunk size for stability
            )
        )
        
        # Ensure chunk size allows for at least 2 chunks for parallelization
        if h // final_chunk_size < 2 and h > final_chunk_size:
            final_chunk_size = h // 2
        
        # Performance logging and statistics
        chunk_stats = {
            'calculated_chunk_size': final_chunk_size,
            'total_chunks': max(1, h // final_chunk_size),
            'memory_per_chunk_mb': (final_chunk_size * w * bytes_per_pixel) / (1024**2),
            'memory_utilization': min(1.0, (final_chunk_size * w * bytes_per_pixel) / 
                                     (max_memory_per_chunk_gb * 1024**3)),
            'cache_efficiency': calculate_cache_efficiency(final_chunk_size),
            'kernel_efficiency': calculate_kernel_efficiency(final_chunk_size, kernel_size),
            'parallel_efficiency': calculate_parallelization_efficiency(final_chunk_size),
            'available_memory_gb': available_memory_gb,
            'cpu_cores': cpu_count
        }
        
        # Store stats as function attribute for debugging
        calculate_optimal_chunk_size.last_stats = chunk_stats
        
        return final_chunk_size

    def determine_optimal_workers():
        """
        Determine the optimal number of worker threads for bilateral filtering based on
        system resources, image characteristics, and performance profiling.
        
        :return: Optimal number of worker threads for maximum performance
        """
        
        # Get comprehensive system information
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory_info = psutil.virtual_memory()
        
        # Get CPU frequency and performance characteristics
        try:
            cpu_freq = psutil.cpu_freq()
            base_freq_ghz = cpu_freq.current / 1000 if cpu_freq else 2.5
        except:
            base_freq_ghz = 2.5  # Default assumption
        
        # Memory constraints analysis
        available_memory_gb = memory_info.available / (1024**3)
        memory_pressure = 1.0 - (memory_info.available / memory_info.total)
        
        # Platform-specific optimizations
        system_platform = platform.system()
        processor_info = platform.processor()
        
        def analyze_cpu_architecture():
            """Analyze CPU architecture for optimal threading."""
            architecture_factors = {
                'thread_efficiency': 1.0,
                'memory_bandwidth': 1.0,
                'cache_efficiency': 1.0,
                'hyperthreading_benefit': 0.3
            }
            
            # Intel vs AMD optimizations
            if 'Intel' in processor_info:
                if 'i9' in processor_info or 'Xeon' in processor_info:
                    architecture_factors['thread_efficiency'] = 1.2
                    architecture_factors['hyperthreading_benefit'] = 0.4
                elif 'i7' in processor_info:
                    architecture_factors['thread_efficiency'] = 1.1
                    architecture_factors['hyperthreading_benefit'] = 0.35
                elif 'i5' in processor_info:
                    architecture_factors['thread_efficiency'] = 1.0
                    architecture_factors['hyperthreading_benefit'] = 0.25
            elif 'AMD' in processor_info:
                if 'Ryzen 9' in processor_info or 'Threadripper' in processor_info:
                    architecture_factors['thread_efficiency'] = 1.25
                    architecture_factors['memory_bandwidth'] = 1.2
                elif 'Ryzen 7' in processor_info:
                    architecture_factors['thread_efficiency'] = 1.15
                    architecture_factors['memory_bandwidth'] = 1.1
                elif 'Ryzen 5' in processor_info:
                    architecture_factors['thread_efficiency'] = 1.05
            
            # ARM processors (Apple Silicon, etc.)
            if 'arm' in processor_info.lower() or 'Apple' in processor_info:
                architecture_factors['thread_efficiency'] = 1.3
                architecture_factors['memory_bandwidth'] = 1.4
                architecture_factors['cache_efficiency'] = 1.2
                architecture_factors['hyperthreading_benefit'] = 0.1  # Less benefit from SMT
            
            return architecture_factors
        
        def calculate_memory_bandwidth_factor():
            """Calculate memory bandwidth limitations on threading."""
            # Estimate memory bandwidth requirements for bilateral filtering
            # Bilateral filtering is memory-intensive due to random access patterns
            
            memory_bandwidth_factor = 1.0
            
            if available_memory_gb < 4:
                memory_bandwidth_factor = 0.6  # Severe memory constraint
            elif available_memory_gb < 8:
                memory_bandwidth_factor = 0.8  # Moderate memory constraint
            elif available_memory_gb < 16:
                memory_bandwidth_factor = 1.0  # Adequate memory
            else:
                memory_bandwidth_factor = 1.2  # Abundant memory
            
            # Adjust for memory pressure
            if memory_pressure > 0.8:
                memory_bandwidth_factor *= 0.7
            elif memory_pressure > 0.6:
                memory_bandwidth_factor *= 0.85
            
            return memory_bandwidth_factor
        
        def calculate_workload_characteristics():
            """Analyze bilateral filtering workload characteristics."""
            # Bilateral filtering characteristics affecting threading:
            # - High memory bandwidth usage
            # - Irregular memory access patterns
            # - CPU-intensive Gaussian calculations
            # - Good parallelization potential with proper chunking
            
            workload_factors = {
                'cpu_intensive': 0.8,    # Moderate CPU intensity
                'memory_intensive': 1.2, # High memory usage
                'cache_sensitive': 1.1,  # Sensitive to cache performance
                'parallelizable': 0.9    # Good but not perfect parallelization
            }
            
            return workload_factors
        
        def benchmark_thread_performance():
            """Quick benchmark to determine optimal thread count."""
            # Simple benchmark using numpy operations similar to bilateral filtering
            test_size = 1000
            test_array = np.random.rand(test_size, test_size, 3).astype(np.float32)
            
            thread_performance = {}
            
            for num_threads in [1, 2, 4, 6, 8, 12, 16]:
                if num_threads > cpu_count_logical:
                    break
                    
                start_time = time.time()
                
                # Simulate bilateral filtering operations
                try:
                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                        futures = []
                        chunk_size = test_size // num_threads
                        
                        for i in range(num_threads):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, test_size)
                            if start_idx < end_idx:
                                future = executor.submit(
                                    np.exp, 
                                    -(test_array[start_idx:end_idx]**2) / 2
                                )
                                futures.append(future)
                        
                        # Wait for completion
                        for future in futures:
                            future.result()
                    
                    elapsed_time = time.time() - start_time
                    thread_performance[num_threads] = 1.0 / elapsed_time  # Performance score
                    
                except Exception:
                    thread_performance[num_threads] = 0  # Failed benchmark
            
            # Find optimal thread count from benchmark
            if thread_performance:
                best_threads = max(thread_performance.keys(), 
                                 key=lambda x: thread_performance[x])
                return best_threads, thread_performance
            else:
                return cpu_count_physical, {}
        
        def apply_platform_optimizations(base_threads):
            """Apply platform-specific threading optimizations."""
            optimized_threads = base_threads
            
            if system_platform == "Windows":
                # Windows has higher thread overhead
                if base_threads > 8:
                    optimized_threads = int(base_threads * 0.85)
                # Windows scheduler works well with power-of-2 thread counts
                power_of_2_options = [2, 4, 6, 8, 12, 16]
                optimized_threads = min(power_of_2_options, 
                                      key=lambda x: abs(x - optimized_threads))
                
            elif system_platform == "Darwin":  # macOS
                # macOS has efficient thread management
                if base_threads <= 8:
                    optimized_threads = min(base_threads + 1, cpu_count_logical)
                # macOS Grand Central Dispatch optimization
                optimized_threads = min(optimized_threads, 12)  # GCD sweet spot
                
            elif system_platform == "Linux":
                # Linux has excellent thread scaling
                optimized_threads = min(base_threads, cpu_count_logical)
                # Linux works well with logical core count
                if cpu_count_logical > cpu_count_physical:
                    optimized_threads = min(optimized_threads, 
                                          int(cpu_count_logical * 0.9))
            
            return max(1, optimized_threads)
        
        def calculate_bilateral_filter_optimal_threads():
            """Calculate optimal threads specifically for bilateral filtering."""
            
            # Get architecture analysis
            arch_factors = analyze_cpu_architecture()
            memory_factor = calculate_memory_bandwidth_factor()
            workload_factors = calculate_workload_characteristics()
            
            # Base calculation considering physical cores
            base_threads = cpu_count_physical
            
            # Apply architecture efficiency
            effective_threads = base_threads * arch_factors['thread_efficiency']
            
            # Consider hyperthreading benefits for bilateral filtering
            if cpu_count_logical > cpu_count_physical:
                ht_benefit = arch_factors['hyperthreading_benefit']
                ht_threads = cpu_count_logical - cpu_count_physical
                effective_threads += ht_threads * ht_benefit
            
            # Apply memory bandwidth constraints
            memory_limited_threads = effective_threads * memory_factor
            
            # Apply workload characteristics
            workload_score = (
                workload_factors['cpu_intensive'] * 
                workload_factors['parallelizable'] /
                workload_factors['memory_intensive']
            )
            optimal_threads = memory_limited_threads * workload_score
            
            # Apply cache efficiency considerations
            cache_limited_threads = min(optimal_threads, 
                                       cpu_count_physical * arch_factors['cache_efficiency'])
            
            return int(round(cache_limited_threads))
        
        # Calculate theoretical optimal threads
        theoretical_optimal = calculate_bilateral_filter_optimal_threads()
        
        # Run performance benchmark for validation
        try:
            benchmarked_optimal, benchmark_results = benchmark_thread_performance()
            
            # Weighted combination of theoretical and benchmarked results
            if benchmark_results:
                final_optimal = int(0.7 * theoretical_optimal + 0.3 * benchmarked_optimal)
            else:
                final_optimal = theoretical_optimal
                
        except Exception:
            final_optimal = theoretical_optimal
        
        # Apply platform-specific optimizations
        platform_optimized = apply_platform_optimizations(final_optimal)
        
        # Final safety constraints
        min_threads = 1
        max_threads = min(cpu_count_logical, 16)  # Cap at 16 for stability
        
        # Apply memory pressure limitations
        if memory_pressure > 0.9:
            max_threads = min(max_threads, 2)
        elif memory_pressure > 0.7:
            max_threads = min(max_threads, 4)
        
        final_threads = max(min_threads, min(platform_optimized, max_threads))
        
        # Store diagnostic information for debugging
        determine_optimal_workers.diagnostics = {
            'cpu_physical': cpu_count_physical,
            'cpu_logical': cpu_count_logical,
            'memory_gb': available_memory_gb,
            'memory_pressure': memory_pressure,
            'platform': system_platform,
            'processor': processor_info,
            'theoretical_optimal': theoretical_optimal,
            'benchmarked_optimal': benchmarked_optimal if 'benchmarked_optimal' in locals() else None,
            'platform_optimized': platform_optimized,
            'final_threads': final_threads,
            'architecture_factors': analyze_cpu_architecture(),
            'memory_factor': calculate_memory_bandwidth_factor(),
            'workload_factors': calculate_workload_characteristics()
        }
        
        return final_threads

    def process_bigdata_bilateral_chunk(image_chunk, chunk_data, weight_representation, intensity_approximator, pad_size, num_workers):
        """
        Process a bilateral filter chunk with advanced optimization techniques for big data scenarios.
        Uses vectorized operations, adaptive sampling, and intelligent memory management.
        
        :param image_chunk: Input image chunk to process
        :param chunk_data: Padded chunk data with metadata
        :param weight_representation: Compressed spatial weight representation
        :param intensity_approximator: ML-based intensity weight approximator
        :param pad_size: Padding size for kernel operations
        :param num_workers: Number of worker threads for parallel processing
        :return: Bilateral filtered chunk with high-quality edge preservation
        """
        if image_chunk is None or image_chunk.size == 0:
            return image_chunk
        
        h, w, c = image_chunk.shape
        filtered_chunk = np.zeros_like(image_chunk, dtype=np.float32)
        
        # Extract padded data and metadata
        padded_data = chunk_data['data']
        top_pad = chunk_data['top_pad']
        bottom_pad = chunk_data['bottom_pad']
        
        # Calculate effective processing region
        padded_h, padded_w = padded_data.shape[:2]
        kernel_size = weight_representation['kernel_size']
        half_kernel = weight_representation['half_kernel']
        
        def create_adaptive_sampling_strategy(image_chunk, intensity_approximator):
            """
            Create an adaptive sampling strategy based on image content analysis.
            Reduces computational load while maintaining quality in homogeneous regions.
            """
            h, w, c = image_chunk.shape
            
            # Analyze local variance for adaptive sampling
            variance_map = np.zeros((h, w), dtype=np.float32)
            sample_density_map = np.ones((h, w), dtype=np.float32)
            
            # Calculate local variance using efficient sliding window
            window_size = min(kernel_size // 2, 8)
            for y in range(0, h, window_size):
                for x in range(0, w, window_size):
                    y_end = min(y + window_size, h)
                    x_end = min(x + window_size, w)
                    
                    region = image_chunk[y:y_end, x:x_end]
                    local_variance = np.var(region)
                    variance_map[y:y_end, x:x_end] = local_variance
                    
                    # Determine sampling density based on variance
                    if local_variance < 10:  # Low variance - reduce sampling
                        sample_density_map[y:y_end, x:x_end] = 0.3
                    elif local_variance < 25:  # Medium variance - moderate sampling
                        sample_density_map[y:y_end, x:x_end] = 0.6
                    else:  # High variance - full sampling
                        sample_density_map[y:y_end, x:x_end] = 1.0
            
            # Edge detection for priority sampling
            gray_chunk = np.mean(image_chunk, axis=2)
            sobel_x = cv2.Sobel(gray_chunk, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_chunk, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Boost sampling near edges
            edge_threshold = np.percentile(edge_magnitude, 75)
            edge_mask = edge_magnitude > edge_threshold
            sample_density_map[edge_mask] = 1.0
            
            return {
                'variance_map': variance_map,
                'sample_density_map': sample_density_map,
                'edge_magnitude': edge_magnitude,
                'edge_mask': edge_mask
            }
        
        def create_vectorized_neighborhood_processor():
            """
            Create a vectorized processor for efficient neighborhood operations.
            Uses advanced indexing and broadcasting for maximum performance.
            """
            valid_coords = weight_representation['valid_coords']
            coord_to_weight = weight_representation['coord_to_weight']
            
            # Pre-compute neighbor offset arrays for vectorization
            neighbor_offsets = valid_coords.copy()
            spatial_weights = np.array([coord_to_weight.get((dy, dx), 0) 
                                       for dy, dx in neighbor_offsets])
            
            # Create efficient index arrays
            valid_mask = spatial_weights > 1e-6  # Filter out negligible weights
            active_offsets = neighbor_offsets[valid_mask]
            active_spatial_weights = spatial_weights[valid_mask]
            
            def process_pixel_vectorized(center_y, center_x, padded_data, center_pixel):
                """Process a single pixel using vectorized neighborhood operations."""
                
                # Calculate neighbor coordinates
                neighbor_y = center_y + active_offsets[:, 0]
                neighbor_x = center_x + active_offsets[:, 1]
                
                # Boundary checking with vectorization
                valid_neighbors = (
                    (neighbor_y >= 0) & (neighbor_y < padded_h) &
                    (neighbor_x >= 0) & (neighbor_x < padded_w)
                )
                
                if not np.any(valid_neighbors):
                    return center_pixel
                
                # Extract valid neighbors efficiently
                valid_y = neighbor_y[valid_neighbors]
                valid_x = neighbor_x[valid_neighbors]
                valid_spatial_w = active_spatial_weights[valid_neighbors]
                
                # Vectorized neighbor pixel extraction
                neighbor_pixels = padded_data[valid_y, valid_x]  # Shape: (n_neighbors, channels)
                
                # Batch compute intensity weights
                intensity_weights = intensity_approximator.batch_compute_weights(
                    center_pixel, neighbor_pixels
                )
                
                # Combined weights (spatial * intensity)
                combined_weights = valid_spatial_w * intensity_weights
                
                # Weighted sum calculation
                weight_sum = np.sum(combined_weights)
                if weight_sum > 1e-6:
                    weighted_pixel_sum = np.sum(
                        neighbor_pixels * combined_weights.reshape(-1, 1), axis=0
                    )
                    filtered_pixel = weighted_pixel_sum / weight_sum
                else:
                    filtered_pixel = center_pixel
                
                return filtered_pixel.astype(np.float32)
            
            return process_pixel_vectorized
        
        def create_block_processing_strategy(h, w, num_workers):
            """
            Create an intelligent block processing strategy for optimal parallelization.
            Balances load distribution with memory efficiency.
            """
            # Calculate optimal block dimensions
            total_pixels = h * w
            pixels_per_worker = total_pixels // num_workers
            
            # Determine block layout (prefer square-ish blocks for cache efficiency)
            block_height = max(8, int(np.sqrt(pixels_per_worker)))
            block_width = max(8, int(pixels_per_worker / block_height))
            
            # Adjust for actual image dimensions
            block_height = min(block_height, h)
            block_width = min(block_width, w)
            
            # Generate block coordinates with overlap handling
            blocks = []
            for y in range(0, h, block_height):
                for x in range(0, w, block_width):
                    y_end = min(y + block_height, h)
                    x_end = min(x + block_width, w)
                    
                    blocks.append({
                        'y_start': y,
                        'y_end': y_end,
                        'x_start': x,
                        'x_end': x_end,
                        'padded_y_start': y + top_pad,
                        'padded_y_end': y_end + top_pad,
                        'padded_x_start': x + pad_size,
                        'padded_x_end': x_end + pad_size,
                        'size': (y_end - y) * (x_end - x)
                    })
            
            # Sort blocks by size for better load balancing
            blocks.sort(key=lambda b: b['size'], reverse=True)
            
            return blocks
        
        def process_block_optimized(block_info, sampling_strategy, vectorized_processor):
            """
            Process a single block with all optimizations applied.
            Uses adaptive sampling and vectorized operations for maximum efficiency.
            """
            y_start, y_end = block_info['y_start'], block_info['y_end']
            x_start, x_end = block_info['x_start'], block_info['x_end']
            padded_y_start = block_info['padded_y_start']
            padded_x_start = block_info['padded_x_start']
            
            block_height = y_end - y_start
            block_width = x_end - x_start
            block_result = np.zeros((block_height, block_width, c), dtype=np.float32)
            
            # Get sampling strategy for this block
            block_sample_density = sampling_strategy['sample_density_map'][y_start:y_end, x_start:x_end]
            block_edge_mask = sampling_strategy['edge_mask'][y_start:y_end, x_start:x_end]
            
            # Process pixels with adaptive sampling
            for local_y in range(block_height):
                for local_x in range(block_width):
                    global_y = y_start + local_y
                    global_x = x_start + local_x
                    
                    # Adaptive sampling decision
                    sample_prob = block_sample_density[local_y, local_x]
                    is_edge = block_edge_mask[local_y, local_x]
                    
                    # Always process edge pixels, sample others based on probability
                    should_process = is_edge or (np.random.random() < sample_prob)
                    
                    if should_process:
                        # Calculate padded coordinates
                        padded_y = padded_y_start + local_y
                        padded_x = padded_x_start + local_x
                        
                        # Get center pixel
                        center_pixel = padded_data[padded_y, padded_x]
                        
                        # Process with vectorized neighborhood operation
                        filtered_pixel = vectorized_processor(
                            padded_y, padded_x, padded_data, center_pixel
                        )
                        
                        block_result[local_y, local_x] = filtered_pixel
                    else:
                        # Use original pixel for non-processed regions
                        block_result[local_y, local_x] = image_chunk[global_y, global_x]
            
            return block_info, block_result
        
        def apply_quality_enhancement(filtered_chunk, original_chunk, sampling_strategy):
            """
            Apply post-processing quality enhancement to maintain detail in under-sampled regions.
            Uses edge-aware interpolation and adaptive sharpening.
            """
            edge_mask = sampling_strategy['edge_mask']
            sample_density = sampling_strategy['sample_density_map']
            
            # Identify under-sampled regions
            under_sampled_mask = sample_density < 0.8
            
            if np.any(under_sampled_mask):
                # Apply edge-preserving smoothing to under-sampled regions
                for ch in range(c):
                    channel_data = filtered_chunk[:, :, ch]
                    
                    # Use bilateral filter for quality enhancement in under-sampled areas
                    enhanced_channel = cv2.bilateralFilter(
                        channel_data.astype(np.uint8),
                        5,  # Small kernel for local enhancement
                        15,  # Intensity sigma
                        15   # Spatial sigma
                    ).astype(np.float32)
                    
                    # Blend enhanced version in under-sampled regions
                    blend_factor = (1.0 - sample_density) * 0.3  # Conservative blending
                    filtered_chunk[:, :, ch] = (
                        filtered_chunk[:, :, ch] * (1 - blend_factor) +
                        enhanced_channel * blend_factor
                    )
            
            # Edge sharpening for enhanced detail preservation
            if np.any(edge_mask):
                # Apply unsharp masking to edge regions
                for ch in range(c):
                    channel_data = filtered_chunk[:, :, ch]
                    blurred = cv2.GaussianBlur(channel_data, (3, 3), 0.5)
                    unsharp_mask = channel_data - blurred
                    
                    # Apply sharpening only to edge regions
                    sharpening_strength = 0.2
                    filtered_chunk[:, :, ch] += unsharp_mask * edge_mask * sharpening_strength
            
            return filtered_chunk
        
        def process_with_memory_optimization():
            """
            Main processing function with advanced memory optimization techniques.
            Manages memory usage and implements intelligent caching strategies.
            """
            # Create processing strategies
            sampling_strategy = create_adaptive_sampling_strategy(image_chunk, intensity_approximator)
            vectorized_processor = create_vectorized_neighborhood_processor()
            block_strategy = create_block_processing_strategy(h, w, num_workers)
            
            # Process blocks in parallel with memory management
            if num_workers > 1 and len(block_strategy) > 1:
                # Parallel processing for multiple blocks
                def process_block_wrapper(block_info):
                    return process_block_optimized(block_info, sampling_strategy, vectorized_processor)
                
                # Use ThreadPoolExecutor with memory-conscious worker count
                effective_workers = min(num_workers, len(block_strategy))
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    # Submit jobs with memory monitoring
                    futures = [executor.submit(process_block_wrapper, block) 
                              for block in block_strategy]
                    
                    # Collect results as they complete
                    for future in futures:
                        block_info, block_result = future.result()
                        
                        # Copy result to output chunk
                        y_start, y_end = block_info['y_start'], block_info['y_end']
                        x_start, x_end = block_info['x_start'], block_info['x_end']
                        filtered_chunk[y_start:y_end, x_start:x_end] = block_result
                        
                        # Memory cleanup
                        del block_result
            else:
                # Sequential processing for small chunks or single worker
                for block_info in block_strategy:
                    block_info, block_result = process_block_optimized(
                        block_info, sampling_strategy, vectorized_processor
                    )
                    
                    y_start, y_end = block_info['y_start'], block_info['y_end']
                    x_start, x_end = block_info['x_start'], block_info['x_end']
                    filtered_chunk[y_start:y_end, x_start:x_end] = block_result
                    
                    del block_result
            
            # Apply post-processing quality enhancement
            filtered_chunk = apply_quality_enhancement(filtered_chunk, image_chunk, sampling_strategy)
            
            return filtered_chunk
        
        def validate_and_clamp_output(filtered_chunk):
            """
            Validate and clamp output values to ensure proper range and data integrity.
            """
            # Clamp to valid range
            filtered_chunk = np.clip(filtered_chunk, 0.0, 255.0)
            
            # Check for NaN or infinite values
            invalid_mask = ~np.isfinite(filtered_chunk)
            if np.any(invalid_mask):
                # Replace invalid values with original pixels
                filtered_chunk[invalid_mask] = image_chunk[invalid_mask]
            
            # Ensure proper data type
            filtered_chunk = filtered_chunk.astype(np.float32)
            
            return filtered_chunk
        
        # Execute main processing with comprehensive error handling
        try:
            # Check for degenerate cases
            if kernel_size <= 1:
                return image_chunk.astype(np.float32)
            
            if padded_data.shape[0] < kernel_size or padded_data.shape[1] < kernel_size:
                return image_chunk.astype(np.float32)
            
            # Main processing
            filtered_chunk = process_with_memory_optimization()
            
            # Validation and output preparation
            filtered_chunk = validate_and_clamp_output(filtered_chunk)
            
            # Performance statistics (for debugging/optimization)
            process_bigdata_bilateral_chunk.last_stats = {
                'chunk_shape': image_chunk.shape,
                'padded_shape': padded_data.shape,
                'kernel_size': kernel_size,
                'num_workers': num_workers,
                'processing_time': time.time() if hasattr(time, '_start_time') else 0,
                'memory_peak': psutil.Process().memory_info().rss / (1024**2)
            }
            
            return filtered_chunk
            
        except Exception as e:
            # Fallback to simple bilateral filter on error
            warnings.warn(f"Advanced bilateral processing failed: {e}. Using fallback method.")
            
            # Simple fallback processing
            fallback_result = np.zeros_like(image_chunk, dtype=np.float32)
            for ch in range(c):
                try:
                    fallback_result[:, :, ch] = cv2.bilateralFilter(
                        image_chunk[:, :, ch].astype(np.uint8),
                        min(kernel_size, 15),  # Limit kernel size for stability
                        intensity_approximator.base_sigma if hasattr(intensity_approximator, 'base_sigma') else 25,
                        weight_representation['spatial_sigma'] if 'spatial_sigma' in weight_representation else 25
                    ).astype(np.float32)
                except:
                    fallback_result[:, :, ch] = image_chunk[:, :, ch]
            
            return fallback_result

    def cleanup_chunk_memory(chunk_data, chunk_result):
        """
        Advanced memory cleanup function for bilateral filter chunk processing.
        Implements comprehensive memory management with garbage collection optimization,
        memory mapping cleanup, and system-level memory pressure management.
        
        :param chunk_data: Chunk data structure containing padded image data and metadata
        :param chunk_result: Processed chunk result that needs memory cleanup
        """
        
        def force_garbage_collection():
            """
            Force comprehensive garbage collection with multiple passes.
            Uses progressive collection strategy for optimal memory reclamation.
            """
            # Multiple garbage collection passes for thorough cleanup
            for generation in range(3):  # Collect all generations
                collected = gc.collect()
                if collected == 0:
                    break  # No more objects to collect
            
            # Force finalization of any pending objects
            if hasattr(gc, 'collect_gen'):
                for gen in [0, 1, 2]:
                    gc.collect_gen(gen)
        
        def cleanup_memory_mapped_arrays(data_object):
            """
            Safely cleanup memory-mapped arrays and temporary files.
            Handles both numpy memmap arrays and custom memory-mapped structures.
            """
            if data_object is None:
                return
            
            try:
                # Check if object has memory mapping attributes
                if hasattr(data_object, '_mmap'):
                    # Standard numpy memmap cleanup
                    data_object._mmap.close()
                    del data_object._mmap
                
                # Check for custom memory mapping cleanup
                if hasattr(data_object, '_temp_file_path'):
                    temp_file_path = data_object._temp_file_path
                    
                    # Close memory map if still open
                    if hasattr(data_object, 'flush'):
                        data_object.flush()
                    
                    # Delete the array reference
                    del data_object
                    
                    # Remove temporary file
                    try:
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                    except (OSError, PermissionError) as e:
                        warnings.warn(f"Could not remove temporary file {temp_file_path}: {e}")
                
                # Handle nested data structures with memory mapping
                if hasattr(data_object, '__dict__'):
                    for attr_name, attr_value in list(data_object.__dict__.items()):
                        if hasattr(attr_value, '_cleanup_required'):
                            cleanup_memory_mapped_arrays(attr_value)
                            setattr(data_object, attr_name, None)
                
            except Exception as e:
                warnings.warn(f"Memory mapped array cleanup failed: {e}")
        
        def cleanup_chunk_data_structure(chunk_data):
            """
            Clean up complex chunk data structures with nested arrays and metadata.
            Handles dictionaries, lists, and custom objects containing image data.
            """
            if chunk_data is None:
                return
            
            try:
                if isinstance(chunk_data, dict):
                    # Clean up dictionary-based chunk data
                    cleanup_items = ['data', 'padded_data', 'cache', 'buffer', 'temp_array']
                    
                    for key in cleanup_items:
                        if key in chunk_data:
                            array_data = chunk_data[key]
                            
                            # Clean up numpy arrays
                            if isinstance(array_data, np.ndarray):
                                cleanup_memory_mapped_arrays(array_data)
                                chunk_data[key] = None
                            
                            # Clean up nested structures
                            elif hasattr(array_data, '__dict__'):
                                cleanup_memory_mapped_arrays(array_data)
                                chunk_data[key] = None
                    
                    # Clear metadata that might hold references
                    metadata_keys = ['processing_cache', 'weight_cache', 'neighbor_cache']
                    for key in metadata_keys:
                        if key in chunk_data:
                            chunk_data[key] = None
                    
                    # Clean up any remaining large objects
                    for key, value in list(chunk_data.items()):
                        if isinstance(value, np.ndarray) and value.nbytes > 1024 * 1024:  # > 1MB
                            cleanup_memory_mapped_arrays(value)
                            chunk_data[key] = None
                
                elif isinstance(chunk_data, (list, tuple)):
                    # Clean up list/tuple-based chunk data
                    for i, item in enumerate(chunk_data):
                        if isinstance(item, np.ndarray):
                            cleanup_memory_mapped_arrays(item)
                            chunk_data[i] = None
                        elif hasattr(item, '__dict__'):
                            cleanup_memory_mapped_arrays(item)
                            chunk_data[i] = None
                
                elif hasattr(chunk_data, '__dict__'):
                    # Clean up object-based chunk data
                    cleanup_memory_mapped_arrays(chunk_data)
            
            except Exception as e:
                warnings.warn(f"Chunk data structure cleanup failed: {e}")
        
        def cleanup_processing_caches():
            """
            Clean up global processing caches and temporary data structures.
            Removes cached weights, lookup tables, and intermediate results.
            """
            try:
                # Clean up function-level caches if they exist
                cache_functions = [
                    'create_compressed_weight_representation',
                    'create_ml_intensity_approximator',
                    'calculate_optimal_chunk_size',
                    'determine_optimal_workers'
                ]
                
                for func_name in cache_functions:
                    if func_name in globals():
                        func = globals()[func_name]
                        
                        # Clear function attributes that might cache data
                        cache_attrs = ['_cache', '_last_result', 'cache', 'stats', 'diagnostics']
                        for attr in cache_attrs:
                            if hasattr(func, attr):
                                setattr(func, attr, None)
                
                # Clear thread-local storage if it exists
                if hasattr(threading, 'local'):
                    try:
                        thread_local = threading.local()
                        if hasattr(thread_local, '__dict__'):
                            thread_local.__dict__.clear()
                    except:
                        pass
            
            except Exception as e:
                warnings.warn(f"Processing cache cleanup failed: {e}")
        
        def monitor_memory_pressure():
            """
            Monitor system memory pressure and apply appropriate cleanup strategies.
            Implements progressive cleanup based on available memory.
            """
            try:
                memory_info = psutil.virtual_memory()
                memory_pressure = 1.0 - (memory_info.available / memory_info.total)
                
                # Apply cleanup strategy based on memory pressure
                if memory_pressure > 0.9:
                    # Critical memory pressure - aggressive cleanup
                    force_garbage_collection()
                    
                    # Clear all possible caches
                    cleanup_processing_caches()
                    
                    # Force system-level memory cleanup if available
                    if platform.system() == "Linux":
                        try:
                            # Sync and drop caches (requires appropriate permissions)
                            os.sync()
                        except:
                            pass
                    
                elif memory_pressure > 0.7:
                    # High memory pressure - moderate cleanup
                    force_garbage_collection()
                    
                elif memory_pressure > 0.5:
                    # Moderate memory pressure - standard cleanup
                    gc.collect()
                
                # Log memory status for debugging
                monitor_memory_pressure.last_pressure = memory_pressure
                monitor_memory_pressure.last_available_mb = memory_info.available / (1024**2)
                
            except Exception as e:
                warnings.warn(f"Memory pressure monitoring failed: {e}")
        
        def cleanup_opencv_memory():
            """
            Clean up OpenCV-specific memory allocations and caches.
            Addresses OpenCV's internal memory management.
            """
            try:
                # Clear OpenCV's internal caches
                if hasattr(cv2, 'setUseOptimized'):
                    # Temporarily disable optimizations to clear caches
                    was_optimized = cv2.useOptimized()
                    cv2.setUseOptimized(False)
                    cv2.setUseOptimized(was_optimized)
                
                # Force OpenCV garbage collection if available
                if hasattr(cv2, 'destroyAllWindows'):
                    cv2.destroyAllWindows()  # Also cleans some internal buffers
                
            except Exception as e:
                warnings.warn(f"OpenCV memory cleanup failed: {e}")
        
        def cleanup_numpy_memory():
            """
            Clean up NumPy-specific memory allocations and optimize memory layout.
            Handles NumPy's memory pools and temporary arrays.
            """
            try:
                # Clear NumPy's internal caches
                if hasattr(np.core, '_internal'):
                    # Clear any cached functions
                    if hasattr(np.core._internal, 'clear_cache'):
                        np.core._internal.clear_cache()
                
                # Force numpy to release unused memory pools
                if hasattr(np, '_NoValue'):
                    # This is a hack to trigger numpy memory cleanup
                    temp_array = np.array([1, 2, 3])
                    del temp_array
                
            except Exception as e:
                warnings.warn(f"NumPy memory cleanup failed: {e}")
        
        def perform_comprehensive_cleanup():
            """
            Perform comprehensive memory cleanup with all strategies applied.
            This is the main cleanup orchestrator function.
            """
            # 1. Clean up chunk-specific data
            cleanup_chunk_data_structure(chunk_data)
            cleanup_chunk_data_structure(chunk_result)
            
            # 2. Clean up memory-mapped arrays
            cleanup_memory_mapped_arrays(chunk_data)
            cleanup_memory_mapped_arrays(chunk_result)
            
            # 3. Clean up library-specific memory
            cleanup_opencv_memory()
            cleanup_numpy_memory()
            
            # 4. Clean up processing caches
            cleanup_processing_caches()
            
            # 5. Monitor and respond to memory pressure
            monitor_memory_pressure()
            
            # 6. Final garbage collection
            force_garbage_collection()
            
            # 7. Memory defragmentation hint (platform-specific)
            try:
                if platform.system() == "Windows":
                    # Windows memory management hint
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                elif platform.system() == "Linux":
                    # Linux memory management hint
                    try:
                        with open('/proc/self/clear_refs', 'w') as f:
                            f.write('1')
                    except:
                        pass
            except Exception:
                pass  # Platform-specific optimizations are optional
        
        def validate_cleanup_success():
            """
            Validate that memory cleanup was successful and log statistics.
            Provides feedback on cleanup effectiveness.
            """
            try:
                # Check memory usage after cleanup
                process = psutil.Process()
                memory_info = process.memory_info()
                
                # Store cleanup statistics
                cleanup_stats = {
                    'memory_rss_mb': memory_info.rss / (1024**2),
                    'memory_vms_mb': memory_info.vms / (1024**2),
                    'gc_collected': gc.get_stats() if hasattr(gc, 'get_stats') else None,
                    'cleanup_timestamp': time.time()
                }
                
                # Store stats as function attribute for monitoring
                cleanup_chunk_memory.last_stats = cleanup_stats
                
            except Exception as e:
                warnings.warn(f"Cleanup validation failed: {e}")
        
        # Execute comprehensive cleanup with error handling
        try:
            # Perform the main cleanup operations
            perform_comprehensive_cleanup()
            
            # Validate cleanup effectiveness
            validate_cleanup_success()
            
            # Additional safety: ensure references are cleared
            chunk_data = None
            chunk_result = None
            
        except Exception as e:
            # Emergency fallback cleanup
            warnings.warn(f"Comprehensive cleanup failed: {e}. Attempting emergency cleanup.")
            
            try:
                # Basic cleanup as fallback
                if chunk_data is not None:
                    del chunk_data
                if chunk_result is not None:
                    del chunk_result
                
                # Force garbage collection
                for _ in range(3):
                    gc.collect()
                    
            except Exception as emergency_error:
                warnings.warn(f"Emergency cleanup also failed: {emergency_error}")
        
        finally:
            # Ensure local variables are cleared
            locals().clear() if hasattr(locals(), 'clear') else None

    def apply_distributed_post_processing(image, filtered_image, spatial_sigma, intensity_sigma):
        """
        Apply advanced distributed post-processing for bilateral filter enhancement.
        Implements edge-aware refinement, adaptive sharpening, noise reduction,
        and quality optimization using distributed computing techniques.
        
        :param image: Original input image (float32, 0-255 range)
        :param filtered_image: Bilateral filtered image to enhance
        :param spatial_sigma: Spatial standard deviation from bilateral filter
        :param intensity_sigma: Intensity standard deviation from bilateral filter
        :return: Enhanced filtered image with improved quality and detail preservation
        """
        if image is None or filtered_image is None:
            return filtered_image
        
        h, w, c = image.shape
        enhanced_image = filtered_image.copy()
        
        def analyze_image_characteristics(original, filtered):
            """
            Analyze image characteristics to determine optimal post-processing strategy.
            Uses statistical analysis and edge detection for adaptive enhancement.
            """
            # Calculate image statistics
            original_mean = np.mean(original)
            filtered_mean = np.mean(filtered)
            
            # Edge analysis using multiple detectors
            gray_original = np.mean(original, axis=2) if c > 1 else original.squeeze()
            gray_filtered = np.mean(filtered, axis=2) if c > 1 else filtered.squeeze()
            
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray_original, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_original, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Canny edge detection for fine details
            gray_uint8 = gray_original.astype(np.uint8)
            canny_edges = cv2.Canny(gray_uint8, 50, 150)
            
            # Texture analysis using Local Binary Patterns
            def calculate_lbp_texture(image_gray):
                """Calculate texture complexity using simplified LBP."""
                texture_score = 0
                for y in range(1, image_gray.shape[0] - 1):
                    for x in range(1, image_gray.shape[1] - 1):
                        center = image_gray[y, x]
                        neighbors = [
                            image_gray[y-1, x-1], image_gray[y-1, x], image_gray[y-1, x+1],
                            image_gray[y, x+1], image_gray[y+1, x+1], image_gray[y+1, x],
                            image_gray[y+1, x-1], image_gray[y, x-1]
                        ]
                        pattern = sum([1 if neighbor >= center else 0 for neighbor in neighbors])
                        texture_score += pattern
                return texture_score / ((image_gray.shape[0] - 2) * (image_gray.shape[1] - 2))
            
            texture_complexity = calculate_lbp_texture(gray_original)
            
            # Noise analysis
            noise_level = np.std(gray_original - cv2.GaussianBlur(gray_original, (3, 3), 0.5))
            
            # Detail preservation analysis
            detail_loss = np.mean(np.abs(gray_original - gray_filtered))
            
            characteristics = {
                'brightness_change': filtered_mean - original_mean,
                'edge_density': np.mean(edge_magnitude > np.percentile(edge_magnitude, 75)),
                'fine_edge_density': np.mean(canny_edges > 0),
                'texture_complexity': texture_complexity,
                'noise_level': noise_level,
                'detail_loss': detail_loss,
                'contrast_ratio': np.std(gray_original) / (np.mean(gray_original) + 1e-8),
                'edge_strength': np.percentile(edge_magnitude, 90)
            }
            
            return characteristics, edge_magnitude, canny_edges
        
        def create_adaptive_enhancement_maps(characteristics, edge_magnitude, canny_edges, h, w):
            """
            Create adaptive enhancement maps for targeted post-processing.
            Uses image analysis to determine enhancement strength per region.
            """
            # Sharpening map based on edge density and detail loss
            sharpening_map = np.ones((h, w), dtype=np.float32)
            
            # Increase sharpening in areas with high detail loss
            if characteristics['detail_loss'] > 5:
                detail_factor = min(2.0, characteristics['detail_loss'] / 10.0)
                sharpening_map *= detail_factor
            
            # Enhance fine edges
            fine_edge_mask = canny_edges > 0
            sharpening_map[fine_edge_mask] *= 1.5
            
            # Reduce sharpening in smooth areas to avoid artifacts
            smooth_areas = edge_magnitude < np.percentile(edge_magnitude, 25)
            sharpening_map[smooth_areas] *= 0.3
            
            # Noise reduction map
            noise_reduction_map = np.ones((h, w), dtype=np.float32)
            if characteristics['noise_level'] > 10:
                noise_factor = min(1.5, characteristics['noise_level'] / 20.0)
                noise_reduction_map *= noise_factor
            
            # Reduce noise reduction on edges to preserve detail
            strong_edges = edge_magnitude > np.percentile(edge_magnitude, 85)
            noise_reduction_map[strong_edges] *= 0.5
            
            # Contrast enhancement map
            contrast_map = np.ones((h, w), dtype=np.float32)
            if characteristics['contrast_ratio'] < 0.3:
                contrast_map *= 1.3  # Boost contrast in low-contrast images
            
            # Edge preservation map
            edge_preservation_map = np.zeros((h, w), dtype=np.float32)
            edge_preservation_map[edge_magnitude > np.percentile(edge_magnitude, 70)] = 1.0
            
            return {
                'sharpening': sharpening_map,
                'noise_reduction': noise_reduction_map,
                'contrast': contrast_map,
                'edge_preservation': edge_preservation_map
            }
        
        def apply_distributed_unsharp_masking(image, filtered_image, sharpening_map, num_workers=None):
            """
            Apply distributed unsharp masking for edge enhancement with parallel processing.
            Uses adaptive sharpening strength based on local image characteristics.
            """
            if num_workers is None:
                num_workers = min(4, psutil.cpu_count())
            
            h, w, c = image.shape
            enhanced = filtered_image.copy()
            
            def process_channel_block(args):
                channel, y_start, y_end, x_start, x_end = args
                
                # Extract region
                region = filtered_image[y_start:y_end, x_start:x_end, channel]
                sharpening_region = sharpening_map[y_start:y_end, x_start:x_end]
                
                # Create unsharp mask
                blurred = cv2.GaussianBlur(region, (3, 3), 0.8)
                unsharp_mask = region - blurred
                
                # Apply adaptive sharpening
                sharpening_strength = 0.4  # Base strength
                enhanced_region = region + unsharp_mask * sharpening_region * sharpening_strength
                
                # Clamp values
                enhanced_region = np.clip(enhanced_region, 0, 255)
                
                return channel, y_start, y_end, x_start, x_end, enhanced_region
            
            # Create processing blocks
            block_size = max(64, min(256, h // num_workers))
            blocks = []
            
            for ch in range(c):
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        y_end = min(y + block_size, h)
                        x_end = min(x + block_size, w)
                        blocks.append((ch, y, y_end, x, x_end))
            
            # Process blocks in parallel
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(process_channel_block, blocks))
            
            # Combine results
            for channel, y_start, y_end, x_start, x_end, enhanced_region in results:
                enhanced[y_start:y_end, x_start:x_end, channel] = enhanced_region
            
            return enhanced
        
        def apply_edge_aware_noise_reduction(image, filtered_image, noise_reduction_map, edge_preservation_map):
            """
            Apply edge-aware noise reduction that preserves important details.
            Uses bilateral filtering with adaptive parameters.
            """
            h, w, c = image.shape
            denoised = filtered_image.copy()
            
            # Calculate adaptive noise reduction parameters
            avg_noise_reduction = np.mean(noise_reduction_map)
            
            if avg_noise_reduction > 1.1:  # Apply noise reduction only if needed
                for ch in range(c):
                    channel_data = filtered_image[:, :, ch].astype(np.uint8)
                    
                    # Apply bilateral filter for noise reduction
                    denoised_channel = cv2.bilateralFilter(
                        channel_data,
                        5,  # Small kernel for local denoising
                        int(intensity_sigma * 0.7),  # Reduced intensity sigma
                        int(spatial_sigma * 0.5)     # Reduced spatial sigma
                    ).astype(np.float32)
                    
                    # Blend with original based on noise reduction map and edge preservation
                    blend_factor = (noise_reduction_map - 1.0) * 0.3  # Conservative blending
                    blend_factor = np.clip(blend_factor, 0, 0.5)
                    
                    # Preserve edges by reducing blending on edge areas
                    blend_factor *= (1.0 - edge_preservation_map * 0.7)
                    
                    denoised[:, :, ch] = (
                        filtered_image[:, :, ch] * (1 - blend_factor) +
                        denoised_channel * blend_factor
                    )
            
            return denoised
        
        def apply_adaptive_contrast_enhancement(image, processed_image, contrast_map):
            """
            Apply adaptive contrast enhancement using CLAHE and local adjustments.
            Enhances contrast while avoiding over-saturation.
            """
            h, w, c = image.shape
            enhanced = processed_image.copy()
            
            avg_contrast_factor = np.mean(contrast_map)
            
            if avg_contrast_factor > 1.05:  # Apply contrast enhancement if needed
                # Convert to LAB color space for better contrast control
                lab_image = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
                
                # Convert back to RGB
                enhanced_contrast = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB).astype(np.float32)
                
                # Blend with original based on contrast map
                blend_factor = (contrast_map - 1.0) * 0.4  # Conservative blending
                blend_factor = np.clip(blend_factor, 0, 0.6)
                
                for ch in range(c):
                    enhanced[:, :, ch] = (
                        processed_image[:, :, ch] * (1 - blend_factor) +
                        enhanced_contrast[:, :, ch] * blend_factor
                    )
            
            return enhanced
        
        def apply_detail_recovery(original, filtered, enhanced, characteristics):
            """
            Recover fine details that may have been lost during bilateral filtering.
            Uses frequency domain analysis and selective detail enhancement.
            """
            h, w, c = original.shape
            detail_recovered = enhanced.copy()
            
            # Check if detail recovery is needed
            if characteristics['detail_loss'] > 3:
                # Calculate detail map from original image
                for ch in range(c):
                    original_channel = original[:, :, ch]
                    filtered_channel = filtered[:, :, ch]
                    enhanced_channel = enhanced[:, :, ch]
                    
                    # Extract high-frequency details from original
                    high_freq_original = original_channel - cv2.GaussianBlur(original_channel, (5, 5), 1.5)
                    high_freq_filtered = filtered_channel - cv2.GaussianBlur(filtered_channel, (5, 5), 1.5)
                    
                    # Calculate detail loss
                    detail_difference = high_freq_original - high_freq_filtered
                    
                    # Selectively recover details
                    recovery_strength = min(0.3, characteristics['detail_loss'] / 20.0)
                    detail_recovered[:, :, ch] = enhanced_channel + detail_difference * recovery_strength
            
            return detail_recovered
        
        def apply_color_balance_correction(original, processed):
            """
            Apply color balance correction to maintain color fidelity.
            Compensates for color shifts introduced during processing.
            """
            h, w, c = original.shape
            
            if c >= 3:  # Only for color images
                # Calculate color statistics
                original_means = [np.mean(original[:, :, ch]) for ch in range(c)]
                processed_means = [np.mean(processed[:, :, ch]) for ch in range(c)]
                
                # Calculate correction factors
                correction_factors = [
                    original_means[ch] / (processed_means[ch] + 1e-8) 
                    for ch in range(c)
                ]
                
                # Apply gentle correction to avoid over-correction
                max_correction = 1.1  # Limit correction strength
                correction_factors = [
                    np.clip(factor, 1/max_correction, max_correction) 
                    for factor in correction_factors
                ]
                
                # Apply correction
                color_corrected = processed.copy()
                for ch in range(c):
                    if abs(correction_factors[ch] - 1.0) > 0.02:  # Only if significant change
                        color_corrected[:, :, ch] *= correction_factors[ch]
                
                return np.clip(color_corrected, 0, 255)
            
            return processed
        
        def quality_assessment_and_refinement(original, processed):
            """
            Perform quality assessment and apply final refinements.
            Uses image quality metrics to ensure optimal results.
            """
            # Calculate quality metrics
            mse = np.mean((original - processed) ** 2)
            psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # Edge preservation metric
            original_edges = cv2.Canny(np.mean(original, axis=2).astype(np.uint8), 50, 150)
            processed_edges = cv2.Canny(np.mean(processed, axis=2).astype(np.uint8), 50, 150)
            edge_preservation = np.sum(original_edges & processed_edges) / (np.sum(original_edges) + 1e-8)
            
            # Apply final refinement if quality is below threshold
            refined = processed.copy()
            
            if psnr < 25 or edge_preservation < 0.7:
                # Apply gentle smoothing to reduce artifacts
                for ch in range(processed.shape[2]):
                    refined[:, :, ch] = cv2.bilateralFilter(
                        refined[:, :, ch].astype(np.uint8),
                        3, 10, 10
                    ).astype(np.float32)
            
            # Store quality metrics
            quality_assessment_and_refinement.metrics = {
                'psnr': psnr,
                'edge_preservation': edge_preservation,
                'mse': mse
            }
            
            return refined
        
        def parallel_post_processing_pipeline():
            """
            Execute the complete post-processing pipeline with parallel optimization.
            Orchestrates all enhancement steps for maximum quality and efficiency.
            """
            # Step 1: Analyze image characteristics
            characteristics, edge_magnitude, canny_edges = analyze_image_characteristics(image, filtered_image)
            
            # Step 2: Create adaptive enhancement maps
            enhancement_maps = create_adaptive_enhancement_maps(
                characteristics, edge_magnitude, canny_edges, h, w
            )
            
            # Step 3: Apply distributed unsharp masking for edge enhancement
            enhanced = apply_distributed_unsharp_masking(
                image, filtered_image, enhancement_maps['sharpening']
            )
            
            # Step 4: Apply edge-aware noise reduction
            enhanced = apply_edge_aware_noise_reduction(
                image, enhanced, 
                enhancement_maps['noise_reduction'], 
                enhancement_maps['edge_preservation']
            )
            
            # Step 5: Apply adaptive contrast enhancement
            enhanced = apply_adaptive_contrast_enhancement(
                image, enhanced, enhancement_maps['contrast']
            )
            
            # Step 6: Apply detail recovery
            enhanced = apply_detail_recovery(image, filtered_image, enhanced, characteristics)
            
            # Step 7: Apply color balance correction
            enhanced = apply_color_balance_correction(image, enhanced)
            
            # Step 8: Final quality assessment and refinement
            enhanced = quality_assessment_and_refinement(image, enhanced)
            
            # Step 9: Final value clamping and validation
            enhanced = np.clip(enhanced, 0.0, 255.0).astype(np.float32)
            
            # Check for any invalid values
            invalid_mask = ~np.isfinite(enhanced)
            if np.any(invalid_mask):
                enhanced[invalid_mask] = filtered_image[invalid_mask]
            
            return enhanced
        
        # Execute the complete post-processing pipeline
        try:
            enhanced_image = parallel_post_processing_pipeline()
            
            # Store processing statistics for debugging/optimization
            apply_distributed_post_processing.stats = {
                'processing_completed': True,
                'enhancement_applied': True,
                'final_shape': enhanced_image.shape,
                'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2),
                'processing_timestamp': time.time()
            }
            
            return enhanced_image
            
        except Exception as e:
            # Fallback to simple post-processing on error
            warnings.warn(f"Advanced post-processing failed: {e}. Using simple enhancement.")
            
            try:
                # Simple fallback enhancement
                simple_enhanced = filtered_image.copy()
                
                # Basic unsharp masking
                for ch in range(c):
                    channel_data = filtered_image[:, :, ch]
                    blurred = cv2.GaussianBlur(channel_data, (3, 3), 0.8)
                    unsharp_mask = channel_data - blurred
                    simple_enhanced[:, :, ch] = channel_data + unsharp_mask * 0.2
                
                # Clamp values
                simple_enhanced = np.clip(simple_enhanced, 0.0, 255.0)
                
                return simple_enhanced.astype(np.float32)
                
            except Exception as fallback_error:
                warnings.warn(f"Fallback post-processing also failed: {fallback_error}")
                return filtered_image

    def convert_to_standard_array(filtered_image):
        """
        Convert memory-mapped or specialized array formats to standard NumPy arrays
        with comprehensive optimization, validation, and memory management.
        
        Handles various array types including memory-mapped arrays, CUDA arrays,
        sparse arrays, and custom data structures while ensuring optimal memory
        usage and data integrity.
        
        :param filtered_image: Input array in various formats (memmap, CUDA, sparse, etc.)
        :return: Standard NumPy array (float32, 0-255 range) with validated data
        """
        
        if filtered_image is None:
            return None
        
        def detect_array_type(array):
            """
            Detect the type of input array for appropriate conversion strategy.
            Supports multiple array formats and custom data structures.
            """
            array_info = {
                'type': type(array).__name__,
                'is_memmap': isinstance(array, np.memmap),
                'is_cuda': hasattr(array, 'cuda') or 'cuda' in str(type(array)).lower(),
                'is_sparse': hasattr(array, 'toarray') or 'sparse' in str(type(array)).lower(),
                'is_dask': 'dask' in str(type(array)).lower(),
                'has_custom_cleanup': hasattr(array, '_cleanup_required'),
                'memory_mapped': hasattr(array, '_mmap') or hasattr(array, 'filename'),
                'is_standard_numpy': isinstance(array, np.ndarray) and not isinstance(array, np.memmap),
                'dtype': getattr(array, 'dtype', None),
                'shape': getattr(array, 'shape', None),
                'size_mb': (array.nbytes / (1024**2)) if hasattr(array, 'nbytes') else 0
            }
            
            return array_info
        
        def validate_array_integrity(array):
            """
            Validate array data integrity and detect potential issues.
            Checks for NaN, infinite values, proper range, and data corruption.
            """
            if not hasattr(array, 'shape') or len(array.shape) == 0:
                return False, "Array has invalid shape"
            
            try:
                # Check for basic array properties
                if array.size == 0:
                    return False, "Array is empty"
                
                # Sample-based validation for large arrays (performance optimization)
                if array.size > 10**7:  # > 10M elements
                    # Sample 1% of the array for validation
                    sample_size = max(1000, array.size // 100)
                    flat_array = array.flatten()
                    sample_indices = np.random.choice(array.size, sample_size, replace=False)
                    sample_data = flat_array[sample_indices]
                else:
                    sample_data = array.flatten()
                
                # Check for NaN values
                nan_count = np.sum(np.isnan(sample_data))
                if nan_count > 0:
                    return False, f"Array contains {nan_count} NaN values"
                
                # Check for infinite values
                inf_count = np.sum(np.isinf(sample_data))
                if inf_count > 0:
                    return False, f"Array contains {inf_count} infinite values"
                
                # Check value range for image data
                min_val, max_val = np.min(sample_data), np.max(sample_data)
                if min_val < -10 or max_val > 300:  # Allow some tolerance
                    return False, f"Array values out of expected range: [{min_val:.2f}, {max_val:.2f}]"
                
                # Check for data type issues
                if hasattr(array, 'dtype'):
                    if array.dtype.kind == 'c':  # Complex numbers
                        return False, "Array contains complex numbers"
                    if array.dtype.kind == 'U' or array.dtype.kind == 'S':  # String data
                        return False, "Array contains string data"
                
                return True, "Array validation passed"
                
            except Exception as e:
                return False, f"Validation failed with error: {str(e)}"
        
        def convert_memmap_array(memmap_array):
            """
            Convert memory-mapped arrays to standard arrays with intelligent copying.
            Handles large memory-mapped files with chunked processing.
            """
            try:
                array_info = detect_array_type(memmap_array)
                
                # For large arrays, use chunked copying to avoid memory overflow
                if array_info['size_mb'] > 500:  # > 500MB
                    return convert_large_memmap_chunked(memmap_array)
                else:
                    # Direct copy for smaller arrays
                    standard_array = np.array(memmap_array, dtype=np.float32, copy=True)
                    return standard_array
                    
            except Exception as e:
                warnings.warn(f"Memory-mapped array conversion failed: {e}")
                # Fallback: try direct access
                try:
                    return np.array(memmap_array[:], dtype=np.float32)
                except:
                    return None
        
        def convert_large_memmap_chunked(memmap_array):
            """
            Convert large memory-mapped arrays using chunked processing.
            Minimizes memory usage for very large arrays.
            """
            shape = memmap_array.shape
            dtype = np.float32
            
            # Create output array
            standard_array = np.zeros(shape, dtype=dtype)
            
            # Calculate optimal chunk size
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            max_chunk_memory_gb = min(available_memory_gb * 0.2, 1.0)  # Use max 20% or 1GB
            
            bytes_per_element = np.dtype(dtype).itemsize * shape[-1] if len(shape) > 2 else np.dtype(dtype).itemsize
            max_chunk_elements = int((max_chunk_memory_gb * 1024**3) / bytes_per_element)
            
            if len(shape) >= 2:
                chunk_rows = max(1, min(shape[0], max_chunk_elements // shape[1]))
            else:
                chunk_rows = max_chunk_elements
            
            # Process in chunks
            for start_row in range(0, shape[0], chunk_rows):
                end_row = min(start_row + chunk_rows, shape[0])
                
                try:
                    # Copy chunk
                    chunk_data = memmap_array[start_row:end_row]
                    standard_array[start_row:end_row] = chunk_data.astype(dtype)
                    
                    # Memory cleanup
                    del chunk_data
                    
                except Exception as e:
                    warnings.warn(f"Chunk conversion failed for rows {start_row}-{end_row}: {e}")
                    # Fill with zeros as fallback
                    standard_array[start_row:end_row] = 0
            
            return standard_array
        
        def convert_cuda_array(cuda_array):
            """
            Convert CUDA arrays to CPU NumPy arrays.
            Handles various CUDA array types and GPU memory management.
            """
            try:
                # Try common CUDA array conversion methods
                if hasattr(cuda_array, 'cpu'):
                    # PyTorch-style CUDA arrays
                    cpu_array = cuda_array.cpu().numpy().astype(np.float32)
                    return cpu_array
                elif hasattr(cuda_array, 'get'):
                    # CuPy-style arrays
                    cpu_array = cuda_array.get().astype(np.float32)
                    return cpu_array
                elif hasattr(cuda_array, 'to_cpu'):
                    # Custom CUDA implementations
                    cpu_array = cuda_array.to_cpu().astype(np.float32)
                    return cpu_array
                else:
                    # Try generic conversion
                    cpu_array = np.array(cuda_array).astype(np.float32)
                    return cpu_array
                    
            except Exception as e:
                warnings.warn(f"CUDA array conversion failed: {e}")
                return None
        
        def convert_sparse_array(sparse_array):
            """
            Convert sparse arrays to dense NumPy arrays.
            Handles various sparse matrix formats from scipy.sparse.
            """
            try:
                if hasattr(sparse_array, 'toarray'):
                    # scipy.sparse matrices
                    dense_array = sparse_array.toarray().astype(np.float32)
                    return dense_array
                elif hasattr(sparse_array, 'todense'):
                    # Some sparse implementations use todense()
                    dense_array = np.array(sparse_array.todense()).astype(np.float32)
                    return dense_array
                else:
                    # Try generic conversion
                    dense_array = np.array(sparse_array).astype(np.float32)
                    return dense_array
                    
            except Exception as e:
                warnings.warn(f"Sparse array conversion failed: {e}")
                return None
        
        def convert_dask_array(dask_array):
            """
            Convert Dask arrays to NumPy arrays with memory-efficient computation.
            Handles distributed Dask arrays and lazy evaluation.
            """
            try:
                # Use Dask's compute method with memory management
                if hasattr(dask_array, 'compute'):
                    # Set appropriate scheduler for memory efficiency
                    with dask_array.__dask_scheduler__('threads'):
                        computed_array = dask_array.compute()
                        return computed_array.astype(np.float32)
                else:
                    # Fallback to direct conversion
                    computed_array = np.array(dask_array).astype(np.float32)
                    return computed_array
                    
            except Exception as e:
                warnings.warn(f"Dask array conversion failed: {e}")
                return None
        
        def apply_data_corrections(array):
            """
            Apply data corrections to ensure proper image data format.
            Fixes common issues with pixel values, data types, and ranges.
            """
            if array is None:
                return None
            
            corrected_array = array.copy()
            
            # Fix NaN and infinite values
            invalid_mask = ~np.isfinite(corrected_array)
            if np.any(invalid_mask):
                # Replace invalid values with local mean
                for ch in range(corrected_array.shape[-1]):
                    channel_data = corrected_array[:, :, ch] if len(corrected_array.shape) > 2 else corrected_array
                    valid_mean = np.mean(channel_data[np.isfinite(channel_data)])
                    
                    if len(corrected_array.shape) > 2:
                        corrected_array[:, :, ch][invalid_mask[:, :, ch]] = valid_mean
                    else:
                        corrected_array[invalid_mask] = valid_mean
            
            # Ensure proper value range (0-255 for image data)
            corrected_array = np.clip(corrected_array, 0.0, 255.0)
            
            # Ensure proper data type
            corrected_array = corrected_array.astype(np.float32)
            
            return corrected_array
        
        def optimize_memory_layout(array):
            """
            Optimize memory layout for better cache performance.
            Ensures C-contiguous layout and optimal memory alignment.
            """
            if array is None:
                return None
            
            # Ensure C-contiguous layout for better performance
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)
            
            # Optimize data alignment if possible
            if hasattr(array, 'flags') and not array.flags['ALIGNED']:
                # Create properly aligned array
                aligned_array = np.empty_like(array)
                aligned_array[:] = array
                array = aligned_array
            
            return array
        
        def perform_comprehensive_conversion():
            """
            Main conversion function that handles all array types and optimizations.
            Orchestrates the complete conversion process with error handling.
            """
            # Step 1: Detect array type
            array_info = detect_array_type(filtered_image)
            
            # Step 2: Initial validation
            is_valid, validation_message = validate_array_integrity(filtered_image)
            if not is_valid:
                warnings.warn(f"Array validation failed: {validation_message}")
                # Try to proceed with conversion anyway, apply corrections later
            
            # Step 3: Type-specific conversion
            converted_array = None
            
            if array_info['is_standard_numpy'] and not array_info['has_custom_cleanup']:
                # Already standard NumPy array
                converted_array = filtered_image.astype(np.float32)
                
            elif array_info['is_memmap'] or array_info['memory_mapped']:
                # Memory-mapped array conversion
                converted_array = convert_memmap_array(filtered_image)
                
            elif array_info['is_cuda']:
                # CUDA array conversion
                converted_array = convert_cuda_array(filtered_image)
                
            elif array_info['is_sparse']:
                # Sparse array conversion
                converted_array = convert_sparse_array(filtered_image)
                
            elif array_info['is_dask']:
                # Dask array conversion
                converted_array = convert_dask_array(filtered_image)
                
            else:
                # Generic conversion for unknown types
                try:
                    converted_array = np.array(filtered_image, dtype=np.float32, copy=True)
                except Exception as e:
                    warnings.warn(f"Generic array conversion failed: {e}")
                    return None
            
            # Step 4: Apply data corrections
            if converted_array is not None:
                converted_array = apply_data_corrections(converted_array)
            
            # Step 5: Optimize memory layout
            if converted_array is not None:
                converted_array = optimize_memory_layout(converted_array)
            
            # Step 6: Final validation
            if converted_array is not None:
                final_valid, final_message = validate_array_integrity(converted_array)
                if not final_valid:
                    warnings.warn(f"Final validation failed: {final_message}")
            
            return converted_array
        
        def cleanup_source_array():
            """
            Clean up the source array if it requires special cleanup.
            Handles memory-mapped files, CUDA memory, and custom cleanup routines.
            """
            try:
                # Handle memory-mapped cleanup
                if hasattr(filtered_image, '_temp_file_path'):
                    temp_file = filtered_image._temp_file_path
                    if hasattr(filtered_image, 'flush'):
                        filtered_image.flush()
                    # Note: Don't delete the file immediately in case it's still being used
                    # The cleanup will be handled by the garbage collector or explicit cleanup calls
                
                # Handle CUDA memory cleanup
                if hasattr(filtered_image, 'cuda') and hasattr(filtered_image, 'cpu'):
                    # PyTorch-style cleanup
                    if filtered_image.is_cuda:
                        torch.cuda.empty_cache()  # This would require torch import
                
                # Handle custom cleanup
                if hasattr(filtered_image, '_cleanup_required') and filtered_image._cleanup_required:
                    if hasattr(filtered_image, 'cleanup'):
                        filtered_image.cleanup()
                
            except Exception as e:
                warnings.warn(f"Source array cleanup failed: {e}")
        
        # Execute the comprehensive conversion process
        try:
            # Main conversion
            result_array = perform_comprehensive_conversion()
            
            # Cleanup source if needed (do this after successful conversion)
            if result_array is not None:
                cleanup_source_array()
            
            # Store conversion statistics for debugging
            convert_to_standard_array.last_conversion_stats = {
                'source_type': type(filtered_image).__name__,
                'source_shape': getattr(filtered_image, 'shape', 'unknown'),
                'source_dtype': getattr(filtered_image, 'dtype', 'unknown'),
                'result_shape': result_array.shape if result_array is not None else None,
                'result_dtype': result_array.dtype if result_array is not None else None,
                'conversion_successful': result_array is not None,
                'memory_usage_mb': result_array.nbytes / (1024**2) if result_array is not None else 0,
                'conversion_timestamp': time.time()
            }
            
            return result_array
            
        except Exception as e:
            # Ultimate fallback
            warnings.warn(f"Array conversion completely failed: {e}. Attempting emergency conversion.")
            
            try:
                # Emergency conversion - just try basic numpy conversion
                emergency_result = np.array(filtered_image, dtype=np.float32)
                emergency_result = np.clip(emergency_result, 0.0, 255.0)
                return emergency_result
                
            except Exception as emergency_error:
                warnings.warn(f"Emergency conversion also failed: {emergency_error}")
                return None

    # Memory-mapped processing for large images
    filtered_image = create_memory_mapped_array(image.shape)
    pad_size = kernel_size // 2

    # Streaming padding to avoid memory overflow
    padded_generator = create_streaming_padded_generator(image, pad_size)

    # Compressed weight representation for memory efficiency
    weight_representation = create_compressed_weight_representation(kernel_size, spatial_sigma)

    # Adaptive intensity approximation with machine learning
    intensity_approximator = create_ml_intensity_approximator(image, intensity_sigma)

    # Parallel chunk processing with load balancing
    chunk_size = calculate_optimal_chunk_size(h, w, kernel_size)
    num_workers = determine_optimal_workers()

    # Process chunks in parallel with sophisticated memory management
    chunk_results = []
    for chunk_id in range(0, h, chunk_size):
        chunk_end = min(chunk_id + chunk_size, h)

        # Get chunk data from streaming generator
        chunk_data = next(padded_generator)

        # Parallel processing with worker pool
        chunk_result = process_bigdata_bilateral_chunk(
            image[chunk_id:chunk_end], chunk_data,
            weight_representation, intensity_approximator, 
            pad_size, num_workers
        )

        # Stream result to memory-mapped output
        filtered_image[chunk_id:chunk_end] = chunk_result

        # Garbage collection for memory management
        cleanup_chunk_memory(chunk_data, chunk_result)

    # Final quality enhancement with edge-aware post-processing
    filtered_image = apply_distributed_post_processing(image, filtered_image, spatial_sigma, intensity_sigma)

    return convert_to_standard_array(filtered_image)




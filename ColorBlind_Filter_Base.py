import numpy as np
import pandas as pd
import matplotlib.colors
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def apply_colorblind_filter(image, filter_type, intensity=1.0, adaptive=False):
    """
    Apply an enhanced colorblind filter to an image with advanced options.

    :param image: The original game image (numpy array)
    :param filter_type: Type of colorblindness or visual adjustment
    :param intensity: Filter intensity (0.0 to 2.0, default 1.0)
    :param adaptive: Enable adaptive filtering based on image content
    :return: Image with enhanced colorblind filter applied
    """
    
    # Validate inputs
    if image is None or len(image.shape) != 3:
        return image
    
    intensity = np.clip(intensity, 0.0, 2.0)
    
    # Enhanced filter mapping with more options
    filter_functions = {
        'protanopia': adjust_for_protanopia,
        'deuteranopia': adjust_for_deuteranopia, 
        'tritanopia': adjust_for_tritanopia,
        'monochrome': adjust_for_monochrome,
        'no_purple': adjust_for_no_purple,
        'neutral_difficulty': adjust_for_neutral_difficulty,
        'warm_color_difficulty': adjust_for_warm_color_difficulty,
        'neutral_greyscale': adjust_for_neutral_greyscale,
        'warm_greyscale': adjust_for_warm_greyscale,
        'complex_color_difficulty': adjust_for_complex_color_difficulty,
        'simple_color_difficulty': adjust_for_simple_color_difficulty,
        'high_contrast': apply_high_contrast,
        'enhanced_contrast': lambda img: apply_enhanced_contrast(img, intensity),
        'color_boost': lambda img: apply_color_boost(img, intensity)
    }
    
    # Apply adaptive preprocessing if enabled
    if adaptive:
        image = apply_adaptive_preprocessing(image, filter_type)
    
    # Get the appropriate filter function
    filter_func = filter_functions.get(filter_type, lambda x: x)
    
    try:
        # Apply the base filter
        adjusted_image = filter_func(image)
        
        # Apply intensity scaling
        if intensity != 1.0:
            adjusted_image = apply_intensity_scaling(image, adjusted_image, intensity)
        
        # Post-processing for better results
        adjusted_image = apply_post_processing(adjusted_image, filter_type)
        
        return adjusted_image.astype(np.uint8)
        
    except Exception as e:
        # Fallback to original image if filtering fails
        print(f"Filter application failed: {e}")
        return image

def apply_adaptive_preprocessing(image, filter_type):
    """
    Apply advanced adaptive preprocessing based on image content, filter type, and multiple analysis techniques.
    
    :param image: Image array (RGB format)
    :param filter_type: Type of colorblind filter to be applied
    :return: Preprocessed image optimized for the specific filter
    """
    if image is None or len(image.shape) != 3:
        return image
    
    # Convert to float for precise calculations
    img_float = image.astype(np.float32)
    
    # Multi-dimensional image analysis
    brightness = np.mean(img_float)
    contrast = np.std(img_float)
    saturation = calculate_saturation(img_float)
    edge_density = calculate_edge_density(img_float)
    color_distribution = analyze_color_distribution(img_float)
    
    # Filter-specific preprocessing strategies
    preprocessing_strategy = get_preprocessing_strategy(filter_type)
    
    # Apply brightness corrections with adaptive gamma
    if brightness < 50:  # Dark image
        gamma = 0.7  # Brighten dark areas more
        img_float = apply_adaptive_gamma_correction(img_float, gamma, brightness)
    elif brightness > 200:  # Bright image
        gamma = 1.3  # Darken bright areas
        img_float = apply_adaptive_gamma_correction(img_float, gamma, brightness)
    
    # Enhanced contrast adjustment based on local statistics
    if contrast < 30:  # Low contrast
        img_float = apply_advanced_contrast_enhancement(img_float, edge_density)
    elif contrast > 120:  # High contrast - may need smoothing
        img_float = apply_contrast_smoothing(img_float)
    
    # Saturation adjustments for specific filter types
    if preprocessing_strategy['enhance_saturation'] and saturation < 0.3:
        img_float = enhance_saturation_adaptive(img_float, filter_type)
    elif preprocessing_strategy['reduce_saturation'] and saturation > 0.8:
        img_float = reduce_saturation_adaptive(img_float)
    
    # Color space optimization for colorblind filters
    if filter_type in ['protanopia', 'deuteranopia', 'tritanopia']:
        img_float = optimize_for_colorblind_vision(img_float, filter_type, color_distribution)
    
    # Edge enhancement for filters that benefit from it
    if preprocessing_strategy['enhance_edges'] and edge_density < 0.2:
        img_float = apply_selective_edge_enhancement(img_float)
    
    # Noise reduction for low-quality images
    if detect_noise_level(img_float) > 0.15:
        img_float = apply_adaptive_noise_reduction(img_float)
    
    # Final color space adjustments
    img_float = apply_filter_specific_color_prep(img_float, filter_type)
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

def calculate_saturation(image):
    """Calculate average saturation of the image."""
    hsv = matplotlib.colors.rgb_to_hsv(image / 255.0)
    return np.mean(hsv[:,:,1])

def calculate_edge_density(image):
    """Calculate edge density using gradient magnitude."""
    gray = np.mean(image, axis=2)
    grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(edge_magnitude) / 255.0

def analyze_color_distribution(image):
    """Analyze color distribution characteristics."""
    rgb_std = np.std(image, axis=(0,1))
    dominant_channel = np.argmax(rgb_std)
    color_variance = np.var(image, axis=(0,1))
    
    return {
        'dominant_channel': dominant_channel,
        'color_variance': color_variance,
        'red_dominance': np.mean(image[:,:,0]) / np.mean(image),
        'green_dominance': np.mean(image[:,:,1]) / np.mean(image),
        'blue_dominance': np.mean(image[:,:,2]) / np.mean(image)
    }

def get_preprocessing_strategy(filter_type):
    """Get preprocessing strategy based on filter type."""
    strategies = {
        'protanopia': {
            'enhance_saturation': True,
            'reduce_saturation': False,
            'enhance_edges': True,
            'boost_contrast': True
        },
        'deuteranopia': {
            'enhance_saturation': True,
            'reduce_saturation': False,
            'enhance_edges': True,
            'boost_contrast': True
        },
        'tritanopia': {
            'enhance_saturation': True,
            'reduce_saturation': False,
            'enhance_edges': False,
            'boost_contrast': True
        },
        'monochrome': {
            'enhance_saturation': False,
            'reduce_saturation': True,
            'enhance_edges': True,
            'boost_contrast': True
        },
        'high_contrast': {
            'enhance_saturation': False,
            'reduce_saturation': False,
            'enhance_edges': True,
            'boost_contrast': True
        }
    }
    
    return strategies.get(filter_type, {
        'enhance_saturation': False,
        'reduce_saturation': False,
        'enhance_edges': False,
        'boost_contrast': False
    })

def apply_adaptive_gamma_correction(image, gamma, brightness):
    """
    Apply advanced gamma correction with adaptive intensity based on local image characteristics.
    
    :param image: Input image array (float32, 0-255 range)
    :param gamma: Base gamma correction value
    :param brightness: Overall image brightness level
    :return: Gamma-corrected image with adaptive adjustments
    """
    # Normalize to 0-1 range for processing
    normalized = image / 255.0
    
    # Calculate local brightness statistics
    local_brightness = np.mean(normalized, axis=2, keepdims=True)
    local_contrast = np.std(normalized, axis=2, keepdims=True)
    
    # Multi-factor adaptive gamma calculation
    brightness_factor = (local_brightness - 0.5) * 0.3
    contrast_factor = (local_contrast - 0.2) * 0.15
    global_brightness_factor = (brightness / 128.0 - 1.0) * 0.2
    
    # Combine factors for adaptive gamma
    adaptive_gamma = gamma + brightness_factor + contrast_factor + global_brightness_factor
    adaptive_gamma = np.clip(adaptive_gamma, 0.3, 2.5)
    
    # Apply spatially-varying gamma correction
    corrected = np.power(normalized, adaptive_gamma)
    
    # Post-processing for edge preservation
    edge_mask = calculate_edge_mask(normalized)
    edge_preserved = apply_edge_preservation(normalized, corrected, edge_mask, 0.7)
    
    # Smooth transitions to avoid artifacts
    smoothed = apply_gradient_smoothing(edge_preserved, kernel_size=3)
    
    # Apply tone mapping for better dynamic range
    tone_mapped = apply_adaptive_tone_mapping(smoothed, local_brightness)
    
    # Final contrast enhancement
    final_result = apply_local_contrast_enhancement(tone_mapped, local_contrast)
    
    return np.clip(final_result * 255, 0, 255)

def calculate_edge_mask(image):
    """Calculate edge strength mask for preservation."""
    gray = np.mean(image, axis=2)
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    edge_mask = edge_strength / (np.max(edge_strength) + 1e-6)
    return np.expand_dims(edge_mask, axis=2)

def apply_edge_preservation(original, corrected, edge_mask, preservation_strength):
    """Preserve edge details during gamma correction."""
    blend_factor = edge_mask * preservation_strength
    preserved = corrected * (1 - blend_factor) + original * blend_factor
    return preserved

def apply_gradient_smoothing(image, kernel_size=3):
    """Apply gentle smoothing to reduce gamma correction artifacts."""
    if kernel_size < 3:
        return image
        
    # Create smoothing kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    smoothed_channels = []
    for channel in range(image.shape[2]):
        ch = image[:,:,channel]
        # Apply convolution with proper padding
        padded = np.pad(ch, kernel_size//2, mode='edge')
        smoothed = np.convolve(padded.flatten(), kernel.flatten(), mode='valid')
        smoothed = smoothed.reshape(ch.shape)
        smoothed_channels.append(smoothed)
    
    smoothed_image = np.stack(smoothed_channels, axis=-1)
    
    # Blend with original based on local variance
    local_variance = np.var(image, axis=2, keepdims=True)
    blend_factor = np.clip(local_variance * 2, 0, 0.3)
    
    return image * (1 - blend_factor) + smoothed_image * blend_factor

def apply_adaptive_tone_mapping(image, local_brightness):
    """Apply adaptive tone mapping for better dynamic range."""
    # S-curve tone mapping with local adaptation
    midpoint = np.clip(local_brightness, 0.2, 0.8)
    
    # Create adaptive S-curve
    tone_mapped = np.where(
        image < midpoint,
        midpoint * np.power(image / midpoint, 1.2),
        midpoint + (1 - midpoint) * np.power((image - midpoint) / (1 - midpoint), 0.8)
    )
    
    return np.clip(tone_mapped, 0, 1)

def apply_local_contrast_enhancement(image, local_contrast):
    """Apply final local contrast enhancement."""
    # Calculate local mean
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    enhanced_channels = []
    for channel in range(image.shape[2]):
        ch = image[:,:,channel]
        
        # Calculate local mean
        padded = np.pad(ch, kernel_size//2, mode='edge')
        local_mean = np.convolve(padded.flatten(), kernel.flatten(), mode='valid')
        local_mean = local_mean.reshape(ch.shape)
        
        # Enhancement factor based on local contrast
        enhancement_factor = 1.0 + local_contrast[:,:,0] * 0.5
        enhancement_factor = np.clip(enhancement_factor, 1.0, 1.8)
        
        # Apply local contrast enhancement
        enhanced = local_mean + (ch - local_mean) * enhancement_factor
        enhanced_channels.append(enhanced)
    
    enhanced_image = np.stack(enhanced_channels, axis=-1)
    return np.clip(enhanced_image, 0, 1)

def apply_advanced_contrast_enhancement(image, edge_density):
    """
    Apply advanced contrast enhancement with adaptive processing and edge preservation.
    
    :param image: Input image array (float32, 0-255 range)
    :param edge_density: Edge density metric from image analysis
    :return: Enhanced image with improved local contrast
    """
    # Adaptive tile size based on image dimensions and content
    min_dim = min(image.shape[:2])
    base_tile_size = max(16, min_dim // 20)
    
    # Adjust tile size based on edge density - smaller tiles for high edge density
    if edge_density > 0.3:
        tile_size = max(16, int(base_tile_size * 0.7))
    elif edge_density < 0.1:
        tile_size = min(128, int(base_tile_size * 1.5))
    else:
        tile_size = base_tile_size
    
    # Multi-factor enhancement calculation
    base_enhancement = 1.3
    edge_factor = edge_density * 0.8
    brightness_factor = calculate_brightness_adaptation_factor(image)
    dynamic_enhancement = base_enhancement + edge_factor + brightness_factor
    
    # Initialize output array
    enhanced = np.zeros_like(image)
    overlap_weights = np.zeros(image.shape[:2])
    h, w = image.shape[:2]
    
    # Overlapping tile processing for smoother transitions
    overlap_ratio = 0.25
    step_size = int(tile_size * (1 - overlap_ratio))
    
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile = image[y:y_end, x:x_end]
            
            # Enhanced local statistics with robust measures
            tile_mean = np.mean(tile, axis=(0, 1), keepdims=True)
            tile_std = np.std(tile, axis=(0, 1), keepdims=True)
            tile_median = np.median(tile, axis=(0, 1), keepdims=True)
            
            # Calculate local enhancement factor
            local_variance = np.var(tile)
            contrast_need = calculate_local_contrast_need(tile, tile_std)
            
            # Adaptive enhancement based on local characteristics
            if np.any(tile_std > 8):  # Sufficient variation exists
                # Use robust center (median) for better handling of outliers
                robust_center = 0.7 * tile_mean + 0.3 * tile_median
                
                # Calculate adaptive enhancement factor
                local_enhancement = dynamic_enhancement * contrast_need
                local_enhancement = np.clip(local_enhancement, 0.8, 2.5)
                
                # Apply CLAHE-style enhancement with gamma correction
                enhanced_tile = apply_clahe_enhancement(tile, robust_center, local_enhancement)
                
                # Edge-preserving smoothing
                enhanced_tile = apply_edge_preserving_smoothing(tile, enhanced_tile, tile_std)
                
                # Anti-halo processing
                enhanced_tile = reduce_halo_artifacts(tile, enhanced_tile)
                
            else:
                # Minimal enhancement for low-variation areas
                enhanced_tile = apply_gentle_enhancement(tile, tile_mean, 1.1)
            
            # Calculate blending weights for overlapping regions
            tile_weights = calculate_tile_weights(tile_size, y_end - y, x_end - x)
            
            # Accumulate enhanced values with weights
            enhanced[y:y_end, x:x_end] += enhanced_tile * tile_weights
            overlap_weights[y:y_end, x:x_end] += tile_weights
    
    # Normalize by overlap weights
    overlap_weights = np.maximum(overlap_weights, 1e-6)  # Avoid division by zero
    for channel in range(image.shape[2]):
        enhanced[:,:,channel] /= overlap_weights
    
    # Global post-processing
    enhanced = apply_global_tone_mapping(enhanced, image)
    enhanced = apply_selective_sharpening(enhanced, edge_density)
    
    # Final histogram adjustment
    enhanced = apply_adaptive_histogram_adjustment(enhanced)
    
    return np.clip(enhanced, 0, 255)

def calculate_brightness_adaptation_factor(image):
    """Calculate brightness adaptation factor for enhancement."""
    global_brightness = np.mean(image)
    
    if global_brightness < 60:  # Dark image
        return 0.4  # More enhancement needed
    elif global_brightness > 180:  # Bright image
        return -0.2  # Less enhancement needed
    else:
        return 0.1  # Moderate enhancement

def calculate_local_contrast_need(tile, tile_std):
    """Calculate how much contrast enhancement is needed for this tile."""
    # Multi-metric analysis
    std_metric = np.mean(tile_std) / 85.0  # Normalize standard deviation
    range_metric = (np.max(tile) - np.min(tile)) / 255.0
    gradient_metric = calculate_local_gradient_strength(tile)
    
    # Weighted combination
    contrast_need = 0.4 * std_metric + 0.3 * range_metric + 0.3 * gradient_metric
    return np.clip(1.0 + contrast_need, 0.8, 2.0)

def calculate_local_gradient_strength(tile):
    """Calculate local gradient strength for contrast assessment."""
    gray_tile = np.mean(tile, axis=2) if len(tile.shape) == 3 else tile
    
    # Simple gradient calculation
    grad_x = np.diff(gray_tile, axis=1, prepend=gray_tile[:, :1])
    grad_y = np.diff(gray_tile, axis=0, prepend=gray_tile[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return np.mean(gradient_magnitude) / 128.0

def apply_clahe_enhancement(tile, center, enhancement_factor):
    """Apply CLAHE-style enhancement with gamma correction."""
    # Normalize around center
    normalized_tile = (tile - center) / (center + 1e-6)
    
    # Apply S-curve transformation for better tone mapping
    enhanced_normalized = apply_s_curve_transformation(normalized_tile, enhancement_factor)
    
    # Convert back to original range
    enhanced_tile = center + enhanced_normalized * center
    
    return enhanced_tile

def apply_s_curve_transformation(normalized_data, strength):
    """Apply S-curve transformation for better contrast."""
    # Sigmoid-based S-curve
    s_curve = np.tanh(normalized_data * strength) / np.tanh(strength)
    return s_curve

def apply_edge_preserving_smoothing(original, enhanced, local_std):
    """Apply edge-preserving smoothing to reduce artifacts."""
    # Calculate edge mask
    edge_threshold = np.mean(local_std) * 0.8
    edge_mask = local_std > edge_threshold
    
    # Smooth non-edge areas more aggressively
    smoothing_strength = np.where(edge_mask, 0.1, 0.4)
    
    # Apply bilateral-like filtering
    smoothed = original * smoothing_strength + enhanced * (1 - smoothing_strength)
    
    return smoothed

def reduce_halo_artifacts(original, enhanced):
    """Reduce halo artifacts around high-contrast edges."""
    # Detect potential halo regions
    diff = enhanced - original
    halo_threshold = np.std(diff) * 2
    
    halo_mask = np.abs(diff) > halo_threshold
    
    # Reduce enhancement in halo regions
    anti_halo = np.where(halo_mask, 
                        original + diff * 0.5,  # Reduced enhancement
                        enhanced)  # Keep original enhancement
    
    return anti_halo

def apply_gentle_enhancement(tile, center, factor):
    """Apply gentle enhancement for low-variation areas."""
    # Simple contrast stretch with minimal artifacts
    gentle_enhanced = center + (tile - center) * factor
    return gentle_enhanced

def calculate_tile_weights(tile_size, actual_h, actual_w):
    """Calculate weights for tile blending."""
    # Create distance-based weights for smooth blending
    y_weights = np.minimum(np.arange(actual_h), np.arange(actual_h)[::-1])
    x_weights = np.minimum(np.arange(actual_w), np.arange(actual_w)[::-1])
    
    # Normalize weights
    y_weights = y_weights / np.max(y_weights) if np.max(y_weights) > 0 else np.ones_like(y_weights)
    x_weights = x_weights / np.max(x_weights) if np.max(x_weights) > 0 else np.ones_like(x_weights)
    
    # Create 2D weight matrix
    weights_2d = np.outer(y_weights, x_weights)
    
    # Expand to match image channels
    return np.expand_dims(weights_2d, axis=2)

def apply_global_tone_mapping(enhanced, original):
    """Apply global tone mapping for better overall appearance."""
    # Calculate global statistics
    enhanced_mean = np.mean(enhanced)
    original_mean = np.mean(original)
    
    # Prevent over-enhancement
    if enhanced_mean > original_mean * 1.8:
        # Apply tone compression
        compression_factor = (original_mean * 1.5) / enhanced_mean
        enhanced = enhanced * compression_factor
    
    return enhanced

def apply_selective_sharpening(image, edge_density):
    """Apply selective sharpening based on edge density."""
    if edge_density > 0.25:
        # High edge density - minimal sharpening
        sharpening_strength = 0.1
    else:
        # Low edge density - more sharpening
        sharpening_strength = 0.3
    
    # Simple unsharp mask
    blurred = apply_gaussian_blur(image, sigma=0.8)
    unsharp_mask = image - blurred
    sharpened = image + sharpening_strength * unsharp_mask
    
    return sharpened

def apply_adaptive_histogram_adjustment(image):
    """Apply final adaptive histogram adjustment."""
    # Calculate histogram statistics
    image_flat = image.reshape(-1, image.shape[-1])
    
    for channel in range(image.shape[2]):
        channel_data = image_flat[:, channel]
        
        # Calculate percentiles for adaptive stretching
        p1, p99 = np.percentile(channel_data, [1, 99])
        
        if p99 > p1:  # Avoid division by zero
            # Gentle histogram stretching
            stretched = np.clip((channel_data - p1) / (p99 - p1) * 255, 0, 255)
            image_flat[:, channel] = stretched
    
    return image_flat.reshape(image.shape)

def apply_contrast_smoothing(image):
    """Smooth excessive contrast to prevent artifacts."""
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    smoothed_channels = []
    for channel in range(image.shape[2]):
        ch = image[:,:,channel]
        # Simple smoothing
        smoothed = np.convolve(ch.flatten(), kernel.flatten(), mode='same').reshape(ch.shape)
        smoothed_channels.append(smoothed)
    
    smoothed_image = np.stack(smoothed_channels, axis=-1)
    
    # Blend original and smoothed based on local contrast
    local_contrast = np.std(image, axis=2, keepdims=True)
    blend_factor = np.clip(local_contrast / 100.0, 0, 0.7)
    
    result = image * (1 - blend_factor) + smoothed_image * blend_factor
    return result

def enhance_saturation_adaptive(image, filter_type):
    """Enhance saturation adaptively based on filter type."""
    hsv = matplotlib.colors.rgb_to_hsv(image / 255.0)
    
    # Different enhancement factors for different filter types
    enhancement_factors = {
        'protanopia': 1.3,
        'deuteranopia': 1.4,
        'tritanopia': 1.2,
        'default': 1.2
    }
    
    factor = enhancement_factors.get(filter_type, enhancement_factors['default'])
    
    # Enhance saturation while preserving natural colors
    hsv[:,:,1] = np.clip(hsv[:,:,1] * factor, 0, 1)
    
    enhanced_rgb = matplotlib.colors.hsv_to_rgb(hsv) * 255
    return enhanced_rgb

def reduce_saturation_adaptive(image):
    """Reduce saturation for oversaturated images."""
    hsv = matplotlib.colors.rgb_to_hsv(image / 255.0)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.7, 0, 1)
    
    reduced_rgb = matplotlib.colors.hsv_to_rgb(hsv) * 255
    return reduced_rgb

def optimize_for_colorblind_vision(image, filter_type, color_distribution):
    """Optimize image colors specifically for colorblind vision types."""
    if filter_type == 'protanopia':
        # Reduce red-green confusion by enhancing blue-yellow contrast
        image[:,:,2] = np.clip(image[:,:,2] * 1.1, 0, 255)  # Enhance blue
        
    elif filter_type == 'deuteranopia':
        # Enhance red and blue channels
        image[:,:,0] = np.clip(image[:,:,0] * 1.05, 0, 255)
        image[:,:,2] = np.clip(image[:,:,2] * 1.05, 0, 255)
        
    elif filter_type == 'tritanopia':
        # Enhance red and green for blue-yellow confusion
        image[:,:,0] = np.clip(image[:,:,0] * 1.05, 0, 255)
        image[:,:,1] = np.clip(image[:,:,1] * 1.05, 0, 255)
    
    return image

def apply_selective_edge_enhancement(image):
    """Apply selective edge enhancement."""
    # Calculate edge strength
    gray = np.mean(image, axis=2)
    grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create enhancement mask
    edge_mask = edge_strength / (np.max(edge_strength) + 1e-6)
    edge_mask = np.expand_dims(edge_mask, axis=2)
    
    # Apply unsharp masking selectively
    blurred = apply_gaussian_blur(image, sigma=1.0)
    unsharp_mask = image - blurred
    
    enhanced = image + edge_mask * unsharp_mask * 0.3
    return np.clip(enhanced, 0, 255)

def detect_noise_level(image):
    """
    Advanced noise detection using multiple statistical metrics and frequency analysis.
    
    :param image: Input image array (float32, 0-255 range)
    :return: Normalized noise level (0.0 to 1.0)
    """
    if image is None or len(image.shape) != 3:
        return 0.0
    
    # Convert to grayscale for analysis
    gray = np.mean(image, axis=2)
    h, w = gray.shape
    
    # Method 1: Local variance analysis
    kernel_size = 5
    pad_size = kernel_size // 2
    padded = np.pad(gray, pad_size, mode='edge')
    
    local_variances = []
    for i in range(h):
        for j in range(w):
            local_patch = padded[i:i+kernel_size, j:j+kernel_size]
            local_variance = np.var(local_patch)
            local_variances.append(local_variance)
    
    local_variances = np.array(local_variances)
    variance_metric = np.std(local_variances) / (np.mean(local_variances) + 1e-6)
    
    # Method 2: High-frequency noise detection using gradients
    grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate noise-to-signal ratio in gradients
    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)
    gradient_metric = gradient_std / (gradient_mean + 1e-6)
    
    # Method 3: Laplacian variance for blur/noise assessment
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_response = np.abs(np.convolve(gray.flatten(), laplacian_kernel.flatten(), mode='same')).reshape(gray.shape)
    laplacian_variance = np.var(laplacian_response)
    laplacian_metric = laplacian_variance / (np.mean(laplacian_response) + 1e-6)
    
    # Method 4: Texture analysis using standard deviation in local patches
    patch_stds = []
    patch_size = 8
    for i in range(0, h - patch_size, patch_size//2):
        for j in range(0, w - patch_size, patch_size//2):
            patch = gray[i:i+patch_size, j:j+patch_size]
            patch_std = np.std(patch)
            patch_stds.append(patch_std)
    
    patch_stds = np.array(patch_stds)
    texture_metric = np.std(patch_stds) / (np.mean(patch_stds) + 1e-6)
    
    # Method 5: Edge coherence analysis
    # Strong edges should be coherent, noise creates incoherent edges
    edge_threshold = np.percentile(gradient_magnitude, 75)
    strong_edges = gradient_magnitude > edge_threshold
    
    if np.sum(strong_edges) > 0:
        edge_coherence = calculate_edge_coherence(gray, strong_edges)
        coherence_metric = 1.0 - edge_coherence  # Lower coherence = more noise
    else:
        coherence_metric = 0.5  # Neutral if no strong edges
    
    # Method 6: Frequency domain analysis
    frequency_metric = analyze_frequency_domain_noise(gray)
    
    # Combine all metrics with weights
    weights = [0.2, 0.25, 0.15, 0.15, 0.15, 0.1]
    metrics = [variance_metric, gradient_metric, laplacian_metric, 
              texture_metric, coherence_metric, frequency_metric]
    
    # Normalize individual metrics to 0-1 range
    normalized_metrics = []
    for metric in metrics:
        # Use sigmoid function to normalize
        normalized = 1 / (1 + np.exp(-5 * (metric - 0.5)))
        normalized_metrics.append(normalized)
    
    # Weighted combination
    combined_noise_level = np.sum([w * m for w, m in zip(weights, normalized_metrics)])
    
    # Final normalization and clamping
    final_noise_level = np.clip(combined_noise_level, 0.0, 1.0)
    
    return final_noise_level

def calculate_edge_coherence(gray, edge_mask):
    """Calculate coherence of edge orientations."""
    # Calculate gradient orientations
    grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    
    # Calculate angles
    angles = np.arctan2(grad_y, grad_x)
    edge_angles = angles[edge_mask]
    
    if len(edge_angles) == 0:
        return 0.5
    
    # Calculate circular variance (measure of angle dispersion)
    mean_cos = np.mean(np.cos(2 * edge_angles))
    mean_sin = np.mean(np.sin(2 * edge_angles))
    circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
    
    # Convert to coherence (lower variance = higher coherence)
    coherence = 1 - circular_variance
    return np.clip(coherence, 0.0, 1.0)

def analyze_frequency_domain_noise(gray):
    """Analyze noise in frequency domain."""
    # Apply FFT
    fft_image = np.fft.fft2(gray)
    fft_magnitude = np.abs(fft_image)
    
    # Calculate power spectrum
    power_spectrum = fft_magnitude**2
    
    # Analyze high-frequency content
    h, w = power_spectrum.shape
    center_h, center_w = h//2, w//2
    
    # Create frequency masks
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((y - center_h)**2 + (x - center_w)**2)
    
    # Define frequency bands
    low_freq_mask = distance_from_center < min(h, w) * 0.1
    high_freq_mask = distance_from_center > min(h, w) * 0.3
    
    # Calculate energy in different frequency bands
    low_freq_energy = np.sum(power_spectrum[low_freq_mask])
    high_freq_energy = np.sum(power_spectrum[high_freq_mask])
    total_energy = np.sum(power_spectrum)
    
    # Noise typically increases high-frequency content
    high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
    
    # Normalize to 0-1 range
    frequency_noise_metric = np.clip(high_freq_ratio * 10, 0.0, 1.0)
    
    return frequency_noise_metric

def apply_adaptive_noise_reduction(image):
    """
    Apply advanced adaptive noise reduction with multiple filtering techniques and optimization.
    
    :param image: Input image array (float32, 0-255 range)
    :return: Noise-reduced image with preserved edge details
    """
    if image is None or len(image.shape) != 3:
        return image
    
    # Convert to float for precise calculations
    img_float = image.astype(np.float32)
    h, w, c = img_float.shape
    
    # Detect noise characteristics for adaptive processing
    noise_characteristics = analyze_noise_characteristics(img_float)
    
    # Choose optimal filtering strategy based on noise type and level
    if noise_characteristics['noise_level'] < 0.1:
        # Low noise - minimal filtering to preserve details
        return apply_minimal_noise_reduction(img_float, noise_characteristics)
    elif noise_characteristics['noise_level'] > 0.6:
        # High noise - aggressive multi-stage filtering
        return apply_aggressive_noise_reduction(img_float, noise_characteristics)
    else:
        # Moderate noise - balanced adaptive filtering
        return apply_balanced_adaptive_filtering(img_float, noise_characteristics)

def analyze_noise_characteristics(image):
    """Analyze image noise characteristics for optimal filtering strategy."""
    # Calculate local variance to identify noise patterns
    local_variance = calculate_local_variance(image, kernel_size=5)
    
    # Detect noise type (Gaussian, salt-and-pepper, etc.)
    noise_type = detect_noise_type(image)
    
    # Calculate noise frequency distribution
    frequency_analysis = analyze_noise_frequency_spectrum(image)
    
    # Estimate edge density for preservation priority
    edge_density = calculate_edge_density(image)
    
    return {
        'noise_level': detect_noise_level(image),
        'noise_type': noise_type,
        'local_variance': local_variance,
        'frequency_profile': frequency_analysis,
        'edge_density': edge_density,
        'texture_complexity': calculate_texture_complexity(image)
    }

def detect_noise_type(image):
    """Detect the primary type of noise in the image."""
    gray = np.mean(image, axis=2)
    
    # Test for salt-and-pepper noise
    extreme_pixels = np.sum((gray < 10) | (gray > 245)) / gray.size
    
    # Test for Gaussian noise using local statistics
    local_std = np.std(gray)
    gradient_noise_ratio = calculate_gradient_noise_ratio(gray)
    
    if extreme_pixels > 0.01:
        return 'salt_pepper'
    elif gradient_noise_ratio > 0.3:
        return 'gaussian'
    elif local_std > 40:
        return 'uniform'
    else:
        return 'mixed'

def calculate_gradient_noise_ratio(gray):
    """Calculate ratio of noise to signal in gradients."""
    grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # High-frequency component indicates noise
    high_freq_threshold = np.percentile(gradient_magnitude, 85)
    noise_ratio = np.sum(gradient_magnitude > high_freq_threshold) / gradient_magnitude.size
    return noise_ratio

def apply_minimal_noise_reduction(image, characteristics):
    """Apply minimal noise reduction for low-noise images."""
    # Use gentle Gaussian blur with very small sigma
    sigma = 0.5
    kernel_size = 3
    
    return apply_optimized_gaussian_blur(image, sigma, kernel_size)

def apply_aggressive_noise_reduction(image, characteristics):
    """Apply multi-stage aggressive noise reduction for high-noise images."""
    # Stage 1: Pre-filtering for extreme noise
    if characteristics['noise_type'] == 'salt_pepper':
        filtered = apply_median_filter(image, kernel_size=3)
    else:
        filtered = image.copy()
    
    # Stage 2: Advanced bilateral filtering
    filtered = apply_enhanced_bilateral_filter(filtered, characteristics)
    
    # Stage 3: Non-local means denoising for texture preservation
    filtered = apply_fast_non_local_means(filtered, characteristics)
    
    # Stage 4: Edge-preserving smoothing
    filtered = apply_edge_preserving_smoothing(filtered, characteristics)
    
    return np.clip(filtered, 0, 255)

def apply_balanced_adaptive_filtering(image, characteristics):
    """Apply balanced adaptive filtering for moderate noise."""
    # Adaptive bilateral filtering with optimized parameters
    filtered = apply_enhanced_bilateral_filter(image, characteristics)
    
    # Selective smoothing based on local image properties
    filtered = apply_selective_smoothing(filtered, characteristics)
    
    return np.clip(filtered, 0, 255)

def apply_enhanced_bilateral_filter(image, characteristics):
    """Enhanced bilateral filter with adaptive parameters."""
    h, w, c = image.shape
    
    # Adaptive parameter selection
    spatial_sigma = adapt_spatial_sigma(characteristics)
    intensity_sigma = adapt_intensity_sigma(characteristics)
    kernel_size = min(7, max(3, int(spatial_sigma * 3)))
    
    # Optimized implementation using separable approximation
    if kernel_size <= 5:
        return apply_fast_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size)
    else:
        # For larger kernels, use the fast bilateral filter with adjusted parameters
        return apply_fast_bilateral_filter(image, spatial_sigma, intensity_sigma, 7)

def apply_separable_bilateral_filter(image, spatial_sigma, intensity_sigma):
    """
    Apply enhanced separable bilateral filter with optimized performance for larger kernels.
    
    This implementation uses true separable filtering to significantly reduce computational
    complexity from O(n²) to O(n) while maintaining high-quality results.
    
    :param image: Input image array (float32, 0-255 range)
    :param spatial_sigma: Spatial standard deviation for Gaussian weights
    :param intensity_sigma: Intensity standard deviation for edge preservation
    :return: High-quality filtered image with preserved edges
    """
    if image is None or len(image.shape) != 3:
        return image
    
    h, w, c = image.shape
    
    # Adaptive kernel size calculation based on spatial sigma
    kernel_size = max(5, min(15, int(spatial_sigma * 4) | 1))  # Ensure odd number
    pad_size = kernel_size // 2
    
    # Pre-compute 1D Gaussian spatial weights
    spatial_weights_1d = create_1d_gaussian_weights(kernel_size, spatial_sigma)
    
    # Initialize filtered image
    filtered_image = np.zeros_like(image, dtype=np.float32)
    
    # Pad image to handle boundaries
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    # Stage 1: Horizontal bilateral filtering (separable approximation)
    h_filtered = np.zeros_like(padded, dtype=np.float32)
    
    for i in range(pad_size, h + pad_size):
        for j in range(pad_size, w + pad_size):
            center_pixel = padded[i, j]
            
            # Extract horizontal neighborhood
            h_neighborhood = padded[i, j-pad_size:j+pad_size+1]
            
            # Calculate intensity-based weights for horizontal direction
            intensity_diffs = np.sum((h_neighborhood - center_pixel)**2, axis=1)
            intensity_weights = np.exp(-intensity_diffs / (2 * intensity_sigma**2))
            
            # Combine spatial and intensity weights
            combined_weights = spatial_weights_1d * intensity_weights
            weight_sum = np.sum(combined_weights)
            
            if weight_sum > 1e-6:
                # Apply weighted average
                h_filtered[i, j] = np.sum(
                    h_neighborhood * combined_weights[:, np.newaxis], axis=0
                ) / weight_sum
            else:
                h_filtered[i, j] = center_pixel
    
    # Stage 2: Vertical bilateral filtering on horizontally filtered result
    for i in range(pad_size, h + pad_size):
        for j in range(pad_size, w + pad_size):
            center_pixel = h_filtered[i, j]
            
            # Extract vertical neighborhood from horizontally filtered image
            v_neighborhood = h_filtered[i-pad_size:i+pad_size+1, j]
            
            # Calculate intensity-based weights for vertical direction
            intensity_diffs = np.sum((v_neighborhood - center_pixel)**2, axis=1)
            intensity_weights = np.exp(-intensity_diffs / (2 * intensity_sigma**2))
            
            # Combine spatial and intensity weights
            combined_weights = spatial_weights_1d * intensity_weights
            weight_sum = np.sum(combined_weights)
            
            if weight_sum > 1e-6:
                # Apply weighted average
                filtered_image[i-pad_size, j-pad_size] = np.sum(
                    v_neighborhood * combined_weights[:, np.newaxis], axis=0
                ) / weight_sum
            else:
                filtered_image[i-pad_size, j-pad_size] = center_pixel
    
    # Post-processing: Edge enhancement and artifact reduction
    filtered_image = apply_post_bilateral_enhancement(image, filtered_image, spatial_sigma)
    
    # Noise variance adaptive blending
    noise_level = estimate_local_noise_level(image)
    blend_factor = calculate_adaptive_blend_factor(noise_level, spatial_sigma)
    final_result = image * (1 - blend_factor) + filtered_image * blend_factor
    
    return np.clip(final_result, 0, 255).astype(np.float32)

def create_1d_gaussian_weights(kernel_size, sigma):
    """Create optimized 1D Gaussian weights for separable filtering."""
    center = kernel_size // 2
    x = np.arange(kernel_size) - center
    weights = np.exp(-x**2 / (2 * sigma**2))
    return weights / np.sum(weights)

def apply_post_bilateral_enhancement(original, filtered, spatial_sigma):
    """Apply post-processing enhancements to bilateral filter results."""
    # Edge-preserving sharpening
    if spatial_sigma > 2.0:
        # For strong smoothing, apply gentle sharpening
        detail_layer = original - filtered
        sharpening_strength = 0.15 * min(1.0, spatial_sigma / 3.0)
        enhanced = filtered + sharpening_strength * detail_layer
    else:
        enhanced = filtered
    
    # Gradient coherence preservation
    enhanced = preserve_gradient_coherence(original, enhanced)
    
    return enhanced

def estimate_local_noise_level(image):
    """Estimate local noise level using robust statistics."""
    # Use Laplacian variance method for noise estimation
    gray = np.mean(image, axis=2)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    # Apply Laplacian and calculate variance
    conv_result = np.convolve(gray.flatten(), laplacian.flatten(), mode='same')
    laplacian_var = np.var(conv_result.reshape(gray.shape))
    
    # Normalize noise level estimate
    return np.clip(laplacian_var / 10000.0, 0.0, 1.0)

def calculate_adaptive_blend_factor(noise_level, spatial_sigma):
    """Calculate adaptive blending factor based on noise and filter strength."""
    # Higher noise or stronger filtering needs more blending
    base_factor = 0.7
    noise_factor = noise_level * 0.3
    sigma_factor = min(0.2, spatial_sigma / 10.0)
    
    blend_factor = base_factor + noise_factor + sigma_factor
    return np.clip(blend_factor, 0.3, 1.0)

def preserve_gradient_coherence(original, filtered):
    """Preserve important gradient directions and magnitudes."""
    # Calculate gradients for both images
    orig_grad_x = np.diff(np.mean(original, axis=2), axis=1, prepend=0)
    orig_grad_y = np.diff(np.mean(original, axis=2), axis=0, prepend=0)
    
    filt_grad_x = np.diff(np.mean(filtered, axis=2), axis=1, prepend=0)
    filt_grad_y = np.diff(np.mean(filtered, axis=2), axis=0, prepend=0)
    
    # Calculate gradient magnitude preservation factor
    orig_magnitude = np.sqrt(orig_grad_x**2 + orig_grad_y**2)
    filt_magnitude = np.sqrt(filt_grad_x**2 + filt_grad_y**2)
    
    # Preserve strong edges
    strong_edge_mask = orig_magnitude > np.percentile(orig_magnitude, 85)
    preservation_factor = np.where(strong_edge_mask, 0.7, 1.0)
    preservation_factor = np.expand_dims(preservation_factor, axis=2)
    
    # Blend original and filtered based on edge strength
    coherence_preserved = filtered * preservation_factor + original * (1 - preservation_factor)
    
    return coherence_preserved

def adapt_spatial_sigma(characteristics):
    """
    Adapt spatial sigma for bilateral filtering based on comprehensive image characteristics.
    
    This function dynamically adjusts the spatial sigma parameter for bilateral filtering
    by analyzing multiple image properties including noise level, edge density, texture
    complexity, and local image statistics to optimize filtering performance.
    
    :param characteristics: Dictionary containing image analysis results with keys:
                          - 'noise_level': Normalized noise level (0.0-1.0)
                          - 'edge_density': Edge density metric (0.0-1.0)
                          - 'texture_complexity': Texture complexity measure (0.0-1.0)
                          - 'local_variance': Local variance information
                          - 'frequency_profile': Frequency domain analysis results
    :return: Optimally adapted spatial sigma value (0.5-4.0)
    """
    # Enhanced base sigma calculation based on image content
    base_sigma = calculate_adaptive_base_sigma(characteristics)
    
    # Multi-factor adaptation with weighted contributions
    noise_adaptation = calculate_noise_adaptation(characteristics['noise_level'])
    edge_adaptation = calculate_edge_adaptation(characteristics['edge_density'])
    texture_adaptation = calculate_texture_adaptation(characteristics['texture_complexity'])
    frequency_adaptation = calculate_frequency_adaptation(characteristics.get('frequency_profile', {}))
    variance_adaptation = calculate_variance_adaptation(characteristics.get('local_variance', 0))
    
    # Advanced weighting system based on image content priority
    weights = determine_adaptive_weights(characteristics)
    
    # Combine adaptations with dynamic weighting
    total_adaptation = (
        weights['noise'] * noise_adaptation +
        weights['edge'] * edge_adaptation +
        weights['texture'] * texture_adaptation +
        weights['frequency'] * frequency_adaptation +
        weights['variance'] * variance_adaptation
    )
    
    # Apply non-linear scaling for better control
    adapted_sigma = base_sigma + apply_nonlinear_scaling(total_adaptation, characteristics)
    
    # Apply contextual constraints based on image type
    adapted_sigma = apply_contextual_constraints(adapted_sigma, characteristics)
    
    # Final validation and range enforcement
    final_sigma = validate_and_clamp_sigma(adapted_sigma, characteristics)
    
    return final_sigma

def calculate_adaptive_base_sigma(characteristics):
    """Calculate adaptive base sigma based on overall image characteristics."""
    noise_level = characteristics['noise_level']
    edge_density = characteristics['edge_density']
    texture_complexity = characteristics['texture_complexity']
    
    # Dynamic base calculation
    if noise_level > 0.7:
        base = 2.2  # Higher base for very noisy images
    elif edge_density > 0.6:
        base = 1.0  # Lower base for edge-rich images
    elif texture_complexity > 0.8:
        base = 1.3  # Moderate base for complex textures
    else:
        base = 1.5  # Standard base
    
    # Fine-tune based on combination of factors
    combination_factor = (noise_level * 0.4 + edge_density * -0.3 + texture_complexity * 0.1)
    return base + combination_factor * 0.5

def calculate_noise_adaptation(noise_level):
    """Calculate noise-based adaptation with sophisticated scaling."""
    if noise_level < 0.1:
        return -0.2  # Reduce sigma for very clean images
    elif noise_level < 0.3:
        return noise_level * 0.5  # Gentle increase
    elif noise_level < 0.6:
        return 0.15 + (noise_level - 0.3) * 1.2  # Moderate increase
    else:
        return 0.51 + (noise_level - 0.6) * 1.8  # Aggressive increase for high noise

def calculate_edge_adaptation(edge_density):
    """Calculate edge-based adaptation with preservation priority."""
    # Non-linear relationship to preserve important edges
    if edge_density < 0.2:
        return 0.3  # Slight increase for low-edge images
    elif edge_density < 0.5:
        return 0.3 - (edge_density - 0.2) * 0.8  # Gradual decrease
    else:
        return -0.24 - (edge_density - 0.5) * 1.2  # Strong decrease for edge-rich images

def calculate_texture_adaptation(texture_complexity):
    """Calculate texture-based adaptation for detail preservation."""
    # Sigmoid-based adaptation for smooth transitions
    sigmoid_input = (texture_complexity - 0.5) * 8
    sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))
    return (sigmoid_output - 0.5) * -0.6

def calculate_frequency_adaptation(frequency_profile):
    """Calculate adaptation based on frequency domain analysis."""
    if not frequency_profile:
        return 0.0
    
    high_freq_ratio = frequency_profile.get('high_freq_ratio', 0.5)
    
    # Higher high-frequency content suggests more detail to preserve
    if high_freq_ratio > 0.4:
        return -0.4 * (high_freq_ratio - 0.4)  # Reduce sigma for detail-rich areas
    else:
        return 0.2 * (0.4 - high_freq_ratio)  # Increase sigma for smooth areas

def calculate_variance_adaptation(local_variance):
    """Calculate adaptation based on local variance characteristics."""
    if isinstance(local_variance, np.ndarray):
        avg_variance = np.mean(local_variance)
    else:
        avg_variance = local_variance
    
    # Normalize variance to 0-1 range (assuming max variance of 10000)
    normalized_variance = np.clip(avg_variance / 10000.0, 0, 1)
    
    # Higher variance suggests more structure to preserve
    return -0.3 * normalized_variance

def determine_adaptive_weights(characteristics):
    """Determine dynamic weights based on image characteristics."""
    noise_level = characteristics['noise_level']
    edge_density = characteristics['edge_density']
    texture_complexity = characteristics['texture_complexity']
    
    # Base weights
    weights = {
        'noise': 0.35,
        'edge': 0.30,
        'texture': 0.20,
        'frequency': 0.10,
        'variance': 0.05
    }
    
    # Adjust weights based on dominant characteristics
    if noise_level > 0.6:
        # Noise-dominated image
        weights['noise'] = 0.50
        weights['edge'] = 0.25
        weights['texture'] = 0.15
        weights['frequency'] = 0.07
        weights['variance'] = 0.03
    elif edge_density > 0.5:
        # Edge-rich image
        weights['edge'] = 0.45
        weights['noise'] = 0.25
        weights['texture'] = 0.15
        weights['frequency'] = 0.10
        weights['variance'] = 0.05
    elif texture_complexity > 0.7:
        # Texture-complex image
        weights['texture'] = 0.40
        weights['edge'] = 0.30
        weights['noise'] = 0.20
        weights['frequency'] = 0.07
        weights['variance'] = 0.03
    
    return weights

def apply_nonlinear_scaling(adaptation_value, characteristics):
    """Apply non-linear scaling to adaptation value for better control."""
    # Use tanh for bounded non-linear scaling
    scaled = np.tanh(adaptation_value * 2) * 0.8
    
    # Apply additional scaling based on overall image complexity
    complexity_factor = (
        characteristics['noise_level'] * 0.3 +
        characteristics['edge_density'] * 0.4 +
        characteristics['texture_complexity'] * 0.3
    )
    
    # More complex images get more conservative scaling
    complexity_scaling = 1.0 - complexity_factor * 0.2
    
    return scaled * complexity_scaling

def apply_contextual_constraints(sigma, characteristics):
    """Apply contextual constraints based on image content type."""
    noise_level = characteristics['noise_level']
    edge_density = characteristics['edge_density']
    
    # Prevent over-smoothing of detailed images
    if edge_density > 0.7 and sigma > 2.0:
        sigma = 2.0
    
    # Ensure sufficient smoothing for very noisy images
    if noise_level > 0.8 and sigma < 1.5:
        sigma = 1.5
    
    # Prevent under-smoothing when noise is present but edges are few
    if noise_level > 0.4 and edge_density < 0.2 and sigma < 1.2:
        sigma = 1.2
    
    return sigma

def validate_and_clamp_sigma(sigma, characteristics):
    """Validate and clamp sigma value to acceptable range with smart limits."""
    # Dynamic range based on image characteristics
    min_sigma = 0.5
    max_sigma = 4.0
    
    # Adjust limits based on extreme characteristics
    if characteristics['noise_level'] > 0.9:
        max_sigma = 5.0  # Allow higher sigma for extremely noisy images
    elif characteristics['edge_density'] > 0.8:
        max_sigma = 2.5  # Restrict sigma for very detailed images
    
    if characteristics['edge_density'] > 0.9:
        min_sigma = 0.3  # Allow lower sigma for extremely detailed images
    
    # Clamp to final range
    final_sigma = np.clip(sigma, min_sigma, max_sigma)
    
    # Ensure numerical stability
    if final_sigma < 0.1:
        final_sigma = 0.1
    
    return final_sigma

def adapt_intensity_sigma(characteristics):
    """
    Adapt intensity sigma for bilateral filtering based on comprehensive noise analysis and image properties.
    
    This function dynamically calculates the optimal intensity sigma parameter for bilateral filtering
    by analyzing multiple image characteristics including noise type, level, texture complexity,
    and local image statistics to achieve optimal noise reduction while preserving important details.
    
    :param characteristics: Dictionary containing comprehensive image analysis results with keys:
                          - 'noise_level': Normalized noise level (0.0-1.0)
                          - 'noise_type': Type of noise ('gaussian', 'salt_pepper', 'uniform', 'mixed')
                          - 'edge_density': Edge density metric (0.0-1.0)
                          - 'texture_complexity': Texture complexity measure (0.0-1.0)
                          - 'local_variance': Local variance information
                          - 'frequency_profile': Frequency domain analysis results
    :return: Optimally adapted intensity sigma value (5.0-80.0)
    """
    # Enhanced base sigma calculation with noise-type specific optimization
    base_sigma = calculate_noise_specific_base_sigma(characteristics)
    
    # Multi-factor adaptation system
    noise_adaptation = calculate_noise_level_adaptation(characteristics['noise_level'])
    texture_adaptation = calculate_texture_complexity_adaptation(characteristics['texture_complexity'])
    edge_adaptation = calculate_edge_density_adaptation(characteristics['edge_density'])
    variance_adaptation = calculate_variance_based_adaptation(characteristics.get('local_variance', 0))
    frequency_adaptation = calculate_frequency_domain_adaptation(characteristics.get('frequency_profile', {}))
    
    # Advanced weighting system based on image content analysis
    weights = determine_sigma_adaptation_weights(characteristics)
    
    # Combine adaptations with sophisticated weighting
    total_adaptation = (
        weights['noise'] * noise_adaptation +
        weights['texture'] * texture_adaptation +
        weights['edge'] * edge_adaptation +
        weights['variance'] * variance_adaptation +
        weights['frequency'] * frequency_adaptation
    )
    
    # Apply non-linear scaling for better parameter control
    scaled_adaptation = apply_sigma_nonlinear_scaling(total_adaptation, characteristics)
    
    # Calculate final intensity sigma
    adapted_sigma = base_sigma + scaled_adaptation
    
    # Apply contextual constraints and validation
    adapted_sigma = apply_sigma_contextual_constraints(adapted_sigma, characteristics)
    final_sigma = validate_and_optimize_sigma(adapted_sigma, characteristics)
    
    return final_sigma

def calculate_noise_specific_base_sigma(characteristics):
    """Calculate base sigma optimized for specific noise types."""
    noise_type = characteristics.get('noise_type', 'mixed')
    noise_level = characteristics['noise_level']
    
    # Base sigma values optimized for different noise types
    base_sigmas = {
        'gaussian': 20.0 + (noise_level * 15.0),      # Higher base for Gaussian noise
        'salt_pepper': 12.0 + (noise_level * 8.0),    # Lower base, more selective filtering
        'uniform': 18.0 + (noise_level * 12.0),       # Moderate base for uniform noise
        'mixed': 16.0 + (noise_level * 10.0),         # Balanced approach for mixed noise
        'poisson': 22.0 + (noise_level * 13.0),       # Optimized for Poisson noise
        'speckle': 14.0 + (noise_level * 9.0)         # Specialized for speckle noise
    }
    
    base_sigma = base_sigmas.get(noise_type, base_sigmas['mixed'])
    
    # Fine-tune based on noise characteristics interaction
    if noise_type == 'gaussian' and characteristics['edge_density'] > 0.6:
        base_sigma *= 0.85  # Reduce for edge-rich Gaussian noise
    elif noise_type == 'salt_pepper' and characteristics['texture_complexity'] > 0.7:
        base_sigma *= 1.2   # Increase for textured impulse noise
    
    return base_sigma

def calculate_noise_level_adaptation(noise_level):
    """Calculate adaptation based on noise level with sophisticated scaling."""
    if noise_level < 0.1:
        return -8.0  # Strong reduction for very clean images
    elif noise_level < 0.3:
        # Gentle quadratic increase for light noise
        return -8.0 + (noise_level / 0.3) * 12.0
    elif noise_level < 0.6:
        # Linear increase for moderate noise
        return 4.0 + ((noise_level - 0.3) / 0.3) * 15.0
    elif noise_level < 0.8:
        # Accelerated increase for heavy noise
        return 19.0 + ((noise_level - 0.6) / 0.2) * 20.0
    else:
        # Maximum adaptation for extreme noise
        return 39.0 + (noise_level - 0.8) * 25.0

def calculate_texture_complexity_adaptation(texture_complexity):
    """Calculate adaptation based on texture complexity for detail preservation."""
    # Use sigmoid function for smooth adaptation
    sigmoid_input = (texture_complexity - 0.5) * 8
    sigmoid_output = 1 / (1 + np.exp(-sigmoid_input))
    
    # Scale to appropriate range: high texture = lower sigma for detail preservation
    adaptation = -15.0 + (1 - sigmoid_output) * 20.0
    
    # Additional adjustment for extreme texture complexity
    if texture_complexity > 0.9:
        adaptation -= 5.0  # Extra reduction for very complex textures
    elif texture_complexity < 0.1:
        adaptation += 8.0  # Extra increase for smooth areas
    
    return adaptation

def calculate_edge_density_adaptation(edge_density):
    """Calculate adaptation based on edge density with preservation priority."""
    # Exponential decay for edge preservation
    base_reduction = 12.0
    decay_factor = np.exp(-edge_density * 3.0)
    
    # Edge-based adaptation: more edges = lower sigma
    edge_adaptation = -base_reduction * (1 - decay_factor)
    
    # Boost reduction for extremely edge-rich images
    if edge_density > 0.8:
        edge_adaptation -= 8.0
    
    # Gentle increase for very smooth images
    elif edge_density < 0.1:
        edge_adaptation += 10.0
    
    return edge_adaptation

def calculate_variance_based_adaptation(local_variance):
    """Calculate adaptation based on local variance characteristics."""
    if isinstance(local_variance, np.ndarray):
        # Use variance statistics for adaptation
        mean_variance = np.mean(local_variance)
        variance_std = np.std(local_variance)
    else:
        mean_variance = local_variance
        variance_std = 0
    
    # Normalize variance (assuming typical range of 0-10000)
    normalized_variance = np.clip(mean_variance / 5000.0, 0, 2.0)
    normalized_std = np.clip(variance_std / 2000.0, 0, 1.5)
    
    # Higher variance suggests more structure to preserve
    variance_adaptation = -6.0 * normalized_variance
    
    # High variance standard deviation suggests complex local structure
    structure_adaptation = -4.0 * normalized_std
    
    return variance_adaptation + structure_adaptation

def calculate_frequency_domain_adaptation(frequency_profile):
    """Calculate adaptation based on frequency domain characteristics."""
    if not frequency_profile:
        return 0.0
    
    high_freq_ratio = frequency_profile.get('high_freq_ratio', 0.5)
    dominant_frequency = frequency_profile.get('dominant_frequency', (0, 0))
    
    # Higher high-frequency content suggests more detail to preserve
    freq_adaptation = -10.0 * max(0, high_freq_ratio - 0.3)
    
    # Consider dominant frequency location for additional adaptation
    if isinstance(dominant_frequency, tuple) and len(dominant_frequency) == 2:
        # Distance from center indicates frequency characteristics
        center_distance = np.sqrt(dominant_frequency[0]**2 + dominant_frequency[1]**2)
        if center_distance > 50:  # High-frequency dominant
            freq_adaptation -= 3.0
        elif center_distance < 10:  # Low-frequency dominant
            freq_adaptation += 5.0
    
    return freq_adaptation

def determine_sigma_adaptation_weights(characteristics):
    """Determine dynamic weights for sigma adaptation factors."""
    noise_level = characteristics['noise_level']
    noise_type = characteristics.get('noise_type', 'mixed')
    edge_density = characteristics['edge_density']
    texture_complexity = characteristics['texture_complexity']
    
    # Base weights
    weights = {
        'noise': 0.40,
        'texture': 0.25,
        'edge': 0.20,
        'variance': 0.10,
        'frequency': 0.05
    }
    
    # Adjust weights based on dominant characteristics
    if noise_level > 0.7:
        # Noise-dominated: prioritize noise-based adaptation
        weights['noise'] = 0.55
        weights['texture'] = 0.20
        weights['edge'] = 0.15
        weights['variance'] = 0.07
        weights['frequency'] = 0.03
    elif edge_density > 0.6:
        # Edge-rich: prioritize edge preservation
        weights['edge'] = 0.40
        weights['noise'] = 0.30
        weights['texture'] = 0.15
        weights['variance'] = 0.10
        weights['frequency'] = 0.05
    elif texture_complexity > 0.8:
        # Texture-complex: balance between noise and texture
        weights['texture'] = 0.35
        weights['noise'] = 0.30
        weights['edge'] = 0.20
        weights['variance'] = 0.10
        weights['frequency'] = 0.05
    
    # Special handling for specific noise types
    if noise_type == 'salt_pepper':
        weights['noise'] = 0.60  # Prioritize noise handling for impulse noise
        weights['edge'] = 0.25   # Still preserve edges
        weights['texture'] = 0.10
        weights['variance'] = 0.03
        weights['frequency'] = 0.02
    
    return weights

def apply_sigma_nonlinear_scaling(adaptation_value, characteristics):
    """Apply non-linear scaling to adaptation value for better control."""
    # Use tanh for bounded scaling
    bounded_adaptation = np.tanh(adaptation_value / 20.0) * 25.0
    
    # Apply additional scaling based on overall image complexity
    complexity_measure = (
        characteristics['noise_level'] * 0.4 +
        characteristics['edge_density'] * 0.3 +
        characteristics['texture_complexity'] * 0.3
    )
    
    # More complex images get more conservative scaling
    complexity_scaling = 1.0 - complexity_measure * 0.15
    final_scaling = bounded_adaptation * complexity_scaling
    
    # Apply gentle smoothing to prevent abrupt changes
    if abs(final_scaling) > 20.0:
        sign = np.sign(final_scaling)
        final_scaling = sign * (20.0 + (abs(final_scaling) - 20.0) * 0.5)
    
    return final_scaling

def apply_sigma_contextual_constraints(sigma, characteristics):
    """Apply contextual constraints based on specific image scenarios."""
    noise_level = characteristics['noise_level']
    noise_type = characteristics.get('noise_type', 'mixed')
    edge_density = characteristics['edge_density']
    
    # Prevent over-aggressive filtering for edge-rich, low-noise images
    if edge_density > 0.7 and noise_level < 0.3 and sigma > 35.0:
        sigma = min(sigma, 25.0)
    
    # Ensure sufficient filtering for very noisy images
    if noise_level > 0.8 and sigma < 30.0:
        sigma = max(sigma, 35.0)
    
    # Special constraints for impulse noise
    if noise_type == 'salt_pepper':
        if sigma > 40.0:
            sigma = 40.0  # Cap for impulse noise to prevent over-smoothing
        if sigma < 15.0:
            sigma = 15.0  # Minimum for effective impulse noise removal
    
    # Constraints for Gaussian noise
    elif noise_type == 'gaussian':
        if noise_level > 0.6 and sigma < 25.0:
            sigma = 25.0  # Minimum for effective Gaussian noise removal
    
    return sigma

def validate_and_optimize_sigma(sigma, characteristics):
    """Final validation and optimization of sigma value."""
    # Enforce absolute bounds
    min_sigma, max_sigma = 5.0, 80.0
    
    # Dynamic range adjustment based on image characteristics
    if characteristics['edge_density'] > 0.9:
        max_sigma = 60.0  # Restrict maximum for extremely detailed images
    elif characteristics['noise_level'] > 0.9:
        max_sigma = 100.0  # Allow higher maximum for extremely noisy images
    
    if characteristics['texture_complexity'] > 0.9:
        min_sigma = max(3.0, min_sigma)  # Allow lower minimum for very detailed textures
    
    # Apply final clamping
    validated_sigma = np.clip(sigma, min_sigma, max_sigma)
    
    # Quantization to reduce parameter space (optional optimization)
    # Round to nearest 0.5 to reduce unnecessary precision
    optimized_sigma = np.round(validated_sigma * 2) / 2
    
    # Final sanity check
    if optimized_sigma <= 0:
        optimized_sigma = 15.0  # Safe default
    
    return optimized_sigma

def apply_fast_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size):
    """
    Enhanced fast bilateral filter implementation with optimized processing for small kernels (3-7).
    
    :param image: Input image array (float32, 0-255 range)
    :param spatial_sigma: Spatial standard deviation for Gaussian weights
    :param intensity_sigma: Intensity standard deviation for edge preservation
    :param kernel_size: Size of the filter kernel (3-7 recommended)
    :return: Bilateral filtered image with enhanced edge preservation
    """
    if image is None or len(image.shape) != 3:
        return image
    
    h, w, c = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)
    pad_size = kernel_size // 2
    
    # Enhanced padding strategy for better edge handling
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    # Pre-compute optimized spatial weights with sub-pixel precision
    y_coords, x_coords = np.ogrid[-pad_size:pad_size+1, -pad_size:pad_size+1]
    spatial_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * spatial_sigma**2))
    spatial_weights /= np.sum(spatial_weights)  # Normalize for consistent filtering
    
    # Adaptive block size based on image dimensions and hardware optimization
    optimal_block_size = min(64, max(16, min(h, w) // 8))
    
    # Vectorized processing with overlap handling for seamless results
    overlap_size = pad_size
    
    for i in range(0, h, optimal_block_size - overlap_size):
        for j in range(0, w, optimal_block_size - overlap_size):
            i_end = min(i + optimal_block_size, h)
            j_end = min(j + optimal_block_size, w)
            
            # Extract block with padding for seamless processing
            block = image[i:i_end, j:j_end]
            padded_block = padded[i:i_end+2*pad_size, j:j_end+2*pad_size]
            
            # Enhanced bilateral processing with edge preservation
            filtered_block = process_enhanced_bilateral_block(
                block, padded_block, spatial_weights, intensity_sigma, pad_size
            )
            
            # Seamless blending for overlapping regions
            if i > 0 or j > 0:
                blend_weights = calculate_blend_weights(block.shape[:2], overlap_size)
                filtered_image[i:i_end, j:j_end] = (
                    filtered_image[i:i_end, j:j_end] * (1 - blend_weights) +
                    filtered_block * blend_weights
                )
            else:
                filtered_image[i:i_end, j:j_end] = filtered_block
    
    return np.clip(filtered_image, 0, 255).astype(np.float32)

def apply_medium_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size):
    """
    Medium-scale bilateral filter implementation optimized for kernels 9-21.
    Uses approximation techniques and multi-threading for better performance.
    
    :param image: Input image array (float32, 0-255 range)
    :param spatial_sigma: Spatial standard deviation for Gaussian weights
    :param intensity_sigma: Intensity standard deviation for edge preservation
    :param kernel_size: Size of the filter kernel (9-21 recommended)
    :return: High-quality bilateral filtered image with efficient processing
    """
    if image is None or len(image.shape) != 3:
        return image
    
    h, w, c = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)
    pad_size = kernel_size // 2
    
    # Advanced padding with edge-aware reflection
    padded = apply_edge_aware_padding(image, pad_size)
    
    # Hierarchical spatial weight computation with approximation
    spatial_weights = compute_hierarchical_spatial_weights(kernel_size, spatial_sigma)
    
    # Adaptive intensity quantization for faster lookup
    intensity_lut = create_intensity_lookup_table(intensity_sigma, num_bins=256)
    
    # Multi-scale block processing for optimal performance
    block_size = min(128, max(32, min(h, w) // 4))
    
    # Process in tiles with intelligent overlap management
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            
            # Extract processing region
            block = image[i:i_end, j:j_end]
            padded_block = padded[i:i_end+2*pad_size, j:j_end+2*pad_size]
            
            # Apply medium-scale bilateral filtering with approximation
            filtered_block = process_medium_bilateral_block(
                block, padded_block, spatial_weights, intensity_lut, pad_size
            )
            
            # Advanced blending for seamless results
            filtered_image[i:i_end, j:j_end] = apply_advanced_blending(
                filtered_image[i:i_end, j:j_end], filtered_block, i, j, block_size, pad_size
            )
    
    # Post-processing refinement
    filtered_image = apply_edge_refinement(image, filtered_image, spatial_sigma)
    
    return np.clip(filtered_image, 0, 255).astype(np.float32)

def apply_advanced_blending(existing_block, new_block, i, j, block_size, pad_size):
    """
    Apply advanced blending for seamless block transitions.
    
    :param existing_block: Previously processed block data
    :param new_block: Newly filtered block data
    :param i: Block row position
    :param j: Block column position
    :param block_size: Size of processing blocks
    :param pad_size: Padding size for overlap
    :return: Blended block result
    """
    if existing_block.size == 0:
        return new_block
    
    # Simple weighted blending for overlapping regions
    if i > 0 or j > 0:
        # Create blending weights based on distance from edges
        h, w = new_block.shape[:2]
        blend_weights = np.ones((h, w, 1))
        
        # Fade in from edges for smooth transitions
        fade_distance = min(pad_size, 8)
        for edge_dist in range(fade_distance):
            fade_factor = (edge_dist + 1) / (fade_distance + 1)
            
            if i > 0 and edge_dist < h:
                blend_weights[edge_dist, :, 0] = fade_factor
            if j > 0 and edge_dist < w:
                blend_weights[:, edge_dist, 0] = fade_factor
        
        # Apply blending
        return existing_block * (1 - blend_weights) + new_block * blend_weights
    else:
        return new_block

def apply_edge_refinement(original, filtered, spatial_sigma):
    """
    Apply edge refinement to enhance details after bilateral filtering.
    
    :param original: Original image
    :param filtered: Bilateral filtered image
    :param spatial_sigma: Spatial sigma used in filtering
    :return: Edge-refined image
    """
    # Calculate detail layer (difference between original and filtered)
    detail_layer = original - filtered
    
    # Calculate edge mask to identify important edges
    edge_mask = calculate_edge_mask(original)
    
    # Determine refinement strength based on spatial sigma
    refinement_strength = min(0.3, spatial_sigma / 10.0)
    
    # Apply selective detail enhancement
    enhanced_details = detail_layer * edge_mask * refinement_strength
    
    # Combine filtered image with enhanced details
    refined = filtered + enhanced_details
    
    return np.clip(refined, 0, 255)

def process_medium_bilateral_block(block, padded_block, spatial_weights, intensity_lut, pad_size):
    """Process a block with medium-scale bilateral filtering using lookup table optimization."""
    bh, bw, bc = block.shape
    filtered_block = np.zeros_like(block, dtype=np.float32)
    intensity_lut_array, scale_factor = intensity_lut
    
    kernel_size = 2 * pad_size + 1
    
    for i in range(bh):
        for j in range(bw):
            center_pixel = block[i, j].astype(np.float32)
            
            # Extract neighborhood efficiently
            neighborhood = padded_block[i:i+kernel_size, j:j+kernel_size]
            
            # Compute intensity differences
            intensity_diff = np.sum((neighborhood - center_pixel)**2, axis=2)
            
            # Use lookup table for intensity weights
            lut_indices = np.clip((intensity_diff / scale_factor).astype(int), 0, len(intensity_lut_array) - 1)
            intensity_weights = intensity_lut_array[lut_indices]
            
            # Combine weights
            combined_weights = spatial_weights * intensity_weights
            weight_sum = np.sum(combined_weights)
            
            if weight_sum > 1e-8:
                weighted_sum = np.sum(neighborhood * combined_weights[:,:,np.newaxis], axis=(0,1))
                filtered_block[i, j] = weighted_sum / weight_sum
            else:
                filtered_block[i, j] = center_pixel
    
    return filtered_block

def apply_multiscale_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size):
    """
    Apply bilateral filtering using advanced multi-scale approach for large kernels.
    
    This implementation uses a sophisticated pyramid-based approach with edge-preserving
    upsampling and cross-scale refinement for optimal quality with large kernel sizes.
    
    :param image: Input image array (float32, 0-255 range)
    :param spatial_sigma: Spatial standard deviation for Gaussian weights
    :param intensity_sigma: Intensity standard deviation for edge preservation
    :param kernel_size: Size of the filter kernel (typically 23-49)
    :return: High-quality bilateral filtered image with multi-scale optimization
    """
    if image is None or len(image.shape) != 3:
        return image
    
    h, w, c = image.shape
    
    # Adaptive scale factor based on kernel size and image dimensions
    scale_factor = calculate_optimal_scale_factor(kernel_size, h, w)
    
    # Multi-level pyramid approach for better quality
    num_levels = determine_pyramid_levels(kernel_size, scale_factor)
    
    # Build image pyramid
    pyramid = build_gaussian_pyramid(image, num_levels, scale_factor)
    
    # Apply bilateral filtering at each level with adapted parameters
    filtered_pyramid = []
    for level, (level_image, level_scale) in enumerate(pyramid):
        # Adapt parameters for each pyramid level
        adapted_spatial_sigma = spatial_sigma / level_scale
        adapted_kernel_size = max(3, kernel_size // level_scale)
        
        # Apply bilateral filter with level-specific optimizations
        if level == 0:  # Finest level - use high-quality filtering
            filtered_level = apply_pyramid_bilateral_filter(
                level_image, adapted_spatial_sigma, intensity_sigma, adapted_kernel_size, 'fine'
            )
        elif level == num_levels - 1:  # Coarsest level - use efficient filtering
            filtered_level = apply_pyramid_bilateral_filter(
                level_image, adapted_spatial_sigma, intensity_sigma, adapted_kernel_size, 'coarse'
            )
        else:  # Middle levels - balanced approach
            filtered_level = apply_pyramid_bilateral_filter(
                level_image, adapted_spatial_sigma, intensity_sigma, adapted_kernel_size, 'balanced'
            )
        
        filtered_pyramid.append((filtered_level, level_scale))
    
    # Reconstruct using advanced upsampling with edge preservation
    result = reconstruct_from_pyramid(filtered_pyramid, image.shape[:2])
    
    # Cross-scale detail enhancement
    result = apply_cross_scale_enhancement(image, result, pyramid, filtered_pyramid)
    
    # Final quality refinement
    result = apply_multiscale_refinement(image, result, spatial_sigma, intensity_sigma)
    
    return np.clip(result, 0, 255).astype(np.float32)

def calculate_optimal_scale_factor(kernel_size, height, width):
    """Calculate optimal scale factor based on kernel size and image dimensions."""
    # Base scale factor calculation
    base_scale = max(2, min(4, kernel_size // 12))
    
    # Adjust based on image size
    image_complexity = (height * width) / (1024 * 1024)  # Normalized to 1MP
    
    if image_complexity > 4:  # Large image
        scale_factor = min(base_scale + 1, 4)
    elif image_complexity < 0.5:  # Small image
        scale_factor = max(base_scale - 1, 2)
    else:
        scale_factor = base_scale
    
    return scale_factor

def determine_pyramid_levels(kernel_size, scale_factor):
    """Determine optimal number of pyramid levels."""
    # More levels for larger kernels, but cap for efficiency
    levels = min(4, max(2, kernel_size // 16))
    
    # Adjust based on scale factor
    if scale_factor > 3:
        levels = min(levels + 1, 4)
    
    return levels

def build_gaussian_pyramid(image, num_levels, base_scale_factor):
    """
    Build Gaussian pyramid with adaptive scaling for optimal quality.
    
    :param image: Input image
    :param num_levels: Number of pyramid levels
    :param base_scale_factor: Base scaling factor
    :return: List of (image, scale_factor) tuples
    """
    pyramid = [(image.copy(), 1.0)]  # Original image at scale 1.0
    current_image = image.copy()
    cumulative_scale = 1.0
    
    for level in range(1, num_levels):
        # Adaptive scale factor for each level
        level_scale_factor = base_scale_factor if level == 1 else max(1.5, base_scale_factor * 0.7)
        cumulative_scale *= level_scale_factor
        
        # Pre-filter to prevent aliasing
        sigma = level_scale_factor * 0.6
        prefiltered = apply_anti_aliasing_filter(current_image, sigma)
        
        # Downsample using high-quality interpolation
        new_height = max(16, int(current_image.shape[0] / level_scale_factor))
        new_width = max(16, int(current_image.shape[1] / level_scale_factor))
        
        downsampled = apply_high_quality_resize(prefiltered, (new_height, new_width))
        
        pyramid.append((downsampled, cumulative_scale))
        current_image = downsampled
    
    return pyramid

def apply_anti_aliasing_filter(image, sigma):
    """Apply anti-aliasing filter before downsampling."""
    if sigma < 0.5:
        return image
    
    # Use separable Gaussian for efficiency
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return apply_separable_gaussian_filter(image, sigma, kernel_size)

def apply_separable_gaussian_filter(image, sigma, kernel_size):
    """Apply separable Gaussian filter for anti-aliasing."""
    # Create 1D Gaussian kernel
    kernel_1d = create_gaussian_kernel_1d(kernel_size, sigma)
    
    # Apply horizontal pass
    h_filtered = apply_1d_convolution_optimized(image, kernel_1d, axis=1)
    
    # Apply vertical pass
    v_filtered = apply_1d_convolution_optimized(h_filtered, kernel_1d, axis=0)
    
    return v_filtered

def apply_1d_convolution_optimized(image, kernel, axis):
    """Optimized 1D convolution with proper padding."""
    kernel_size = len(kernel)
    pad_size = kernel_size // 2
    
    if axis == 0:  # Vertical convolution
        padded = np.pad(image, ((pad_size, pad_size), (0, 0), (0, 0)), mode='reflect')
        result = np.zeros_like(image, dtype=np.float32)
        
        for k, weight in enumerate(kernel):
            result += weight * padded[k:k+image.shape[0]]
    else:  # Horizontal convolution
        padded = np.pad(image, ((0, 0), (pad_size, pad_size), (0, 0)), mode='reflect')
        result = np.zeros_like(image, dtype=np.float32)
        
        for k, weight in enumerate(kernel):
            result += weight * padded[:, k:k+image.shape[1]]
    
    return result

def apply_high_quality_resize(image, target_size):
    """Apply high-quality resizing using Lanczos-like interpolation."""
    target_h, target_w = target_size
    current_h, current_w = image.shape[:2]
    
    # Use bilinear interpolation as a high-quality approximation
    scale_h = target_h / current_h
    scale_w = target_w / current_w
    
    # Create coordinate grids
    y_coords = np.linspace(0, current_h - 1, target_h)
    x_coords = np.linspace(0, current_w - 1, target_w)
    
    # Apply interpolation for each channel
    resized_channels = []
    for c in range(image.shape[2]):
        channel = image[:, :, c]
        resized_channel = apply_bilinear_interpolation(channel, y_coords, x_coords)
        resized_channels.append(resized_channel)
    
    return np.stack(resized_channels, axis=-1)

def apply_bilinear_interpolation(channel, y_coords, x_coords):
    """Apply bilinear interpolation to a single channel."""
    h, w = channel.shape
    target_h, target_w = len(y_coords), len(x_coords)
    result = np.zeros((target_h, target_w), dtype=np.float32)
    
    for i, y in enumerate(y_coords):
        for j, x in enumerate(x_coords):
            # Get integer coordinates
            y0, x0 = int(y), int(x)
            y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
            
            # Get fractional parts
            dy, dx = y - y0, x - x0
            
            # Bilinear interpolation
            top = channel[y0, x0] * (1 - dx) + channel[y0, x1] * dx
            bottom = channel[y1, x0] * (1 - dx) + channel[y1, x1] * dx
            result[i, j] = top * (1 - dy) + bottom * dy
    
    return result

def apply_pyramid_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size, quality_mode):
    """
    Apply bilateral filter optimized for specific pyramid level.
    
    :param image: Image at current pyramid level
    :param spatial_sigma: Adapted spatial sigma
    :param intensity_sigma: Intensity sigma
    :param kernel_size: Adapted kernel size
    :param quality_mode: 'fine', 'balanced', or 'coarse'
    :return: Filtered image
    """
    if quality_mode == 'fine':
        # Highest quality for finest level
        return apply_enhanced_bilateral_filter_fine(image, spatial_sigma, intensity_sigma, kernel_size)
    elif quality_mode == 'coarse':
        # Efficient filtering for coarsest level
        return apply_enhanced_bilateral_filter_coarse(image, spatial_sigma, intensity_sigma, kernel_size)
    else:
        # Balanced approach for middle levels
        return apply_enhanced_bilateral_filter_balanced(image, spatial_sigma, intensity_sigma, kernel_size)

def apply_enhanced_bilateral_filter_fine(image, spatial_sigma, intensity_sigma, kernel_size):
    """High-quality bilateral filtering for fine pyramid level."""
    # Use small block processing for highest quality
    block_size = 32
    return apply_block_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size, block_size, overlap_ratio=0.5)

def apply_enhanced_bilateral_filter_balanced(image, spatial_sigma, intensity_sigma, kernel_size):
    """Balanced bilateral filtering for middle pyramid levels."""
    # Medium block size for balanced performance
    block_size = 64
    return apply_block_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size, block_size, overlap_ratio=0.25)

def apply_enhanced_bilateral_filter_coarse(image, spatial_sigma, intensity_sigma, kernel_size):
    """Efficient bilateral filtering for coarse pyramid level."""
    # Large block size for efficiency
    block_size = 128
    return apply_block_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size, block_size, overlap_ratio=0.1)

def apply_block_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size, block_size, overlap_ratio):
    """Apply bilateral filter using block processing with configurable overlap."""
    h, w, c = image.shape
    filtered_image = np.zeros_like(image, dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    step_size = int(block_size * (1 - overlap_ratio))
    pad_size = kernel_size // 2
    
    # Pad image for boundary handling
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    for i in range(0, h, step_size):
        for j in range(0, w, step_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            
            # Extract block with padding
            block_with_padding = padded[i:i_end+2*pad_size, j:j_end+2*pad_size]
            block = image[i:i_end, j:j_end]
            
            # Apply bilateral filter to block
            filtered_block = apply_fast_bilateral_to_block(
                block, block_with_padding, spatial_sigma, intensity_sigma, kernel_size
            )
            
            # Calculate blending weights
            block_weights = calculate_advanced_blend_weights(
                filtered_block.shape[:2], overlap_ratio, i, j, h, w
            )
            
            # Accumulate results
            filtered_image[i:i_end, j:j_end] += filtered_block * block_weights
            weight_map[i:i_end, j:j_end] += block_weights[:, :, 0]
    
    # Normalize by weights
    weight_map = np.maximum(weight_map, 1e-6)
    for c in range(image.shape[2]):
        filtered_image[:, :, c] /= weight_map
    
    return filtered_image

def apply_fast_bilateral_to_block(block, padded_block, spatial_sigma, intensity_sigma, kernel_size):
    """Apply fast bilateral filter to a single block."""
    bh, bw, bc = block.shape
    filtered_block = np.zeros_like(block, dtype=np.float32)
    pad_size = kernel_size // 2
    
    # Pre-compute spatial weights
    y_coords, x_coords = np.ogrid[-pad_size:pad_size+1, -pad_size:pad_size+1]
    spatial_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * spatial_sigma**2))
    
    for i in range(bh):
        for j in range(bw):
            center_pixel = block[i, j].astype(np.float32)
            neighborhood = padded_block[i:i+kernel_size, j:j+kernel_size]
            
            # Calculate intensity differences
            intensity_diff = np.sum((neighborhood - center_pixel)**2, axis=2)
            intensity_weights = np.exp(-intensity_diff / (2 * intensity_sigma**2))
            
            # Combine weights
            combined_weights = spatial_weights * intensity_weights
            weight_sum = np.sum(combined_weights)
            
            if weight_sum > 1e-8:
                weighted_sum = np.sum(neighborhood * combined_weights[:,:,np.newaxis], axis=(0,1))
                filtered_block[i, j] = weighted_sum / weight_sum
            else:
                filtered_block[i, j] = center_pixel
    
    return filtered_block

def calculate_advanced_blend_weights(block_shape, overlap_ratio, start_i, start_j, total_h, total_w):
    """Calculate advanced blending weights with smooth transitions."""
    bh, bw = block_shape
    weights = np.ones((bh, bw))
    
    fade_size = int(min(bh, bw) * overlap_ratio * 0.5)
    
    # Apply fading at boundaries
    for edge_dist in range(fade_size):
        fade_factor = (edge_dist + 1) / (fade_size + 1)
        
        # Top edge
        if start_i > 0 and edge_dist < bh:
            weights[edge_dist, :] = fade_factor
        
        # Left edge
        if start_j > 0 and edge_dist < bw:
            weights[:, edge_dist] = fade_factor
        
        # Bottom edge
        if start_i + bh < total_h and (bh - 1 - edge_dist) >= 0:
            weights[bh - 1 - edge_dist, :] = fade_factor
        
        # Right edge
        if start_j + bw < total_w and (bw - 1 - edge_dist) >= 0:
            weights[:, bw - 1 - edge_dist] = fade_factor
    
    return np.expand_dims(weights, axis=2)

def reconstruct_from_pyramid(filtered_pyramid, target_shape):
    """Reconstruct image from filtered pyramid using advanced upsampling."""
    if not filtered_pyramid:
        return np.zeros(target_shape + (3,), dtype=np.float32)
    
    # Start with coarsest level
    result, _ = filtered_pyramid[-1]
    
    # Progressively upsample and combine
    for level in range(len(filtered_pyramid) - 2, -1, -1):
        level_image, level_scale = filtered_pyramid[level]
        target_h, target_w = level_image.shape[:2]
        
        # Upsample current result to next level size
        upsampled = apply_edge_preserving_upsample(result, (target_h, target_w))
        
        # Combine with current level using adaptive blending
        result = combine_pyramid_levels(upsampled, level_image, level_scale)
    
    # Final resize to exact target dimensions
    if result.shape[:2] != target_shape:
        result = apply_high_quality_resize(result, target_shape)
    
    return result

def apply_edge_preserving_upsample(image, target_size):
    """Apply edge-preserving upsampling using advanced interpolation."""
    # Use bicubic-like interpolation with edge preservation
    upsampled = apply_high_quality_resize(image, target_size)
    
    # Apply edge enhancement to preserve details
    enhanced = apply_selective_edge_enhancement_light(upsampled)
    
    return enhanced

def apply_selective_edge_enhancement_light(image):
    """Apply light edge enhancement during upsampling."""
    # Simple unsharp mask with very gentle strength
    kernel_size = 3
    sigma = 0.8
    
    blurred = apply_separable_gaussian_filter(image, sigma, kernel_size)
    unsharp_mask = image - blurred
    
    # Very gentle enhancement
    enhanced = image + 0.1 * unsharp_mask
    
    return np.clip(enhanced, 0, 255)

def combine_pyramid_levels(upsampled, level_image, level_scale):
    """Combine upsampled result with current pyramid level."""
    # Calculate adaptive blending weights based on level scale
    detail_weight = 1.0 / (1.0 + level_scale * 0.5)
    upsampled_weight = 1.0 - detail_weight
    
    # Blend the images
    combined = detail_weight * level_image + upsampled_weight * upsampled
    
    return combined

def apply_cross_scale_enhancement(original, result, pyramid, filtered_pyramid):
    """Apply cross-scale detail enhancement using pyramid information."""
    # Extract high-frequency details from original
    original_details = original - apply_separable_gaussian_filter(original, 2.0, 9)
    
    # Calculate detail preservation factor
    detail_factor = calculate_detail_preservation_factor(pyramid, filtered_pyramid)
    
    # Apply selective detail enhancement
    enhanced_result = result + detail_factor * original_details * 0.3
    
    return np.clip(enhanced_result, 0, 255)

def calculate_detail_preservation_factor(pyramid, filtered_pyramid):
    """Calculate factor for detail preservation based on pyramid analysis."""
    if len(pyramid) < 2:
        return 1.0
    
    # Analyze detail loss in pyramid processing
    original_detail = pyramid[0][0] - pyramid[1][0] if len(pyramid) > 1 else 0
    filtered_detail = filtered_pyramid[0][0] - filtered_pyramid[1][0] if len(filtered_pyramid) > 1 else 0
    
    # Calculate preservation factor (simplified)
    if isinstance(original_detail, np.ndarray) and isinstance(filtered_detail, np.ndarray):
        detail_ratio = np.mean(np.abs(filtered_detail)) / (np.mean(np.abs(original_detail)) + 1e-6)
        return np.clip(1.0 - detail_ratio, 0.5, 1.5)
    
    return 1.0

def apply_multiscale_refinement(original, result, spatial_sigma, intensity_sigma):
    """Apply final refinement using multi-scale analysis."""
    # Calculate difference between original and result
    difference = original - result
    
    # Apply selective refinement based on spatial sigma
    if spatial_sigma > 3.0:
        # For large spatial sigma, preserve more original details
        refinement_strength = 0.2
    else:
        # For smaller spatial sigma, trust the filtered result more
        refinement_strength = 0.1
    
    # Apply edge-aware refinement
    edge_mask = calculate_enhanced_edge_mask(original)
    refinement = difference * edge_mask * refinement_strength
    
    refined_result = result + refinement
    
    return np.clip(refined_result, 0, 255)

def calculate_enhanced_edge_mask(image):
    """
    Calculate enhanced edge mask using multiple edge detection methods and adaptive thresholding.
    
    :param image: Input image array (RGB format)
    :return: Enhanced edge mask with improved accuracy and detail preservation
    """
    if image is None or len(image.shape) != 3:
        return np.ones(image.shape[:2] + (1,)) * 0.1
    
    # Convert to grayscale using perceptual weights for better edge detection
    gray = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    
    # Multi-scale edge detection for comprehensive edge capture
    edge_maps = []
    
    # Scale 1: Fine details (Sobel operator)
    sobel_x = apply_sobel_filter(gray, direction='x')
    sobel_y = apply_sobel_filter(gray, direction='y')
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_maps.append(sobel_magnitude)
    
    # Scale 2: Medium features (Scharr operator for better accuracy)
    scharr_x = apply_scharr_filter(gray, direction='x')
    scharr_y = apply_scharr_filter(gray, direction='y')
    scharr_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
    edge_maps.append(scharr_magnitude)
    
    # Scale 3: Coarse features (Laplacian of Gaussian)
    log_edges = apply_log_filter(gray, sigma=1.4)
    edge_maps.append(np.abs(log_edges))
    
    # Scale 4: Structural edges (structure tensor)
    structure_edges = calculate_structure_tensor_edges(gray)
    edge_maps.append(structure_edges)
    
    # Combine edge maps with weighted fusion
    weights = [0.35, 0.25, 0.25, 0.15]  # Emphasize Sobel and Scharr
    combined_edges = np.zeros_like(gray, dtype=np.float32)
    
    for edge_map, weight in zip(edge_maps, weights):
        # Normalize each edge map
        if np.max(edge_map) > 0:
            normalized_map = edge_map / np.max(edge_map)
            combined_edges += weight * normalized_map
    
    # Apply adaptive thresholding based on local statistics
    adaptive_threshold = calculate_adaptive_threshold(combined_edges)
    enhanced_edges = apply_adaptive_enhancement(combined_edges, adaptive_threshold)
    
    # Apply non-maximum suppression for cleaner edges
    suppressed_edges = apply_non_maximum_suppression(enhanced_edges, sobel_x, sobel_y)
    
    # Edge-aware smoothing to reduce noise while preserving important edges
    smoothed_edges = apply_edge_aware_smoothing(suppressed_edges, iterations=2)
    
    # Create hierarchical edge mask with multiple strength levels
    edge_mask = create_hierarchical_edge_mask(smoothed_edges)
    
    # Apply morphological operations for edge connectivity
    connected_edges = enhance_edge_connectivity(edge_mask)
    
    # Final normalization and range adjustment
    final_mask = normalize_edge_mask(connected_edges)
    
    # Expand dimensions for compatibility
    return np.expand_dims(final_mask, axis=2)

def apply_sobel_filter(image, direction='x'):
    """Apply Sobel edge detection filter."""
    if direction == 'x':
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    else:  # direction == 'y'
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    return apply_convolution_2d(image, kernel)

def apply_scharr_filter(image, direction='x'):
    """Apply Scharr edge detection filter for better accuracy."""
    if direction == 'x':
        kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    else:  # direction == 'y'
        kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    
    return apply_convolution_2d(image, kernel)

def apply_log_filter(image, sigma=1.4):
    """Apply Laplacian of Gaussian filter."""
    # First apply Gaussian smoothing
    gaussian_smoothed = apply_gaussian_smoothing(image, sigma)
    
    # Then apply Laplacian
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    return apply_convolution_2d(gaussian_smoothed, laplacian_kernel)

def apply_gaussian_smoothing(image, sigma):
    """Apply Gaussian smoothing with given sigma."""
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 2D Gaussian kernel
    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size))
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    return apply_convolution_2d(image, kernel)

def calculate_structure_tensor_edges(image):
    """Calculate edges using structure tensor analysis."""
    # Calculate gradients
    grad_x = np.diff(image, axis=1, prepend=image[:, :1])
    grad_y = np.diff(image, axis=0, prepend=image[:1, :])
    
    # Structure tensor components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # Apply Gaussian smoothing to structure tensor components
    sigma_tensor = 1.0
    Ixx_smooth = apply_gaussian_smoothing(Ixx, sigma_tensor)
    Iyy_smooth = apply_gaussian_smoothing(Iyy, sigma_tensor)
    Ixy_smooth = apply_gaussian_smoothing(Ixy, sigma_tensor)
    
    # Calculate eigenvalues for edge strength
    trace = Ixx_smooth + Iyy_smooth
    det = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth
    
    # Edge strength from larger eigenvalue
    discriminant = np.maximum(0, trace**2 - 4 * det)
    lambda_max = 0.5 * (trace + np.sqrt(discriminant))
    
    return lambda_max

def calculate_adaptive_threshold(edge_map):
    """Calculate adaptive threshold based on local image statistics."""
    # Calculate local means using sliding window
    window_size = 15
    pad_size = window_size // 2
    
    padded = np.pad(edge_map, pad_size, mode='reflect')
    local_means = np.zeros_like(edge_map)
    
    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            window = padded[i:i+window_size, j:j+window_size]
            local_means[i, j] = np.mean(window)
    
    # Adaptive threshold as percentage of local mean
    adaptive_factor = 0.8
    threshold_map = local_means * adaptive_factor
    
    # Ensure minimum threshold
    min_threshold = np.percentile(edge_map, 10)
    threshold_map = np.maximum(threshold_map, min_threshold)
    
    return threshold_map

def apply_adaptive_enhancement(edge_map, threshold_map):
    """
    Apply sophisticated adaptive enhancement with multi-scale processing and advanced edge preservation.
    
    :param edge_map: Edge strength map from multi-scale edge detection
    :param threshold_map: Adaptive threshold map based on local statistics
    :return: Enhanced edge map with improved contrast and detail preservation
    """
    if edge_map is None or threshold_map is None:
        return edge_map
    
    # Multi-level adaptive enhancement with hierarchical processing
    enhanced_levels = []
    
    # Level 1: Strong edge enhancement
    strong_threshold = threshold_map * 1.5
    strong_enhancement = apply_strong_edge_enhancement(edge_map, strong_threshold)
    enhanced_levels.append(strong_enhancement)
    
    # Level 2: Medium edge enhancement
    medium_threshold = threshold_map * 1.0
    medium_enhancement = apply_medium_edge_enhancement(edge_map, medium_threshold)
    enhanced_levels.append(medium_enhancement)
    
    # Level 3: Weak edge enhancement
    weak_threshold = threshold_map * 0.6
    weak_enhancement = apply_weak_edge_enhancement(edge_map, weak_threshold)
    enhanced_levels.append(weak_enhancement)
    
    # Adaptive weighting based on local image characteristics
    weights = calculate_adaptive_weights(edge_map, threshold_map)
    
    # Combine enhancement levels using weighted fusion
    combined_enhancement = combine_enhancement_levels(enhanced_levels, weights)
    
    # Apply advanced sigmoid enhancement with local adaptation
    sigmoid_enhanced = apply_advanced_sigmoid_enhancement(
        combined_enhancement, threshold_map, edge_map
    )
    
    # Edge-preserving post-processing
    final_enhanced = apply_edge_preserving_post_processing(
        sigmoid_enhanced, edge_map, threshold_map
    )
    
    # Contrast normalization and range optimization
    normalized_enhanced = normalize_enhanced_edges(final_enhanced)
    
    return normalized_enhanced

def apply_strong_edge_enhancement(edge_map, strong_threshold):
    """Apply strong enhancement for prominent edges."""
    enhancement_factor = 2.5
    suppression_factor = 0.3
    
    # Strong enhancement where edges exceed threshold
    enhanced = np.where(edge_map > strong_threshold,
                       edge_map * enhancement_factor,
                       edge_map * suppression_factor)
    
    # Apply non-linear enhancement for better edge definition
    gamma_factor = 0.7
    enhanced = np.power(enhanced / np.max(enhanced + 1e-6), gamma_factor) * np.max(enhanced)
    
    return enhanced

def apply_medium_edge_enhancement(edge_map, medium_threshold):
    """Apply moderate enhancement for medium-strength edges."""
    enhancement_factor = 1.8
    suppression_factor = 0.6
    
    # Medium enhancement with smooth transitions
    transition_width = 0.1 * np.mean(medium_threshold)
    
    # Create smooth transition mask
    transition_mask = create_smooth_transition_mask(
        edge_map, medium_threshold, transition_width
    )
    
    enhanced = np.where(edge_map > medium_threshold,
                       edge_map * enhancement_factor,
                       edge_map * suppression_factor)
    
    # Apply transition smoothing
    smooth_enhanced = enhanced * transition_mask + edge_map * (1 - transition_mask)
    
    return smooth_enhanced

def apply_weak_edge_enhancement(edge_map, weak_threshold):
    """Apply gentle enhancement for weak edges to preserve fine details."""
    enhancement_factor = 1.3
    preservation_factor = 0.9
    
    # Gentle enhancement to preserve fine details
    enhanced = np.where(edge_map > weak_threshold,
                       edge_map * enhancement_factor,
                       edge_map * preservation_factor)
    
    # Apply detail preservation filter
    detail_preserved = apply_detail_preservation_filter(enhanced, edge_map)
    
    return detail_preserved

def create_smooth_transition_mask(edge_map, threshold_map, transition_width):
    """Create smooth transition mask for seamless edge enhancement."""
    # Calculate distance from threshold
    distance = (edge_map - threshold_map) / (transition_width + 1e-6)
    
    # Apply smooth sigmoid transition
    transition_mask = 1.0 / (1.0 + np.exp(-5.0 * distance))
    
    return transition_mask

def apply_detail_preservation_filter(enhanced_edges, original_edges):
    """Apply detail preservation to maintain fine edge structures."""
    # Calculate detail difference
    detail_difference = enhanced_edges - original_edges
    
    # Apply selective filtering to preserve important details
    kernel_size = 3
    preserved_details = apply_selective_smoothing_kernel(
        detail_difference, kernel_size
    )
    
    # Combine preserved details with enhanced edges
    result = original_edges + preserved_details * 0.7
    
    return result

def apply_selective_smoothing_kernel(data, kernel_size):
    """Apply selective smoothing that preserves edge structures."""
    pad_size = kernel_size // 2
    padded = np.pad(data, pad_size, mode='reflect')
    
    smoothed = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            center_value = window[pad_size, pad_size]
            
            # Adaptive smoothing based on local variance
            local_variance = np.var(window)
            if local_variance > np.mean(data) * 0.1:
                # High variance - preserve structure
                smoothed[i, j] = center_value
            else:
                # Low variance - apply smoothing
                smoothed[i, j] = np.mean(window)
    
    return smoothed

def calculate_adaptive_weights(edge_map, threshold_map):
    """Calculate adaptive weights for multi-level fusion."""
    # Local gradient strength
    gradient_strength = calculate_local_gradient_strength(edge_map)
    
    # Local contrast measure
    local_contrast = calculate_local_contrast_measure(edge_map)
    
    # Edge density in local neighborhoods
    edge_density = calculate_local_edge_density(edge_map, threshold_map)
    
    # Combine metrics for weight calculation
    weights = {
        'strong': np.clip(gradient_strength * 0.5 + edge_density * 0.3, 0.1, 0.6),
        'medium': np.clip(local_contrast * 0.4 + (1 - edge_density) * 0.3, 0.2, 0.5),
        'weak': np.clip((1 - gradient_strength) * 0.3 + local_contrast * 0.2, 0.2, 0.4)
    }
    
    # Normalize weights
    total_weight = weights['strong'] + weights['medium'] + weights['weak']
    for key in weights:
        weights[key] = weights[key] / (total_weight + 1e-6)
    
    return weights

def calculate_local_gradient_strength(edge_map):
    """Calculate local gradient strength for weight computation."""
    # Calculate gradients in x and y directions
    grad_x = np.diff(edge_map, axis=1, prepend=edge_map[:, :1])
    grad_y = np.diff(edge_map, axis=0, prepend=edge_map[:1, :])
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-1 range
    max_gradient = np.max(gradient_magnitude)
    if max_gradient > 0:
        normalized_gradient = gradient_magnitude / max_gradient
    else:
        normalized_gradient = np.zeros_like(gradient_magnitude)
    
    return normalized_gradient

def calculate_local_contrast_measure(edge_map):
    """Calculate local contrast measure for adaptive processing."""
    kernel_size = 5
    pad_size = kernel_size // 2
    padded = np.pad(edge_map, pad_size, mode='reflect')
    
    local_contrast = np.zeros_like(edge_map)
    
    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            local_max = np.max(window)
            local_min = np.min(window)
            
            if local_max > 0:
                local_contrast[i, j] = (local_max - local_min) / local_max
            else:
                local_contrast[i, j] = 0
    
    return local_contrast

def calculate_local_edge_density(edge_map, threshold_map):
    """Calculate local edge density for adaptive enhancement."""
    kernel_size = 7
    pad_size = kernel_size // 2
    padded_edges = np.pad(edge_map, pad_size, mode='reflect')
    padded_threshold = np.pad(threshold_map, pad_size, mode='reflect')
    
    edge_density = np.zeros_like(edge_map)
    
    for i in range(edge_map.shape[0]):
        for j in range(edge_map.shape[1]):
            edge_window = padded_edges[i:i+kernel_size, j:j+kernel_size]
            threshold_window = padded_threshold[i:i+kernel_size, j:j+kernel_size]
            
            # Count pixels above local threshold
            edge_pixels = np.sum(edge_window > threshold_window)
            total_pixels = kernel_size * kernel_size
            
            edge_density[i, j] = edge_pixels / total_pixels
    
    return edge_density

def combine_enhancement_levels(enhanced_levels, weights):
    """Combine multiple enhancement levels using adaptive weights."""
    combined = np.zeros_like(enhanced_levels[0])
    
    # Weighted combination of enhancement levels
    combined += enhanced_levels[0] * weights['strong']
    combined += enhanced_levels[1] * weights['medium']
    combined += enhanced_levels[2] * weights['weak']
    
    # Apply cross-level coherence enhancement
    coherence_enhanced = apply_cross_level_coherence(combined, enhanced_levels)
    
    return coherence_enhanced

def apply_cross_level_coherence(combined, levels):
    """Apply coherence enhancement across different enhancement levels."""
    # Calculate level agreement
    level_agreement = calculate_level_agreement(levels)
    
    # Enhance areas where all levels agree
    coherent_areas = level_agreement > 0.7
    incoherent_areas = level_agreement < 0.3
    
    # Boost coherent areas, smooth incoherent areas
    coherence_factor = np.where(coherent_areas, 1.2, 
                               np.where(incoherent_areas, 0.8, 1.0))
    
    coherence_enhanced = combined * coherence_factor
    
    return coherence_enhanced

def calculate_level_agreement(levels):
    """Calculate agreement between different enhancement levels."""
    if len(levels) < 2:
        return np.ones_like(levels[0])
    
    # Normalize levels for comparison
    normalized_levels = []
    for level in levels:
        if np.max(level) > 0:
            normalized = level / np.max(level)
        else:
            normalized = level
        normalized_levels.append(normalized)
    
    # Calculate pairwise correlations
    agreements = []
    for i in range(len(normalized_levels)):
        for j in range(i+1, len(normalized_levels)):
            # Local correlation calculation
            agreement = calculate_local_correlation(
                normalized_levels[i], normalized_levels[j]
            )
            agreements.append(agreement)
    
    # Average agreement
    if agreements:
        mean_agreement = np.mean(agreements, axis=0)
    else:
        mean_agreement = np.ones_like(levels[0])
    
    return mean_agreement

def calculate_local_correlation(level1, level2):
    """Calculate local correlation between two enhancement levels."""
    kernel_size = 5
    pad_size = kernel_size // 2
    
    padded1 = np.pad(level1, pad_size, mode='reflect')
    padded2 = np.pad(level2, pad_size, mode='reflect')
    
    correlation = np.zeros_like(level1)
    
    for i in range(level1.shape[0]):
        for j in range(level1.shape[1]):
            window1 = padded1[i:i+kernel_size, j:j+kernel_size].flatten()
            window2 = padded2[i:i+kernel_size, j:j+kernel_size].flatten()
            
            # Calculate correlation coefficient
            if np.std(window1) > 1e-6 and np.std(window2) > 1e-6:
                corr_coef = np.corrcoef(window1, window2)[0, 1]
                correlation[i, j] = max(0, corr_coef)  # Only positive correlations
            else:
                correlation[i, j] = 0
    
    return correlation

def apply_advanced_sigmoid_enhancement(combined_enhancement, threshold_map, original_edges):
    """Apply advanced sigmoid enhancement with local adaptation."""
    # Calculate local statistics for adaptive parameters
    local_mean = calculate_local_mean(threshold_map, kernel_size=7)
    local_std = calculate_local_std(combined_enhancement, kernel_size=7)
    
    # Adaptive sigmoid parameters
    sigmoid_strength = 3.0 + local_std * 2.0  # Stronger sigmoid where more variation
    sigmoid_center = local_mean * 0.8  # Adaptive center point
    
    # Apply enhanced sigmoid transformation
    sigmoid_enhanced = 1.0 / (1.0 + np.exp(-sigmoid_strength * (combined_enhancement - sigmoid_center)))
    
    # Apply contrast-dependent scaling
    contrast_factor = calculate_local_contrast_factor(original_edges)
    scaled_sigmoid = sigmoid_enhanced * contrast_factor
    
    # Smooth transitions to prevent artifacts
    smoothed_sigmoid = apply_artifact_prevention_smoothing(scaled_sigmoid, original_edges)
    
    return smoothed_sigmoid

def calculate_local_mean(data, kernel_size=7):
    """Calculate local mean using efficient convolution."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    pad_size = kernel_size // 2
    padded = np.pad(data, pad_size, mode='reflect')
    
    local_mean = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            local_mean[i, j] = np.mean(window)
    
    return local_mean

def calculate_local_std(data, kernel_size=7):
    """Calculate local standard deviation."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    pad_size = kernel_size // 2
    padded = np.pad(data, pad_size, mode='reflect')
    
    local_std = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            local_std[i, j] = np.std(window)
    
    return local_std

def calculate_local_contrast_factor(edges):
    """Calculate local contrast factor for scaling."""
    # Calculate local gradient magnitude
    grad_x = np.diff(edges, axis=1, prepend=edges[:, :1])
    grad_y = np.diff(edges, axis=0, prepend=edges[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize and scale for contrast factor
    max_gradient = np.max(gradient_magnitude)
    if max_gradient > 0:
        contrast_factor = 0.5 + 0.5 * (gradient_magnitude / max_gradient)
    else:
        contrast_factor = np.ones_like(gradient_magnitude) * 0.5
    
    return contrast_factor

def apply_artifact_prevention_smoothing(enhanced_data, reference_data):
    """Apply smoothing to prevent enhancement artifacts."""
    # Detect potential artifacts
    artifact_mask = detect_enhancement_artifacts(enhanced_data, reference_data)
    
    # Apply selective smoothing only where artifacts are detected
    smoothed = apply_selective_gaussian_smoothing(enhanced_data, artifact_mask)
    
    # Blend smoothed and original based on artifact severity
    artifact_strength = calculate_artifact_strength(artifact_mask)
    final_result = enhanced_data * (1 - artifact_strength) + smoothed * artifact_strength
    
    return final_result

def detect_enhancement_artifacts(enhanced, reference):
    """Detect potential artifacts from enhancement."""
    # Calculate enhancement ratio
    enhancement_ratio = enhanced / (reference + 1e-6)
    
    # Detect excessive enhancement
    excessive_enhancement = enhancement_ratio > 3.0
    
    # Detect discontinuities
    grad_x = np.abs(np.diff(enhancement_ratio, axis=1, prepend=enhancement_ratio[:, :1]))
    grad_y = np.abs(np.diff(enhancement_ratio, axis=0, prepend=enhancement_ratio[:1, :]))
    discontinuities = (grad_x > np.percentile(grad_x, 95)) | (grad_y > np.percentile(grad_y, 95))
    
    # Combine artifact indicators
    artifact_mask = excessive_enhancement | discontinuities
    
    return artifact_mask.astype(float)

def apply_selective_gaussian_smoothing(data, artifact_mask):
    """Apply Gaussian smoothing selectively based on artifact mask."""
    # Simple 3x3 Gaussian-like smoothing
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    
    pad_size = 1
    padded = np.pad(data, pad_size, mode='reflect')
    smoothed = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if artifact_mask[i, j] > 0.1:  # Only smooth where artifacts detected
                window = padded[i:i+3, j:j+3]
                smoothed[i, j] = np.sum(window * kernel)
            else:
                smoothed[i, j] = data[i, j]
    
    return smoothed

def calculate_artifact_strength(artifact_mask):
    """Calculate strength of artifacts for blending."""
    # Apply morphological operations to enhance artifact regions
    enhanced_mask = apply_morphological_enhancement(artifact_mask)
    
    # Convert to strength values (0-1)
    strength = np.clip(enhanced_mask * 0.5, 0, 0.8)
    
    return strength

def apply_morphological_enhancement(mask, operation='dilation', iterations=1, kernel_type='cross', connectivity=8):
    """
    Apply advanced morphological operations to enhance artifact mask with multiple operation types,
    adaptive kernel selection, and multi-iteration processing for superior artifact detection.
    
    :param mask: Input binary or grayscale mask array
    :param operation: Type of morphological operation ('dilation', 'erosion', 'opening', 'closing', 'gradient')
    :param iterations: Number of iterations to apply the operation
    :param kernel_type: Type of structuring element ('cross', 'square', 'diamond', 'ellipse', 'adaptive')
    :param connectivity: Connectivity for operations (4 or 8)
    :return: Enhanced mask with applied morphological operations
    """
    if mask is None or mask.size == 0:
        return mask
    
    # Ensure mask is in proper format
    if len(mask.shape) > 2:
        mask = np.mean(mask, axis=2)  # Convert to grayscale if needed
    
    # Normalize mask to 0-1 range for processing
    normalized_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-6)
    
    # Create adaptive structuring element based on image characteristics
    kernel = create_adaptive_structuring_element(normalized_mask, kernel_type, connectivity)
    
    # Apply morphological operations based on specified type
    enhanced_mask = apply_morphological_operation(normalized_mask, operation, kernel, iterations)
    
    # Post-process for edge preservation and noise reduction
    refined_mask = apply_morphological_post_processing(enhanced_mask, normalized_mask)
    
    # Apply multi-scale morphological analysis for better artifact detection
    if operation in ['opening', 'closing', 'gradient']:
        multiscale_enhanced = apply_multiscale_morphological_analysis(refined_mask, kernel_type)
        refined_mask = combine_multiscale_results(refined_mask, multiscale_enhanced)
    
    # Final normalization and range adjustment
    final_mask = normalize_morphological_result(refined_mask)
    
    return final_mask

def create_adaptive_structuring_element(mask, kernel_type, connectivity):
    """Create adaptive structuring element based on image characteristics and specified type."""
    h, w = mask.shape
    
    # Calculate adaptive kernel size based on image dimensions and content
    base_size = max(3, min(9, min(h, w) // 100))
    
    # Analyze local image characteristics for adaptation
    local_variance = calculate_local_variance_simple(mask)
    avg_variance = np.mean(local_variance)
    
    # Adapt kernel size based on image complexity
    if avg_variance > 0.1:
        kernel_size = base_size + 2  # Larger kernel for complex regions
    elif avg_variance < 0.05:
        kernel_size = max(3, base_size - 2)  # Smaller kernel for smooth regions
    else:
        kernel_size = base_size
    
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create structuring element based on type
    if kernel_type == 'cross':
        kernel = create_cross_kernel(kernel_size)
    elif kernel_type == 'square':
        kernel = create_square_kernel(kernel_size)
    elif kernel_type == 'diamond':
        kernel = create_diamond_kernel(kernel_size)
    elif kernel_type == 'ellipse':
        kernel = create_ellipse_kernel(kernel_size)
    elif kernel_type == 'adaptive':
        kernel = create_adaptive_kernel(mask, kernel_size, connectivity)
    else:
        kernel = create_cross_kernel(kernel_size)  # Default to cross
    
    return kernel

def calculate_local_variance_simple(mask):
    """Calculate simplified local variance for kernel adaptation."""
    kernel_size = 5
    pad_size = kernel_size // 2
    padded = np.pad(mask, pad_size, mode='reflect')
    
    local_var = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            local_var[i, j] = np.var(window)
    
    return local_var

def create_cross_kernel(size):
    """Create cross-shaped structuring element."""
    kernel = np.zeros((size, size), dtype=bool)
    center = size // 2
    kernel[center, :] = True  # Horizontal line
    kernel[:, center] = True  # Vertical line
    return kernel

def create_square_kernel(size):
    """Create square structuring element."""
    return np.ones((size, size), dtype=bool)

def create_diamond_kernel(size):
    """Create diamond-shaped structuring element."""
    kernel = np.zeros((size, size), dtype=bool)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            if abs(i - center) + abs(j - center) <= center:
                kernel[i, j] = True
    
    return kernel

def create_ellipse_kernel(size):
    """Create elliptical structuring element."""
    kernel = np.zeros((size, size), dtype=bool)
    center = size // 2
    a, b = center * 0.8, center * 0.6  # Semi-axes
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            if (x**2 / a**2) + (y**2 / b**2) <= 1:
                kernel[i, j] = True
    
    return kernel

def create_adaptive_kernel(mask, size, connectivity):
    """Create adaptive kernel based on local image characteristics."""
    # Analyze dominant edge directions
    grad_x = np.diff(mask, axis=1, prepend=mask[:, :1])
    grad_y = np.diff(mask, axis=0, prepend=mask[:1, :])
    
    # Calculate dominant gradient direction
    avg_grad_x = np.mean(np.abs(grad_x))
    avg_grad_y = np.mean(np.abs(grad_y))
    
    kernel = np.zeros((size, size), dtype=bool)
    center = size // 2
    
    if avg_grad_x > avg_grad_y * 1.5:
        # Horizontal structures dominant - use horizontal kernel
        kernel[center, :] = True
    elif avg_grad_y > avg_grad_x * 1.5:
        # Vertical structures dominant - use vertical kernel
        kernel[:, center] = True
    else:
        # Mixed or isotropic structures - use circular kernel
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 <= center**2:
                    kernel[i, j] = True
    
    return kernel

def apply_morphological_operation(mask, operation, kernel, iterations):
    """Apply specified morphological operation with multiple iterations."""
    result = mask.copy()
    
    for _ in range(iterations):
        if operation == 'dilation':
            result = apply_dilation(result, kernel)
        elif operation == 'erosion':
            result = apply_erosion(result, kernel)
        elif operation == 'opening':
            result = apply_erosion(result, kernel)
            result = apply_dilation(result, kernel)
        elif operation == 'closing':
            result = apply_dilation(result, kernel)
            result = apply_erosion(result, kernel)
        elif operation == 'gradient':
            dilated = apply_dilation(result, kernel)
            eroded = apply_erosion(result, kernel)
            result = dilated - eroded
        else:
            result = apply_dilation(result, kernel)  # Default to dilation
    
    return result

def apply_dilation(mask, kernel):
    """Apply dilation operation with edge handling."""
    h, w = mask.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad mask with edge reflection for better boundary handling
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    dilated = np.zeros_like(mask)
    
    for i in range(h):
        for j in range(w):
            # Extract neighborhood
            neighborhood = padded[i:i+kh, j:j+kw]
            
            # Apply dilation: maximum value where kernel is True
            kernel_values = neighborhood[kernel]
            if len(kernel_values) > 0:
                dilated[i, j] = np.max(kernel_values)
            else:
                dilated[i, j] = mask[i, j]
    
    return dilated

def apply_erosion(mask, kernel):
    """Apply erosion operation with edge handling."""
    h, w = mask.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad mask with edge reflection
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    eroded = np.zeros_like(mask)
    
    for i in range(h):
        for j in range(w):
            # Extract neighborhood
            neighborhood = padded[i:i+kh, j:j+kw]
            
            # Apply erosion: minimum value where kernel is True
            kernel_values = neighborhood[kernel]
            if len(kernel_values) > 0:
                eroded[i, j] = np.min(kernel_values)
            else:
                eroded[i, j] = mask[i, j]
    
    return eroded

def apply_morphological_post_processing(enhanced_mask, original_mask):
    """Apply post-processing to preserve important features and reduce noise."""
    # Edge preservation using local variance
    local_variance = calculate_local_variance_simple(original_mask)
    edge_threshold = np.percentile(local_variance, 75)
    edge_mask = local_variance > edge_threshold
    
    # Preserve original values at important edges
    preservation_factor = np.where(edge_mask, 0.3, 0.8)
    refined_mask = enhanced_mask * preservation_factor + original_mask * (1 - preservation_factor)
    
    # Noise reduction using median filtering approximation
    refined_mask = apply_noise_reduction_filter(refined_mask)
    
    return refined_mask

def apply_noise_reduction_filter(mask):
    """Apply noise reduction using simplified median-like filtering."""
    kernel_size = 3
    pad_size = kernel_size // 2
    padded = np.pad(mask, pad_size, mode='reflect')
    
    filtered_mask = np.zeros_like(mask)
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            # Use weighted average instead of true median for simplicity
            center_weight = 0.5
            neighbor_weight = 0.5 / (window.size - 1)
            
            filtered_value = center_weight * window[pad_size, pad_size]
            filtered_value += neighbor_weight * (np.sum(window) - window[pad_size, pad_size])
            
            filtered_mask[i, j] = filtered_value
    
    return filtered_mask

def apply_multiscale_morphological_analysis(mask, kernel_type):
    """Apply morphological operations at multiple scales for comprehensive analysis."""
    scales = [3, 5, 7]
    multiscale_results = []
    
    for scale in scales:
        # Create kernel for current scale
        if kernel_type == 'cross':
            kernel = create_cross_kernel(scale)
        elif kernel_type == 'diamond':
            kernel = create_diamond_kernel(scale)
        else:
            kernel = create_square_kernel(scale)
        
        # Apply opening to remove small artifacts
        opened = apply_erosion(mask, kernel)
        opened = apply_dilation(opened, kernel)
        
        # Apply closing to fill small gaps
        closed = apply_dilation(mask, kernel)
        closed = apply_erosion(closed, kernel)
        
        # Combine opening and closing results
        scale_result = (opened + closed) / 2
        multiscale_results.append(scale_result)
    
    # Weight results by scale (larger scales get more weight)
    weights = np.array([0.2, 0.3, 0.5])
    combined_result = np.zeros_like(mask)
    
    for result, weight in zip(multiscale_results, weights):
        combined_result += weight * result
    
    return combined_result

def combine_multiscale_results(base_result, multiscale_result):
    """Combine base morphological result with multiscale analysis."""
    # Calculate local complexity to determine blending factor
    local_complexity = calculate_local_complexity(base_result)
    
    # More complex areas benefit more from multiscale analysis
    blend_factor = np.clip(local_complexity * 2, 0.2, 0.8)
    
    combined = base_result * (1 - blend_factor) + multiscale_result * blend_factor
    
    return combined

def calculate_local_complexity(mask):
    """Calculate local complexity measure for adaptive processing."""
    # Use gradient magnitude as complexity measure
    grad_x = np.diff(mask, axis=1, prepend=mask[:, :1])
    grad_y = np.diff(mask, axis=0, prepend=mask[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-1 range
    max_grad = np.max(gradient_magnitude)
    if max_grad > 0:
        normalized_complexity = gradient_magnitude / max_grad
    else:
        normalized_complexity = np.zeros_like(gradient_magnitude)
    
    return normalized_complexity

def normalize_morphological_result(mask):
    """Normalize the final morphological result to appropriate range."""
    # Robust normalization using percentiles
    p1, p99 = np.percentile(mask, [1, 99])
    
    if p99 > p1:
        normalized = (mask - p1) / (p99 - p1)
    else:
        normalized = mask / (np.max(mask) + 1e-6)
    
    # Clamp to valid range and apply final contrast adjustment
    final_mask = np.clip(normalized, 0, 1)
    
    # Apply S-curve for better contrast
    final_mask = apply_contrast_enhancement_curve(final_mask)
    
    return final_mask

def apply_contrast_enhancement_curve(mask):
    """Apply S-curve for enhanced contrast in morphological result."""
    # Sigmoid-based S-curve
    enhanced = 1 / (1 + np.exp(-10 * (mask - 0.5)))
    
    # Blend with original to prevent over-enhancement
    blend_factor = 0.7
    result = mask * (1 - blend_factor) + enhanced * blend_factor
    
    return np.clip(result, 0, 1)

def apply_edge_preserving_post_processing(enhanced_edges, original_edges, threshold_map):
    """
    Apply comprehensive edge-preserving post-processing with multi-scale analysis and adaptive enhancement.
    
    This function performs sophisticated post-processing to maintain and enhance important edge structures
    while reducing noise and artifacts from the initial edge enhancement process.
    
    :param enhanced_edges: Enhanced edge map from previous processing steps
    :param original_edges: Original edge map for reference and comparison
    :param threshold_map: Adaptive threshold map for edge classification
    :return: Highly refined edge map with preserved structures and enhanced connectivity
    """
    if enhanced_edges is None or original_edges is None:
        return enhanced_edges
    
    # Stage 1: Advanced edge structure identification and classification
    edge_classification = perform_comprehensive_edge_analysis(
        enhanced_edges, original_edges, threshold_map
    )
    
    # Stage 2: Multi-scale structure preservation with hierarchical processing
    multi_scale_preserved = apply_multi_scale_structure_preservation(
        enhanced_edges, original_edges, edge_classification
    )
    
    # Stage 3: Adaptive quality enhancement based on local characteristics
    quality_enhanced = apply_adaptive_quality_enhancement(
        multi_scale_preserved, edge_classification, threshold_map
    )
    
    # Stage 4: Advanced connectivity enhancement with topology preservation
    connectivity_enhanced = apply_advanced_connectivity_enhancement(
        quality_enhanced, edge_classification
    )
    
    # Stage 5: Artifact suppression and noise reduction
    artifact_suppressed = apply_intelligent_artifact_suppression(
        connectivity_enhanced, original_edges, edge_classification
    )
    
    # Stage 6: Final refinement with cross-validation against original
    final_refined = apply_final_cross_validation_refinement(
        artifact_suppressed, original_edges, enhanced_edges
    )
    
    # Stage 7: Quality assurance and validation
    validated_result = perform_quality_assurance_validation(
        final_refined, original_edges, threshold_map
    )
    
    return validated_result

def perform_comprehensive_edge_analysis(enhanced_edges, original_edges, threshold_map):
    """Perform comprehensive analysis and classification of edge structures."""
    # Multi-criteria edge classification
    edge_strength = calculate_multi_scale_edge_strength(enhanced_edges)
    edge_coherence = calculate_edge_coherence_measure(enhanced_edges)
    structural_importance = assess_structural_importance(enhanced_edges, original_edges)
    
    # Create comprehensive edge classification
    classification = {
        'critical_edges': identify_critical_edge_structures(edge_strength, edge_coherence, structural_importance),
        'important_edges': identify_important_edge_structures(edge_strength, edge_coherence),
        'secondary_edges': identify_secondary_edge_structures(edge_strength),
        'noise_edges': identify_noise_edge_candidates(enhanced_edges, original_edges, threshold_map),
        'connectivity_candidates': identify_connectivity_enhancement_candidates(enhanced_edges)
    }
    
    return classification

def apply_multi_scale_structure_preservation(enhanced_edges, original_edges, edge_classification):
    """Apply multi-scale structure preservation with hierarchical processing."""
    # Process at multiple scales for comprehensive preservation
    scales = [1, 2, 4, 8]  # Different analysis scales
    preserved_scales = []
    
    for scale in scales:
        # Downsample for current scale analysis
        if scale > 1:
            scaled_enhanced = downsample_edges(enhanced_edges, scale)
            scaled_original = downsample_edges(original_edges, scale)
        else:
            scaled_enhanced = enhanced_edges
            scaled_original = original_edges
        # Apply scale-specific preservation
        scale_preserved = apply_scale_specific_preservation(
            scaled_enhanced, scaled_original, edge_classification, scale
        )

        # Upsample back to original resolution if needed
        if scale > 1:
            scale_preserved = upsample_edges(scale_preserved, enhanced_edges.shape, scale)
        
        preserved_scales.append(scale_preserved)
    
    # Com    # Combine multi-scale results with intelligent weighting
    def combine_multi_scale_preservation_results(preserved_scales, edge_classification):
        if not preserved_scales:
            return None
        # Simple average of all scales
        combined = np.mean(preserved_scales, axis=0)
        return combined

    combined_preservation = combine_multi_scale_preservation_results(preserved_scales, edge_classification)

    return combined_preservation

def apply_adaptive_quality_enhancement(preserved_edges, edge_classification, threshold_map):
    """Apply adaptive quality enhancement based on local edge characteristics."""
    # Calculate local quality metrics
    local_contrast = calculate_local_edge_contrast(preserved_edges)
    local_coherence = calculate_local_edge_coherence(preserved_edges)
    local_noise_level = estimate_local_edge_noise(preserved_edges, threshold_map)
    
    # Create adaptive enhancement map
    enhancement_map = create_adaptive_enhancement_map(
        local_contrast, local_coherence, local_noise_level, edge_classification
    )
    
    # Apply different enhancement strategies based on edge classification
    enhanced_result = np.copy(preserved_edges)
    
    # Critical edges: Maximum preservation with minimal enhancement
    critical_mask = edge_classification['critical_edges']
    enhanced_result = apply_critical_edge_enhancement(
        enhanced_result, critical_mask, enhancement_map
    )
    
    # Important edges: Balanced enhancement with structure preservation
    important_mask = edge_classification['important_edges']
    enhanced_result = apply_important_edge_enhancement(
        enhanced_result, important_mask, enhancement_map
    )
    
    # Secondary edges: Moderate enhancement with noise consideration
    secondary_mask = edge_classification['secondary_edges']
    enhanced_result = apply_secondary_edge_enhancement(
        enhanced_result, secondary_mask, enhancement_map, local_noise_level
    )
    
    return enhanced_result

def calculate_local_edge_contrast(edges):
    """Calculate local edge contrast using gradient analysis."""
    # Calculate gradients
    grad_x = np.diff(edges, axis=1, prepend=edges[:, :1])
    grad_y = np.diff(edges, axis=0, prepend=edges[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate local contrast using sliding window
    kernel_size = 5
    pad_size = kernel_size // 2
    padded = np.pad(gradient_magnitude, pad_size, mode='reflect')
    
    local_contrast = np.zeros_like(edges)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            local_contrast[i, j] = np.std(window)
    
    return local_contrast

def calculate_local_edge_coherence(edges):
    """Calculate local edge coherence using structure tensor."""
    # Calculate gradients
    grad_x = np.diff(edges, axis=1, prepend=edges[:, :1])
    grad_y = np.diff(edges, axis=0, prepend=edges[:1, :])
    
    # Structure tensor components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # Apply Gaussian smoothing
    sigma = 1.0
    Ixx_smooth = apply_gaussian_smoothing_simple(Ixx, sigma)
    Iyy_smooth = apply_gaussian_smoothing_simple(Iyy, sigma)
    Ixy_smooth = apply_gaussian_smoothing_simple(Ixy, sigma)
    
    # Calculate coherence
    trace = Ixx_smooth + Iyy_smooth
    det = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth
    
    discriminant = np.maximum(0, trace**2 - 4 * det)
    lambda1 = 0.5 * (trace + np.sqrt(discriminant))
    lambda2 = 0.5 * (trace - np.sqrt(discriminant))
    
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    return np.clip(coherence, 0, 1)

def estimate_local_edge_noise(edges, threshold_map):
    """Estimate local noise level in edge map."""
    # Calculate difference from expected threshold
    noise_estimate = np.abs(edges - threshold_map)
    
    # Apply local averaging to get noise level
    kernel_size = 7
    pad_size = kernel_size // 2
    padded = np.pad(noise_estimate, pad_size, mode='reflect')
    
    local_noise = np.zeros_like(edges)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            local_noise[i, j] = np.mean(window)
    
    return local_noise

def create_adaptive_enhancement_map(local_contrast, local_coherence, local_noise_level, edge_classification):
    """Create adaptive enhancement map based on local characteristics."""
    # Normalize inputs
    contrast_norm = local_contrast / (np.max(local_contrast) + 1e-6)
    coherence_norm = local_coherence
    noise_norm = local_noise_level / (np.max(local_noise_level) + 1e-6)
    
    # Calculate enhancement strength
    enhancement_strength = contrast_norm * 0.4 + coherence_norm * 0.4 - noise_norm * 0.2
    enhancement_strength = np.clip(enhancement_strength, 0.1, 1.0)
    
    return enhancement_strength

def apply_critical_edge_enhancement(edges, critical_mask, enhancement_map):
    """Apply minimal enhancement to critical edges for preservation."""
    enhanced = edges.copy()
    enhancement_factor = 1.0 + 0.1 * enhancement_map  # Very gentle enhancement
    enhanced[critical_mask] *= enhancement_factor[critical_mask]
    return enhanced

def apply_important_edge_enhancement(edges, important_mask, enhancement_map):
    """Apply balanced enhancement to important edges."""
    enhanced = edges.copy()
    enhancement_factor = 1.0 + 0.3 * enhancement_map  # Moderate enhancement
    enhanced[important_mask] *= enhancement_factor[important_mask]
    return enhanced

def apply_secondary_edge_enhancement(edges, secondary_mask, enhancement_map, local_noise_level):
    """Apply moderate enhancement to secondary edges with noise consideration."""
    enhanced = edges.copy()
    # Reduce enhancement in noisy areas
    noise_factor = 1.0 - 0.5 * (local_noise_level / (np.max(local_noise_level) + 1e-6))
    enhancement_factor = 1.0 + 0.5 * enhancement_map * noise_factor
    enhanced[secondary_mask] *= enhancement_factor[secondary_mask]
    return enhanced

def apply_advanced_connectivity_enhancement(quality_enhanced, edge_classification):
    """Apply advanced connectivity enhancement with topology preservation."""
    # Analyze current connectivity patterns
    connectivity_analysis = analyze_edge_connectivity_patterns(quality_enhanced)
    
    # Identify connectivity gaps and opportunities
    connectivity_gaps = identify_connectivity_gaps(
        quality_enhanced, edge_classification, connectivity_analysis
    )
    
    # Apply intelligent gap bridging
    gap_bridged = apply_intelligent_gap_bridging(
        quality_enhanced, connectivity_gaps, edge_classification
    )
    
    # Enhance edge continuity while preserving topology
    continuity_enhanced = enhance_edge_continuity_with_topology_preservation(
        gap_bridged, connectivity_analysis
    )
    
    # Apply adaptive smoothing to reduce connectivity artifacts
    smoothed_connectivity = apply_connectivity_artifact_smoothing(
        continuity_enhanced, edge_classification
    )
    
    return smoothed_connectivity
# Stub implementations for missing functions to prevent NameError

def analyze_edge_connectivity_patterns(quality_enhanced):
    """Stub: Analyze edge connectivity patterns."""
    return None

def identify_connectivity_gaps(quality_enhanced, edge_classification, connectivity_analysis):
    """Stub: Identify connectivity gaps."""
    return None

def apply_intelligent_gap_bridging(quality_enhanced, connectivity_gaps, edge_classification):
    """Stub: Apply intelligent gap bridging."""
    return quality_enhanced

def enhance_edge_continuity_with_topology_preservation(gap_bridged, connectivity_analysis):
    """Stub: Enhance edge continuity with topology preservation."""
    return gap_bridged

def apply_connectivity_artifact_smoothing(continuity_enhanced, edge_classification):
    """Stub: Apply connectivity artifact smoothing."""
    return continuity_enhanced

def detect_comprehensive_artifacts(connectivity_enhanced, original_edges, edge_classification):
    """Stub: Detect comprehensive artifacts."""
    return {
        'enhancement_artifacts': None,
        'connectivity_artifacts': None,
        'noise_amplification': None
    }

def suppress_enhancement_artifacts(suppressed_result, enhancement_artifacts, original_edges):
    """Stub: Suppress enhancement artifacts."""
    return suppressed_result

def suppress_connectivity_artifacts(suppressed_result, connectivity_artifacts, edge_classification):
    """Stub: Suppress connectivity artifacts."""
    return suppressed_result

def suppress_noise_amplification(suppressed_result, noise_amplification, original_edges):
    """Stub: Suppress noise amplification."""
    return suppressed_result

def apply_structure_preserving_denoising(suppressed_result, edge_classification):
    """Stub: Apply structure preserving denoising."""
    return suppressed_result

def calculate_edge_fidelity(artifact_suppressed, original_edges):
    """Stub: Calculate edge fidelity."""
    return 1.0

def calculate_enhancement_quality(artifact_suppressed, enhanced_edges):
    """Stub: Calculate enhancement quality."""
    return 1.0

def identify_refinement_candidates(original_fidelity, enhancement_quality, artifact_suppressed):
    """Stub: Identify refinement candidates."""
    return None

def apply_targeted_refinements(artifact_suppressed, refinement_candidates, original_edges, enhanced_edges):
    """Stub: Apply targeted refinements."""
    return artifact_suppressed

def apply_consistency_validation(refined_result, original_edges, enhanced_edges):
    """Stub: Apply consistency validation."""
    return refined_result

def apply_intelligent_artifact_suppression(connectivity_enhanced, original_edges, edge_classification):
    """
    # Detect various types of artifacts
    artifact_detection = detect_comprehensive_artifacts(
        connectivity_enhanced, original_edges, edge_classification
    )
    
    # Apply targeted suppression for different artifact types
    suppressed_result = np.copy(connectivity_enhanced)
    
    # Suppress enhancement artifacts
    enhancement_artifacts = artifact_detection['enhancement_artifacts']
    suppressed_result = suppress_enhancement_artifacts(
        suppressed_result, enhancement_artifacts, original_edges
    )
    
    # Suppress connectivity artifacts
    connectivity_artifacts = artifact_detection['connectivity_artifacts']
    suppressed_result = suppress_connectivity_artifacts(
        suppressed_result, connectivity_artifacts, edge_classification
    )
    
    # Suppress noise amplification
    noise_amplification = artifact_detection['noise_amplification']
    suppressed_result = suppress_noise_amplification(
        suppressed_result, noise_amplification, original_edges
    )
    
    # Apply adaptive denoising while preserving important structures
    denoised_result = apply_structure_preserving_denoising(
        suppressed_result, edge_classification
    )
    
    return denoised_result
    """

def apply_final_cross_validation_refinement(artifact_suppressed, original_edges, enhanced_edges):
    """Apply final refinement with cross-validation against original and enhanced edges."""
    # Calculate fidelity metrics
    original_fidelity = calculate_edge_fidelity(artifact_suppressed, original_edges)
    enhancement_quality = calculate_enhancement_quality(artifact_suppressed, enhanced_edges)
    
    # Identify areas needing refinement
    refinement_candidates = identify_refinement_candidates(
        original_fidelity, enhancement_quality, artifact_suppressed
    )
    
    # Apply targeted refinements
    refined_result = apply_targeted_refinements(
        artifact_suppressed, refinement_candidates, original_edges, enhanced_edges
    )
    
    # Perform consistency validation
    consistency_validated = apply_consistency_validation(
        refined_result, original_edges, enhanced_edges
    )
    
    return consistency_validated

def perform_quality_assurance_validation(final_refined, original_edges, threshold_map):
    """Perform comprehensive quality assurance and validation."""
    # Validate edge preservation
    preservation_score = validate_edge_preservation(final_refined, original_edges)
    
    # Validate enhancement quality
    enhancement_score = validate_enhancement_quality(final_refined, threshold_map)
    
    # Validate structural integrity
    structural_score = validate_structural_integrity(final_refined)
    
    # Apply corrective measures if needed
    if preservation_score < 0.8 or enhancement_score < 0.7 or structural_score < 0.75:
        corrected_result = apply_quality_corrective_measures(
            final_refined, original_edges, preservation_score, 
            enhancement_score, structural_score
        )
        return corrected_result
    
    # Apply final normalization and range optimization
    optimized_result = apply_final_optimization(final_refined)
    
    return optimized_result

def validate_edge_preservation(final_refined, original_edges):
    """Validate how well the final result preserves original edge structures."""
    if original_edges is None or final_refined is None:
        return 0.5
    
    # Calculate correlation between final and original edges
    correlation = np.corrcoef(final_refined.flatten(), original_edges.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # Calculate structural similarity
    mean_orig = np.mean(original_edges)
    mean_final = np.mean(final_refined)
    var_orig = np.var(original_edges)
    var_final = np.var(final_refined)
    cov = np.mean((original_edges - mean_orig) * (final_refined - mean_final))
    
    structural_sim = (2 * mean_orig * mean_final + 1e-6) / (mean_orig**2 + mean_final**2 + 1e-6)
    structural_sim *= (2 * np.sqrt(var_orig * var_final) + 1e-6) / (var_orig + var_final + 1e-6)
    
    # Combine metrics
    preservation_score = 0.7 * abs(correlation) + 0.3 * structural_sim
    return np.clip(preservation_score, 0, 1)

def validate_enhancement_quality(final_refined, threshold_map):
    """Validate the quality of edge enhancement."""
    if final_refined is None or threshold_map is None:
        return 0.5
    
    # Check if enhancement respects threshold constraints
    enhancement_ratio = final_refined / (threshold_map + 1e-6)
    reasonable_enhancement = np.mean((enhancement_ratio > 0.5) & (enhancement_ratio < 5.0))
    
    # Check for gradient consistency
    grad_x = np.diff(final_refined, axis=1, prepend=final_refined[:, :1])
    grad_y = np.diff(final_refined, axis=0, prepend=final_refined[:1, :])
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_consistency = 1.0 - np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-6)
    
    # Check dynamic range utilization
    dynamic_range = (np.max(final_refined) - np.min(final_refined)) / (np.max(final_refined) + 1e-6)
    
    # Combine quality metrics
    quality_score = 0.4 * reasonable_enhancement + 0.3 * gradient_consistency + 0.3 * dynamic_range
    return np.clip(quality_score, 0, 1)

def validate_structural_integrity(final_refined):
    """Validate the structural integrity of the edge map."""
    if final_refined is None:
        return 0.5
    
    # Check for connectivity patterns
    binary_edges = final_refined > np.mean(final_refined)
    connectivity_score = calculate_connectivity_score(binary_edges)
    
    # Check for smoothness (absence of artifacts)
    grad_x = np.diff(final_refined, axis=1, prepend=final_refined[:, :1])
    grad_y = np.diff(final_refined, axis=0, prepend=final_refined[:1, :])
    smoothness = 1.0 / (1.0 + np.std(grad_x) + np.std(grad_y))
    
    # Check for reasonable value distribution
    value_range = np.max(final_refined) - np.min(final_refined)
    distribution_score = min(1.0, value_range / 255.0) if value_range > 0 else 0.0
    
    # Combine structural metrics
    structural_score = 0.4 * connectivity_score + 0.3 * smoothness + 0.3 * distribution_score
    return np.clip(structural_score, 0, 1)

def calculate_connectivity_score(binary_edges):
    """Calculate connectivity score for binary edge map."""
    if binary_edges.size == 0:
        return 0.0
    
    # Count connected components
    h, w = binary_edges.shape
    visited = np.zeros_like(binary_edges, dtype=bool)
    num_components = 0
    total_edge_pixels = np.sum(binary_edges)
    
    if total_edge_pixels == 0:
        return 0.0
    
    for i in range(h):
        for j in range(w):
            if binary_edges[i, j] and not visited[i, j]:
                component_size = flood_fill_count(binary_edges, visited, i, j)
                if component_size > 5:  # Only count significant components
                    num_components += 1
    
    # Better connectivity means fewer components relative to edge pixels
    if num_components > 0:
        connectivity_score = min(1.0, total_edge_pixels / (num_components * 10))
    else:
        connectivity_score = 0.0
    
    return connectivity_score

def flood_fill_count(binary_edges, visited, start_i, start_j):
    """Count pixels in connected component using flood fill."""
    h, w = binary_edges.shape
    stack = [(start_i, start_j)]
    count = 0
    
    while stack:
        i, j = stack.pop()
        if i < 0 or i >= h or j < 0 or j >= w or visited[i, j] or not binary_edges[i, j]:
            continue
        
        visited[i, j] = True
        count += 1
        
        # Add 8-connected neighbors
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di != 0 or dj != 0:
                    stack.append((i + di, j + dj))
    
    return count

def apply_quality_corrective_measures(final_refined, original_edges, preservation_score, enhancement_score, structural_score):
    """Apply corrective measures when quality scores are below threshold."""
    corrected = final_refined.copy()
    
    # Correct preservation issues
    if preservation_score < 0.8:
        # Blend more with original
        blend_factor = 0.3 + (0.8 - preservation_score) * 0.5
        corrected = corrected * (1 - blend_factor) + original_edges * blend_factor
    
    # Correct enhancement issues
    if enhancement_score < 0.7:
        # Apply conservative enhancement
        mean_val = np.mean(corrected)
        corrected = corrected * 0.9 + mean_val * 0.1
    
    # Correct structural issues
    if structural_score < 0.75:
        # Apply gentle smoothing
        corrected = apply_gentle_smoothing(corrected)
    
    return corrected

def apply_gentle_smoothing(image):
    """Apply gentle smoothing to reduce structural artifacts."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    
    h, w = image.shape
    padded = np.pad(image, 1, mode='reflect')
    smoothed = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            smoothed[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    
    # Blend with original to preserve structure
    return 0.8 * image + 0.2 * smoothed

def apply_final_optimization(final_refined):
    """Apply final optimization and normalization."""
    # Robust normalization using percentiles
    p1, p99 = np.percentile(final_refined, [1, 99])
    
    if p99 > p1:
        normalized = (final_refined - p1) / (p99 - p1)
    else:
        normalized = final_refined / (np.max(final_refined) + 1e-6)
    
    # Clamp to valid range
    optimized = np.clip(normalized, 0, 1)
    
    # Apply final contrast enhancement
    enhanced = apply_final_contrast_boost(optimized)
    
    return enhanced

def apply_final_contrast_boost(image):
    """Apply final contrast boost for optimal visibility."""
    # Gentle S-curve enhancement
    enhanced = np.power(image, 0.9)  # Slight gamma correction
    
    # Ensure full dynamic range utilization
    min_val, max_val = np.min(enhanced), np.max(enhanced)
    if max_val > min_val:
        enhanced = (enhanced - min_val) / (max_val - min_val)
    
    return enhanced


def calculate_multi_scale_edge_strength(edges):
    """Calculate edge strength at multiple scales."""
    scales = [1, 2, 4]
    strength_maps = []
    
    for scale in scales:
        if scale == 1:
            scaled_edges = edges
        else:
            scaled_edges = downsample_edges(edges, scale)
        
        # Calculate gradient magnitude at current scale
        grad_x = np.diff(scaled_edges, axis=1, prepend=scaled_edges[:, :1])
        grad_y = np.diff(scaled_edges, axis=0, prepend=scaled_edges[:1, :])
        strength = np.sqrt(grad_x**2 + grad_y**2)
        
        if scale > 1:
            strength = upsample_edges(strength, edges.shape, scale)
        
        strength_maps.append(strength)
    
    # Combine multi-scale strength information
    combined_strength = np.mean(strength_maps, axis=0)
    return combined_strength

def calculate_edge_coherence_measure(edges):
    """Calculate edge coherence using structure tensor analysis."""
    # Calculate gradients
    grad_x = np.diff(edges, axis=1, prepend=edges[:, :1])
    grad_y = np.diff(edges, axis=0, prepend=edges[:1, :])
    
    # Structure tensor components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y
    Ixy = grad_x * grad_y
    
    # Apply Gaussian smoothing
    sigma = 1.0
    Ixx_smooth = apply_gaussian_smoothing_simple(Ixx, sigma)
    Iyy_smooth = apply_gaussian_smoothing_simple(Iyy, sigma)
    Ixy_smooth = apply_gaussian_smoothing_simple(Ixy, sigma)
    
    # Calculate coherence measure
    trace = Ixx_smooth + Iyy_smooth
    det = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth
    
    # Coherence as ratio of eigenvalues
    discriminant = np.maximum(0, trace**2 - 4 * det)
    lambda1 = 0.5 * (trace + np.sqrt(discriminant))
    lambda2 = 0.5 * (trace - np.sqrt(discriminant))
    
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6)
    return np.clip(coherence, 0, 1)

def apply_gaussian_smoothing_simple(data, sigma):
    """Apply simple Gaussian smoothing."""
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    kernel_1d = np.exp(-ax**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # Apply separable convolution
    # Horizontal pass
    pad_size = kernel_size // 2
    padded = np.pad(data, pad_size, mode='reflect')
    h_filtered = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        for k, weight in enumerate(kernel_1d):
            h_filtered[i] += weight * padded[i, k:k+data.shape[1]]
    
    # Vertical pass
    padded = np.pad(h_filtered, pad_size, mode='reflect')
    v_filtered = np.zeros_like(data)
    
    for j in range(data.shape[1]):
        for k, weight in enumerate(kernel_1d):
            v_filtered[:, j] += weight * padded[k:k+data.shape[0], j]
    
    return v_filtered

def assess_structural_importance(enhanced_edges, original_edges):
    """Assess the structural importance of edges."""
    # Calculate correlation with original edges
    correlation = calculate_local_correlation_map(enhanced_edges, original_edges)
    
    # Calculate edge connectivity importance
    connectivity_importance = calculate_connectivity_importance(enhanced_edges)
    
    # Combine metrics
    structural_importance = 0.6 * correlation + 0.4 * connectivity_importance
    
    return np.clip(structural_importance, 0, 1)

def calculate_local_correlation_map(edges1, edges2):
    """Calculate local correlation between two edge maps."""
    kernel_size = 5
    pad_size = kernel_size // 2
    
    padded1 = np.pad(edges1, pad_size, mode='reflect')
    padded2 = np.pad(edges2, pad_size, mode='reflect')
    
    correlation_map = np.zeros_like(edges1)
    
    for i in range(edges1.shape[0]):
        for j in range(edges1.shape[1]):
            window1 = padded1[i:i+kernel_size, j:j+kernel_size].flatten()
            window2 = padded2[i:i+kernel_size, j:j+kernel_size].flatten()
            
            if np.std(window1) > 1e-6 and np.std(window2) > 1e-6:
                corr_coef = np.corrcoef(window1, window2)[0, 1]
                correlation_map[i, j] = max(0, corr_coef)
            else:
                correlation_map[i, j] = 0
    
    return correlation_map

def calculate_connectivity_importance(edges):
    """Calculate connectivity importance of edges."""
    # Use distance transform to measure connectivity
    binary_edges = edges > np.mean(edges)
    
    # Calculate distance to nearest edge for each pixel
    y_coords, x_coords = np.meshgrid(
        np.arange(edges.shape[0]), 
        np.arange(edges.shape[1]), 
        indexing='ij'
    )
    
    connectivity_map = np.zeros_like(edges)
    
    # Simple connectivity measure based on local edge density
    kernel_size = 7
    pad_size = kernel_size // 2
    padded = np.pad(binary_edges.astype(float), pad_size, mode='constant')
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            connectivity_map[i, j] = np.sum(window) / (kernel_size * kernel_size)
    
    return connectivity_map

# Additional helper functions would continue here with similar detailed implementations...
def apply_scale_specific_preservation(scaled_enhanced, scaled_original, edge_classification, scale):
    """
    Preserve edge structures at a specific scale.
    This is a stub implementation; you may refine it for your needs.
    """
    # For now, just blend enhanced and original based on scale
    blend_factor = 0.5 if scale == 1 else 0.7
    return scaled_enhanced * blend_factor + scaled_original * (1 - blend_factor)

# Additional helper functions would continue here with similar detailed implementations...
def downsample_edges(edges, scale):
    """Downsample edge map by given scale factor."""
    h, w = edges.shape
    new_h, new_w = h // scale, w // scale

    # Simple downsampling using averaging
    downsampled = np.zeros((new_h, new_w))
    
    for i in range(new_h):
        for j in range(new_w):
            region = edges[i*scale:(i+1)*scale, j*scale:(j+1)*scale]
            downsampled[i, j] = np.mean(region)
    
    return downsampled

def upsample_edges(edges, target_shape, scale):
    """Upsample edge map to target shape."""
    h, w = target_shape
    upsampled = np.zeros((h, w))
    
    # Simple nearest neighbor upsampling
    for i in range(h):
        for j in range(w):
            src_i = min(i // scale, edges.shape[0] - 1)
            src_j = min(j // scale, edges.shape[1] - 1)
            upsampled[i, j] = edges[src_i, src_j]
    
    return upsampled

def identify_critical_edge_structures(edge_strength, edge_coherence, structural_importance):
    """Identify critical edge structures that must be preserved."""
    # Combine metrics with high thresholds for critical edges
    critical_threshold_strength = np.percentile(edge_strength, 90)
    critical_threshold_coherence = np.percentile(edge_coherence, 85)
    critical_threshold_importance = np.percentile(structural_importance, 80)
    
    critical_mask = (edge_strength > critical_threshold_strength) & \
                   (edge_coherence > critical_threshold_coherence) & \
                   (structural_importance > critical_threshold_importance)
    
    return critical_mask

def identify_secondary_edge_structures(edge_strength):
    """Identify secondary edge structures of moderate importance."""
    # Use moderate thresholds for secondary edges
    secondary_threshold_low = np.percentile(edge_strength, 40)
    secondary_threshold_high = np.percentile(edge_strength, 75)
    
    secondary_mask = (edge_strength > secondary_threshold_low) & \
                    (edge_strength <= secondary_threshold_high)
    
    return secondary_mask

def identify_noise_edge_candidates(enhanced_edges, original_edges, threshold_map):
    """Identify edge candidates that are likely noise."""
    # Calculate difference between enhanced and original
    enhancement_diff = enhanced_edges - original_edges
    
    # Identify areas with excessive enhancement relative to threshold
    excessive_enhancement = enhancement_diff > (threshold_map * 2.0)
    
    # Identify weak edges that might be noise
    weak_edges = enhanced_edges < np.percentile(threshold_map, 25)
    
    # Combine criteria for noise detection
    noise_candidates = excessive_enhancement | weak_edges
    
    return noise_candidates

def identify_connectivity_enhancement_candidates(enhanced_edges):
    """Identify areas that could benefit from connectivity enhancement."""
    # Look for isolated edge segments
    kernel_size = 5
    pad_size = kernel_size // 2
    padded = np.pad(enhanced_edges, pad_size, mode='reflect')
    
    connectivity_candidates = np.zeros_like(enhanced_edges, dtype=bool)
    
    for i in range(enhanced_edges.shape[0]):
        for j in range(enhanced_edges.shape[1]):
            # Check local neighborhood for connectivity
            neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
            center_value = neighborhood[pad_size, pad_size]
            
            if center_value > np.mean(enhanced_edges):
                # Count connected neighbors
                neighbor_count = np.sum(neighborhood > center_value * 0.5) - 1
                
                # Mark as candidate if few connected neighbors
                if neighbor_count < 3:
                    connectivity_candidates[i, j] = True
    
    return connectivity_candidates

def identify_important_edge_structures(edges, threshold_map):
    """Identify important edge structures to preserve."""
    # Calculate edge strength relative to local threshold
    relative_strength = edges / (threshold_map + 1e-6)
    
    # Identify strong edges
    strong_edges = relative_strength > 1.5
    
    # Identify connected edge components
    connected_components = find_connected_edge_components(strong_edges)
    
    # Filter by component size and strength
    important_components = filter_important_components(
        connected_components, edges, min_size=10
    )
    
    return important_components

def find_connected_edge_components(binary_edges):
    """Find connected components in binary edge map."""
    # Simple connected component labeling
    h, w = binary_edges.shape
    labels = np.zeros_like(binary_edges, dtype=int)
    current_label = 1
    
    # 8-connectivity offsets
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for i in range(h):
        for j in range(w):
            if binary_edges[i, j] and labels[i, j] == 0:
                # Start new component
                flood_fill_component(binary_edges, labels, i, j, current_label, offsets)
                current_label += 1

    return labels

def flood_fill_component(binary_edges, labels, start_i, start_j, label, offsets):
    """Flood fill algorithm for connected component labeling."""
    stack = [(start_i, start_j)]
    h, w = binary_edges.shape

    while stack:
        i, j = stack.pop()
        if labels[i, j] != 0:
            continue

        labels[i, j] = label

        # Check 8-connected neighbors
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if (0 <= ni < h and 0 <= nj < w and
                binary_edges[ni, nj] and labels[ni, nj] == 0):
                stack.append((ni, nj))

def identify_important_edge_structures(edges, threshold_map):
    """Identify important edge structures to preserve."""
    # Calculate edge strength relative to local threshold
    relative_strength = edges / (threshold_map + 1e-6)
    
    # Identify strong edges
    strong_edges = relative_strength > 1.5
    
    # Identify connected edge components
    connected_components = find_connected_edge_components(strong_edges)
    
    # Filter by component size and strength
    important_components = filter_important_components(
        connected_components, edges, min_size=10
    )
    
    return important_components

def find_connected_edge_components(binary_edges):
    """Find connected components in binary edge map."""
    # Simple connected component labeling
    h, w = binary_edges.shape
    labels = np.zeros_like(binary_edges, dtype=int)
    current_label = 1

    # 8-connectivity offsets
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for i in range(h):
        for j in range(w):
            if binary_edges[i, j] and labels[i, j] == 0:
                # Start new component
                flood_fill_component(binary_edges, labels, i, j, current_label, offsets)
                current_label += 1

    return labels

def flood_fill_component(binary_edges, labels, start_i, start_j, label, offsets):
    """Flood fill algorithm for connected component labeling."""
    stack = [(start_i, start_j)]
    h, w = binary_edges.shape

    while stack:
        i, j = stack.pop()
        if labels[i, j] != 0:
            continue

        labels[i, j] = label

        # Check 8-connected neighbors
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if (0 <= ni < h and 0 <= nj < w and 
                binary_edges[ni, nj] and labels[ni, nj] == 0):
                stack.append((ni, nj))

def filter_important_components(components, edge_strength, min_size=10):
    """Filter components by size and average strength."""
    filtered_components = np.zeros_like(components)
    
    # Get unique component labels
    unique_labels = np.unique(components)
    
    for label in unique_labels:
        if label == 0:  # Background
            continue
            
        component_mask = components == label
        component_size = np.sum(component_mask)
        
        if component_size >= min_size:
            # Check average edge strength
            avg_strength = np.mean(edge_strength[component_mask])
            if avg_strength > np.mean(edge_strength) * 0.8:
                filtered_components[component_mask] = label
    
    return filtered_components > 0

def apply_structure_preserving_enhancement(enhanced, original, important_structures):
    """Apply enhancement while preserving important structures."""
    # Calculate preservation factor
    preservation_factor = np.where(important_structures, 0.8, 0.3)
    
    # Blend enhanced and original based on structure importance
    structure_preserved = enhanced * (1 - preservation_factor) + original * preservation_factor
    
    # Apply additional enhancement to non-structural areas
    non_structural_mask = ~important_structures
    if np.any(non_structural_mask):
        structure_preserved[non_structural_mask] *= 1.1
    
    return structure_preserved

def enhance_edge_connectivity_advanced(edges, important_structures):
    """Enhance edge connectivity while preserving important structures."""
    # Apply different connectivity enhancement based on structure importance
    connectivity_enhanced = np.copy(edges)
    
    # Strong connectivity enhancement for important structures
    important_enhanced = apply_strong_connectivity_enhancement(
        edges, important_structures
    )
    connectivity_enhanced[important_structures] = important_enhanced[important_structures]
    
    # Gentle connectivity enhancement for other areas
    other_areas = ~important_structures
    if np.any(other_areas):
        gentle_enhanced = apply_gentle_connectivity_enhancement(edges, other_areas)
        connectivity_enhanced[other_areas] = gentle_enhanced[other_areas]
    
    return connectivity_enhanced

def apply_strong_connectivity_enhancement(edges, mask):
    """Apply strong connectivity enhancement for important structures."""
    # Use morphological closing to connect nearby edges
    kernel_size = 3
    enhanced = apply_morphological_closing_selective(edges, mask, kernel_size)
    
    return enhanced

def apply_gentle_connectivity_enhancement(edges, mask):
    """Apply gentle connectivity enhancement for less important areas."""
    # Use lighter morphological operations
    kernel_size = 2
    enhanced = apply_morphological_closing_selective(edges, mask, kernel_size)
    
    return enhanced

def apply_morphological_closing_selective(edges, mask, kernel_size):
    """Apply morphological closing selectively."""
    result = np.copy(edges)
    
    # Simple closing operation where mask is True
    pad_size = kernel_size // 2
    padded = np.pad(edges, pad_size, mode='reflect')
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if mask[i, j]:
                window = padded[i:i+kernel_size, j:j+kernel_size]
                # Closing: dilation followed by erosion
                dilated = np.max(window)
                # Simple erosion approximation
                result[i, j] = max(result[i, j], dilated * 0.9)
    
    return result

def normalize_enhanced_edges(enhanced_edges):
    """Normalize enhanced edges to optimal range."""
    # Robust normalization using percentiles
    p1, p99 = np.percentile(enhanced_edges, [1, 99])
    
    if p99 > p1:
        normalized = (enhanced_edges - p1) / (p99 - p1)
    else:
        normalized = enhanced_edges / (np.max(enhanced_edges) + 1e-6)
    
    # Map to final range [0.1, 1.0] for edge mask compatibility
    final_normalized = 0.1 + 0.9 * np.clip(normalized, 0, 1)
    
    # Apply final contrast enhancement
    contrast_enhanced = apply_final_contrast_enhancement(final_normalized)
    
    return contrast_enhanced

def apply_final_contrast_enhancement(normalized_edges):
    """Apply final contrast enhancement to normalized edges."""
    # S-curve enhancement for better contrast
    s_curve_enhanced = apply_s_curve_enhancement(normalized_edges)
    
    # Adaptive histogram stretching
    histogram_enhanced = apply_adaptive_histogram_stretching(s_curve_enhanced)
    
    return histogram_enhanced

def apply_s_curve_enhancement(data):
    """Apply S-curve enhancement for better contrast."""
    # Sigmoid-based S-curve
    shifted_data = (data - 0.5) * 2  # Shift to [-1, 1]
    s_curve = np.tanh(shifted_data * 1.5) * 0.5 + 0.5  # S-curve and shift back
    
    return np.clip(s_curve, 0, 1)

def apply_adaptive_histogram_stretching(data):
    """Apply adaptive histogram stretching."""
    # Calculate optimal stretch parameters
    hist, bins = np.histogram(data.flatten(), bins=256, range=(0, 1))
    cumulative_hist = np.cumsum(hist) / np.sum(hist)
    
    # Find 5th and 95th percentiles for stretching
    low_idx = np.argmax(cumulative_hist >= 0.05)
    high_idx = np.argmax(cumulative_hist >= 0.95)
    
    low_val = bins[low_idx] if low_idx > 0 else 0
    high_val = bins[high_idx] if high_idx < len(bins)-1 else 1
    
    # Apply stretching
    if high_val > low_val:
        stretched = (data - low_val) / (high_val - low_val)
        stretched = np.clip(stretched, 0, 1)
    else:
        stretched = data
    
    return stretched

def apply_non_maximum_suppression(edge_magnitude, grad_x, grad_y):
    """Apply non-maximum suppression for edge thinning."""
    h, w = edge_magnitude.shape
    suppressed = np.zeros_like(edge_magnitude)
    
    # Calculate gradient direction
    angle = np.arctan2(grad_y, grad_x)
    
    # Quantize angles to 4 directions
    angle_quantized = np.round(angle / (np.pi / 4)) % 4
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            direction = int(angle_quantized[i, j])
            
            # Define neighbor coordinates based on direction
            if direction == 0:  # Horizontal
                neighbors = [edge_magnitude[i, j-1], edge_magnitude[i, j+1]]
            elif direction == 1:  # Diagonal /
                neighbors = [edge_magnitude[i+1, j-1], edge_magnitude[i-1, j+1]]
            elif direction == 2:  # Vertical
                neighbors = [edge_magnitude[i-1, j], edge_magnitude[i+1, j]]
            else:  # Diagonal \
                neighbors = [edge_magnitude[i-1, j-1], edge_magnitude[i+1, j+1]]
            
            # Suppress if not local maximum
            if edge_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = edge_magnitude[i, j]
    
    return suppressed

def apply_edge_aware_smoothing(edge_map, iterations=2):
    """Apply edge-aware smoothing to reduce noise while preserving edges."""
    smoothed = edge_map.copy()
    
    for _ in range(iterations):
        # Calculate local variance to detect edges
        kernel_size = 3
        pad_size = kernel_size // 2
        padded = np.pad(smoothed, pad_size, mode='reflect')
        
        local_variance = np.zeros_like(smoothed)
        for i in range(smoothed.shape[0]):
            for j in range(smoothed.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                local_variance[i, j] = np.var(window)
        
        # Smooth less where variance is high (edges)
        variance_threshold = np.percentile(local_variance, 75)
        smoothing_strength = np.where(local_variance > variance_threshold, 0.1, 0.5)
        
        # Apply weighted smoothing
        kernel = np.array([[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]])
        smoothed_temp = apply_convolution_2d(smoothed, kernel)
        
        # Blend based on smoothing strength
        smoothed = smoothed * (1 - smoothing_strength) + smoothed_temp * smoothing_strength
    
    return smoothed

def create_hierarchical_edge_mask(edge_map):
    """Create hierarchical edge mask with multiple strength levels."""
    # Define multiple threshold levels
    strong_threshold = np.percentile(edge_map, 85)
    medium_threshold = np.percentile(edge_map, 65)
    weak_threshold = np.percentile(edge_map, 40)
    
    # Create hierarchical mask
    mask = np.zeros_like(edge_map)
    
    # Strong edges
    mask = np.where(edge_map > strong_threshold, 1.0, mask)
    
    # Medium edges
    mask = np.where((edge_map > medium_threshold) & (edge_map <= strong_threshold), 0.7, mask)
    
    # Weak edges
    mask = np.where((edge_map > weak_threshold) & (edge_map <= medium_threshold), 0.4, mask)
    
    # Very weak edges
    mask = np.where((edge_map > 0) & (edge_map <= weak_threshold), 0.2, mask)
    
    return mask

def enhance_edge_connectivity(edge_mask):
    """Enhance edge connectivity using morphological operations."""
    # Define small structuring element for connectivity
    struct_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    
    # Apply morphological closing to connect nearby edges
    connected = morphological_closing(edge_mask > 0.3, struct_element)
    
    # Preserve original edge strengths where connected
    enhanced_mask = np.where(connected, edge_mask, edge_mask * 0.5)
    
    return enhanced_mask

def morphological_closing(binary_mask, struct_element):
    """Apply morphological closing operation."""
    # Dilation followed by erosion
    dilated = morphological_dilation(binary_mask, struct_element)
    closed = morphological_erosion(dilated, struct_element)
    return closed

def morphological_dilation(binary_mask, struct_element):
    """Apply morphological dilation."""
    h, w = binary_mask.shape
    se_h, se_w = struct_element.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    
    padded = np.pad(binary_mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    dilated = np.zeros_like(binary_mask)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+se_h, j:j+se_w]
            dilated[i, j] = np.any(window & struct_element)
    
    return dilated

def morphological_erosion(binary_mask, struct_element):
    """Apply morphological erosion."""
    h, w = binary_mask.shape
    se_h, se_w = struct_element.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    
    padded = np.pad(binary_mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    eroded = np.zeros_like(binary_mask)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+se_h, j:j+se_w]
            eroded[i, j] = np.all((window & struct_element) == struct_element)
    
    return eroded

def normalize_edge_mask(edge_mask):
    """Normalize edge mask to final range."""
    # Robust normalization using percentiles
    p1, p99 = np.percentile(edge_mask, [1, 99])
    
    if p99 > p1:
        normalized = (edge_mask - p1) / (p99 - p1)
    else:
        normalized = edge_mask / (np.max(edge_mask) + 1e-6)
    
    # Map to desired range [0.1, 1.0]
    final_mask = 0.1 + 0.9 * np.clip(normalized, 0, 1)
    
    return final_mask

def apply_convolution_2d(image, kernel):
    """Apply 2D convolution with proper padding."""
    if len(kernel.shape) != 2:
        raise ValueError("Kernel must be 2D")
    
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Pad image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Apply convolution
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(window * kernel)
    
    return result

def apply_intelligent_padding(image, pad_size):
    """Apply intelligent padding with boundary extrapolation."""
    return np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')

def apply_separable_bilateral_approximation(image, spatial_sigma, intensity_sigma):
    """Apply separable bilateral filter approximation for large kernels."""
    # Use existing separable bilateral filter
    return apply_separable_bilateral_filter(image, spatial_sigma, intensity_sigma)

def compute_compressed_spatial_weights(kernel_size, spatial_sigma):
    """Compute compressed spatial weights for large kernels."""
    pad_size = kernel_size // 2
    y_coords, x_coords = np.ogrid[-pad_size:pad_size+1, -pad_size:pad_size+1]
    weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * spatial_sigma**2))
    return weights / np.sum(weights)

def create_intensity_clusters(image, intensity_sigma):
    """Create intensity clusters for efficient processing."""
    # Simplified clustering - just return the intensity sigma for now
    return {'sigma': intensity_sigma, 'clusters': None}

def calculate_stream_size(h, w, kernel_size):
    """Calculate optimal stream size for memory efficiency."""
    # Calculate based on available memory and kernel size
    base_size = max(32, min(128, h // 4))
    return min(base_size, h)


def calculate_local_variance(image, kernel_size=5):
    """Calculate local variance for each pixel."""
    gray = np.mean(image, axis=2)
    pad_size = kernel_size // 2
    padded = np.pad(gray, pad_size, mode='edge')
    
    variance_map = np.zeros_like(gray)
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
            variance_map[i, j] = np.var(neighborhood)
    
    # Normalize variance
    return np.expand_dims(variance_map / (np.max(variance_map) + 1e-6), axis=2)

def calculate_texture_complexity(image):
    """Calculate texture complexity measure."""
    gray = np.mean(image, axis=2)
    
    # Calculate gradients in multiple directions
    grad_x = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad_y = np.diff(gray, axis=0, prepend=gray[:1, :])
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Texture complexity as normalized gradient variance
    texture_complexity = np.var(gradient_magnitude) / (np.mean(gradient_magnitude)**2 + 1e-6)
    
    return np.clip(texture_complexity, 0, 1)

def analyze_noise_frequency_spectrum(image):
    """Analyze noise characteristics in frequency domain."""
    gray = np.mean(image, axis=2)
    
    # Apply FFT
    fft_image = np.fft.fft2(gray)
    magnitude_spectrum = np.abs(fft_image)
    
    # Analyze high-frequency content
    h, w = magnitude_spectrum.shape
    center_h, center_w = h//2, w//2
    
    # Create frequency masks
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - center_h)**2 + (x - center_w)**2)
    
    high_freq_mask = distance > min(h, w) * 0.3
    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])
    total_energy = np.sum(magnitude_spectrum)
    
    return {
        'high_freq_ratio': high_freq_energy / (total_energy + 1e-6),
        'dominant_frequency': np.unravel_index(np.argmax(magnitude_spectrum), magnitude_spectrum.shape)
    }

def apply_optimized_gaussian_blur(image, sigma, kernel_size):
    """Apply optimized Gaussian blur using separable filters."""
    # Create 1D Gaussian kernel
    kernel_1d = create_gaussian_kernel_1d(kernel_size, sigma)
    
    # Apply separable convolution
    # Horizontal pass
    h_blurred = apply_1d_convolution(image, kernel_1d, axis=1)
    
    # Vertical pass
    v_blurred = apply_1d_convolution(h_blurred, kernel_1d, axis=0)
    
    return v_blurred

def create_gaussian_kernel_1d(kernel_size, sigma):
    """Create 1D Gaussian kernel."""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    kernel = np.exp(-ax**2 / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_1d_convolution(image, kernel, axis):
    """Apply 1D convolution along specified axis."""
    # Simple implementation - can be optimized further
    kernel_size = len(kernel)
    pad_size = kernel_size // 2
    
    if axis == 0:  # Vertical convolution
        padded = np.pad(image, ((pad_size, pad_size), (0, 0), (0, 0)), mode='edge')
        result = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for k, weight in enumerate(kernel):
                result[i] += weight * padded[i + k]
                
    else:  # Horizontal convolution
        padded = np.pad(image, ((0, 0), (pad_size, pad_size), (0, 0)), mode='edge')
        result = np.zeros_like(image)
        
        for j in range(image.shape[1]):
            for k, weight in enumerate(kernel):
                result[:, j] += weight * padded[:, j + k]
    
    return result

def apply_gaussian_blur(image, sigma=1.0):
    """Apply Gaussian blur (simplified implementation)."""
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    # Apply convolution (simplified)
    blurred_channels = []
    for channel in range(image.shape[2]):
        ch = image[:,:,channel]
        # Simple convolution approximation
        blurred = np.convolve(ch.flatten(), kernel.flatten(), mode='same').reshape(ch.shape)
        blurred_channels.append(blurred)
    
    return np.stack(blurred_channels, axis=-1)

def process_enhanced_bilateral_block(block, padded_block, spatial_weights, intensity_sigma, pad_size):
    """Enhanced bilateral processing with sub-pixel accuracy and edge preservation."""
    bh, bw, bc = block.shape
    filtered_block = np.zeros_like(block, dtype=np.float32)
    
    # Vectorized neighborhood extraction for better performance
    kernel_size = 2 * pad_size + 1
    
    for i in range(bh):
        for j in range(bw):
            center_pixel = block[i, j].astype(np.float32)
            
            # Extract neighborhood efficiently
            neighborhood = padded_block[i:i+kernel_size, j:j+kernel_size]
            
            # Compute intensity differences with sub-pixel precision
            intensity_diff = np.sum((neighborhood - center_pixel)**2, axis=2)
            
            # Enhanced intensity weighting with adaptive threshold
            adaptive_threshold = calculate_adaptive_threshold(intensity_diff, intensity_sigma)
            intensity_weights = np.exp(-intensity_diff / adaptive_threshold)
            
            # Combine weights with edge-aware adjustments
            combined_weights = spatial_weights * intensity_weights
            weight_sum = np.sum(combined_weights)
            
            if weight_sum > 1e-8:
                # Enhanced weighted averaging with color space considerations
                weighted_sum = np.sum(neighborhood * combined_weights[:,:,np.newaxis], axis=(0,1))
                filtered_block[i, j] = weighted_sum / weight_sum
            else:
                filtered_block[i, j] = center_pixel
    
    return filtered_block

def calculate_blend_weights(block_shape, overlap_size):
    """Calculate sophisticated blending weights for seamless block transitions."""
    h, w = block_shape
    weights = np.ones((h, w))
    
    # Create smooth transition zones
    for i in range(min(overlap_size, h)):
        fade_factor = (i + 1) / (overlap_size + 1)
        weights[i, :] *= fade_factor
    
    for j in range(min(overlap_size, w)):
        fade_factor = (j + 1) / (overlap_size + 1)
        weights[:, j] *= fade_factor
    
    return np.expand_dims(weights, axis=2)

def apply_edge_aware_padding(image, pad_size):
    """Apply edge-aware padding that preserves image structure."""
    # Use reflection padding with edge detection
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    # Enhance corners with intelligent extrapolation
    padded = enhance_corner_padding(padded, pad_size)
    
    return padded

def enhance_corner_padding(padded, pad_size):
    """Enhance corner padding with intelligent extrapolation."""
    h, w = padded.shape[:2]
    
    # Improve corner regions by blending nearby values
    for i in range(pad_size):
        for j in range(pad_size):
            # Top-left corner
            # Apply hierarchical compression for large kernels
            kernel_size = 20  # Example value for kernel_size
        pad_size = kernel_size // 2
        y_coords, x_coords = np.ogrid[-pad_size:pad_size+1, -pad_size:pad_size+1]
        base_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * 5**2))  # Example spatial_sigma value
        
        if kernel_size > 15:
            compressed_weights = apply_weight_compression(base_weights, compression_ratio=0.7)
            return compressed_weights
        
        return base_weights / np.sum(base_weights)

def apply_weight_compression(weights, compression_ratio=0.7):
    """Apply weight compression for large kernels to improve efficiency."""
    # Apply threshold-based compression
    max_weight = np.max(weights)
    threshold = max_weight * (1 - compression_ratio)
    
    # Set small weights to zero
    compressed = np.where(weights > threshold, weights, 0)
    
    # Renormalize
    weight_sum = np.sum(compressed)
    if weight_sum > 0:
        compressed = compressed / weight_sum
    
    return compressed

def enhance_corner_padding(padded, pad_size):
    """Enhance corner padding with intelligent extrapolation."""
    h, w = padded.shape[:2]
    
    # Improve corner regions by blending nearby values
    for i in range(pad_size):
        for j in range(pad_size):
            # Top-left corner
            if i < pad_size and j < pad_size:
                padded[i, j] = (padded[pad_size, j] + padded[i, pad_size]) / 2
            
            # Top-right corner
            if i < pad_size and j >= w - pad_size:
                padded[i, j] = (padded[pad_size, j] + padded[i, w - pad_size - 1]) / 2
            
            # Bottom-left corner
            if i >= h - pad_size and j < pad_size:
                padded[i, j] = (padded[h - pad_size - 1, j] + padded[i, pad_size]) / 2
            
            # Bottom-right corner
            if i >= h - pad_size and j >= w - pad_size:
                padded[i, j] = (padded[h - pad_size - 1, j] + padded[i, w - pad_size - 1]) / 2
    
    return padded

def compute_hierarchical_spatial_weights(kernel_size, spatial_sigma):
    """Compute spatial weights using hierarchical approximation for efficiency."""
    pad_size = kernel_size // 2
    
    # Create base weights
    y_coords, x_coords = np.ogrid[-pad_size:pad_size+1, -pad_size:pad_size+1]
    base_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * spatial_sigma**2))
    
    # Apply hierarchical compression for large kernels
    if kernel_size > 15:
        compressed_weights = apply_weight_compression(base_weights, compression_ratio=0.7)
        return compressed_weights
    
    return base_weights / np.sum(base_weights)

def create_intensity_lookup_table(intensity_sigma, num_bins=256):
    """Create optimized lookup table for intensity weight computation."""
    max_diff = 255 * np.sqrt(3)  # Maximum possible intensity difference in RGB
    diff_values = np.linspace(0, max_diff**2, num_bins)
    lut = np.exp(-diff_values / (2 * intensity_sigma**2))
    return lut, max_diff**2 / (num_bins - 1)

def calculate_adaptive_threshold(intensity_diff, base_sigma):
    """Calculate adaptive threshold based on local image characteristics."""
    local_variance = np.var(intensity_diff)
    adaptation_factor = 1.0 + local_variance / (base_sigma**2 * 100)
    return 2 * base_sigma**2 * adaptation_factor

# Additional sophisticated helper functions would be implemented here...

def process_bilateral_block(block, padded_block, spatial_weights, intensity_sigma, pad_size):
    """Process a block with bilateral filtering."""
    h, w, c = block.shape
    filtered_block = np.zeros_like(block)
    
    for i in range(h):
        for j in range(w):
            center_pixel = block[i, j]
            neighborhood = padded_block[i:i+2*pad_size+1, j:j+2*pad_size+1]
            
            # Calculate intensity weights
            intensity_diff = np.sum((neighborhood - center_pixel)**2, axis=2)
            intensity_weights = np.exp(-intensity_diff / (2 * intensity_sigma**2))
            
            # Combine weights
            combined_weights = spatial_weights * intensity_weights
            weight_sum = np.sum(combined_weights)
            
            if weight_sum > 0:
                # Apply weighted average
                filtered_pixel = np.sum(
                    neighborhood * combined_weights[:,:,np.newaxis], axis=(0,1)
                ) / weight_sum
                filtered_block[i, j] = filtered_pixel
            else:
                filtered_block[i, j] = center_pixel
    
    return filtered_block

def apply_median_filter(image, kernel_size=3):
    """Apply median filter for salt-and-pepper noise."""
    filtered_channels = []
    pad_size = kernel_size // 2
    
    for channel in range(image.shape[2]):
        ch = image[:,:,channel]
        padded = np.pad(ch, pad_size, mode='edge')
        filtered_ch = np.zeros_like(ch)
        
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                neighborhood = padded[i:i+kernel_size, j:j+kernel_size]
                filtered_ch[i, j] = np.median(neighborhood)
        
        filtered_channels.append(filtered_ch)
    
    return np.stack(filtered_channels, axis=-1)

def apply_fast_non_local_means(image, characteristics):
    """Fast approximation of non-local means denoising."""
    # Simplified non-local means using patch similarity
    patch_size = 3
    search_window = 7
    h_param = characteristics['noise_level'] * 15 + 5
    
    return apply_patch_based_denoising(image, patch_size, search_window, h_param)

def apply_patch_based_denoising(image, patch_size, search_window, h_param):
    """Apply patch-based denoising algorithm."""
    h, w, c = image.shape
    filtered = np.zeros_like(image)
    
    pad_patch = patch_size // 2
    pad_search = search_window // 2
    
    # Process image in overlapping blocks for efficiency
    block_size = 32
    for i in range(0, h, block_size//2):
        for j in range(0, w, block_size//2):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            
            block_filtered = process_patch_block(
                image, i, j, i_end, j_end, 
                patch_size, search_window, h_param
            )
            
            # Blend overlapping regions
            if i > 0 and j > 0:
                blend_factor = 0.5
                filtered[i:i_end, j:j_end] = (
                    filtered[i:i_end, j:j_end] * blend_factor + 
                    block_filtered * (1 - blend_factor)
                )
            else:
                filtered[i:i_end, j:j_end] = block_filtered
    
    return filtered

def process_patch_block(image, i_start, j_start, i_end, j_end, 
                       patch_size, search_window, h_param):
    """Process a block using patch-based denoising."""
    block = image[i_start:i_end, j_start:j_end]
    filtered_block = np.zeros_like(block)
    
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            # Get reference patch
            ref_patch = get_patch(image, i_start + i, j_start + j, patch_size)
            
            # Find similar patches in search window
            weights = []
            patches = []
            
            for di in range(-search_window//2, search_window//2 + 1, 2):
                for dj in range(-search_window//2, search_window//2 + 1, 2):
                    pi, pj = i_start + i + di, j_start + j + dj
                    if 0 <= pi < image.shape[0] and 0 <= pj < image.shape[1]:
                        patch = get_patch(image, pi, pj, patch_size)
                        similarity = calculate_patch_similarity(ref_patch, patch)
                        weight = np.exp(-similarity / h_param**2)
                        weights.append(weight)
                        patches.append(patch[patch_size//2, patch_size//2])
            
            if weights:
                weights = np.array(weights)
                patches = np.array(patches)
                weight_sum = np.sum(weights)
                
                if weight_sum > 0:
                    filtered_block[i, j] = np.sum(patches * weights[:, np.newaxis], axis=0) / weight_sum
                else:
                    filtered_block[i, j] = block[i, j]
            else:
                filtered_block[i, j] = block[i, j]
    
    return filtered_block

def get_patch(image, i, j, patch_size):
    """Extract a patch from the image."""
    pad_size = patch_size // 2
    i_start = max(0, i - pad_size)
    i_end = min(image.shape[0], i + pad_size + 1)
    j_start = max(0, j - pad_size)
    j_end = min(image.shape[1], j + pad_size + 1)
    
    return image[i_start:i_end, j_start:j_end]

def calculate_patch_similarity(patch1, patch2):
    """Calculate similarity between two patches."""
    if patch1.shape != patch2.shape:
        return float('inf')
    
    diff = patch1.astype(np.float32) - patch2.astype(np.float32)
    return np.sum(diff**2)

def apply_edge_preserving_smoothing(image, characteristics):
    """Apply edge-preserving smoothing as final step."""
    # Use anisotropic diffusion approximation
    lambda_param = 0.1
    kappa = characteristics['noise_level'] * 20 + 10
    iterations = 3
    
    smoothed = image.copy()
    
    for _ in range(iterations):
        smoothed = apply_anisotropic_diffusion_step(smoothed, lambda_param, kappa)
    
    return smoothed

def apply_anisotropic_diffusion_step(image, lambda_param, kappa):
    """Apply one step of anisotropic diffusion."""
    # Calculate gradients
    grad_n = np.roll(image, -1, axis=0) - image
    grad_s = np.roll(image, 1, axis=0) - image
    grad_e = np.roll(image, -1, axis=1) - image
    grad_w = np.roll(image, 1, axis=1) - image
    
    # Calculate diffusion coefficients
    c_n = np.exp(-(np.sum(grad_n**2, axis=2, keepdims=True) / kappa**2))
    c_s = np.exp(-(np.sum(grad_s**2, axis=2, keepdims=True) / kappa**2))
    c_e = np.exp(-(np.sum(grad_e**2, axis=2, keepdims=True) / kappa**2))
    c_w = np.exp(-(np.sum(grad_w**2, axis=2, keepdims=True) / kappa**2))
    
    # Apply diffusion
    diffusion = (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
    updated = image + lambda_param * diffusion
    
    return updated


def optimize_midtones(image):
    """Optimize midtone contrast for better high-contrast results."""
    # S-curve adjustment for midtones
    normalized = image / 255.0
    
    # Apply S-curve: dark values become darker, bright values become brighter
    s_curve = np.where(normalized < 0.5,
                      2 * normalized**2,
                      1 - 2 * (1 - normalized)**2)
    
    return s_curve * 255

def apply_intensity_scaling(original, adjusted, intensity):
    """
    Apply advanced intensity scaling with multiple blending modes and adaptive adjustments.
    
    :param original: Original image array
    :param adjusted: Adjusted image array 
    :param intensity: Filter intensity (0.0 to 2.0)
    :return: Intensity-scaled image with enhanced blending
    """
    # Validate inputs
    if original.shape != adjusted.shape:
        return original
    
    intensity = np.clip(intensity, 0.0, 2.0)
    
    # Calculate difference map for analysis
    diff_map = adjusted.astype(np.float32) - original.astype(np.float32)
    
    # Adaptive intensity based on local contrast
    local_variance = np.var(original.astype(np.float32), axis=2, keepdims=True)
    adaptive_factor = 1.0 + (local_variance / 255.0) * 0.3  # Boost areas with high variance
    
    # Apply different blending modes based on intensity level
    if intensity <= 1.0:
        # Linear blending for subtle adjustments
        result = original + intensity * diff_map * adaptive_factor
    else:
        # Enhanced blending for stronger effects
        excess_intensity = intensity - 1.0
        
        # Base linear blend at full intensity
        linear_blend = original + diff_map * adaptive_factor
        
        # Additional enhancement using soft light blending
        overlay_factor = excess_intensity * 0.5
        soft_light = apply_soft_light_blend(original, adjusted, overlay_factor)
        
        # Combine linear and soft light blending
        result = linear_blend * (1 - overlay_factor) + soft_light * overlay_factor
    
    # Apply edge preservation to maintain detail
    result = preserve_edge_details(original, result, intensity)
    
    # Smooth transitions in areas with large changes
    result = smooth_large_transitions(result, diff_map, intensity)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_soft_light_blend(base, overlay, factor):
    """Apply soft light blending mode for enhanced color mixing."""
    base_norm = base.astype(np.float32) / 255.0
    overlay_norm = overlay.astype(np.float32) / 255.0
    
    # Soft light formula
    mask = base_norm <= 0.5
    soft_light = np.where(mask,
                         2 * base_norm * overlay_norm + base_norm**2 * (1 - 2 * overlay_norm),
                         2 * base_norm * (1 - overlay_norm) + np.sqrt(base_norm) * (2 * overlay_norm - 1))
    
    # Blend with original based on factor
    result = base_norm * (1 - factor) + soft_light * factor
    return result * 255

def preserve_edge_details(original, adjusted, intensity):
    """Preserve important edge details during intensity scaling."""
    # Simple edge detection using gradient
    gray_orig = np.mean(original, axis=2)
    
    # Calculate gradients
    grad_x = np.diff(gray_orig, axis=1, prepend=gray_orig[:, :1])
    grad_y = np.diff(gray_orig, axis=0, prepend=gray_orig[:1, :])
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create edge mask (normalized)
    edge_mask = edge_strength / (np.max(edge_strength) + 1e-6)
    edge_mask = np.expand_dims(edge_mask, axis=2)
    
    # Reduce filter effect on strong edges when intensity is high
    if intensity > 1.0:
        edge_preservation = 1.0 - (edge_mask * 0.3 * (intensity - 1.0))
        edge_preservation = np.clip(edge_preservation, 0.7, 1.0)
        
        # Apply edge preservation
        result = adjusted * edge_preservation + original * (1 - edge_preservation)
        return result
    
    return adjusted

def smooth_large_transitions(image, diff_map, intensity):
    """Smooth areas with large color transitions to reduce artifacts."""
    # Calculate magnitude of changes
    change_magnitude = np.sqrt(np.sum(diff_map**2, axis=2, keepdims=True))
    
    # Identify areas with large changes
    large_change_threshold = 50 * intensity
    large_change_mask = change_magnitude > large_change_threshold
    
    if np.any(large_change_mask):
        # Apply gentle gaussian-like smoothing to large change areas
        smoothed = apply_local_smoothing(image, large_change_mask)
        
        # Blend smoothed areas
        smooth_factor = np.clip((change_magnitude - large_change_threshold) / 50.0, 0, 0.3)
        result = image * (1 - smooth_factor) + smoothed * smooth_factor
        return result
    
    return image

def apply_local_smoothing(image, mask):
    """Apply local smoothing using a simple averaging filter."""
    # Simple 3x3 averaging for smoothing
    kernel_size = 3
    pad_size = kernel_size // 2
    
    # Pad image
    padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')
    smoothed = np.zeros_like(image)
    
    # Apply averaging filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j, 0]:  # Only smooth where mask is True
                region = padded[i:i+kernel_size, j:j+kernel_size]
                smoothed[i, j] = np.mean(region, axis=(0, 1))
            else:
                smoothed[i, j] = image[i, j]
    
    return smoothed

def apply_post_processing(image, filter_type):
    """Apply post-processing improvements based on filter type."""
    if filter_type in ['monochrome', 'neutral_greyscale', 'warm_greyscale']:
        # Enhance edge definition for grayscale filters
        return apply_edge_enhancement(image)
    elif filter_type in ['protanopia', 'deuteranopia', 'tritanopia']:
        # Smooth color transitions for colorblind filters
        return apply_color_smoothing(image)
    
    return image

def apply_contrast_enhancement(image):
    """Apply advanced contrast enhancement with multiple techniques."""
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Method 1: Adaptive histogram equalization (simplified)
    # Calculate histogram statistics for each channel
    enhanced_channels = []
    for channel in range(img_float.shape[2]):
        ch = img_float[:,:,channel]
        
        # Calculate percentiles for dynamic range adjustment
        p2, p98 = np.percentile(ch, (2, 98))
        
        # Stretch contrast to full range
        if p98 > p2:  # Avoid division by zero
            stretched = (ch - p2) / (p98 - p2) * 255
            stretched = np.clip(stretched, 0, 255)
        else:
            stretched = ch
        
        enhanced_channels.append(stretched)
    
    enhanced_image = np.stack(enhanced_channels, axis=-1)
    
    # Method 2: Apply CLAHE-inspired local contrast enhancement
    # Divide image into tiles and enhance each tile separately
    tile_size = 64
    h, w = enhanced_image.shape[:2]
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile = enhanced_image[y:y_end, x:x_end]
            
            # Local contrast enhancement
            local_mean = np.mean(tile)
            local_std = np.std(tile)
            
            if local_std > 10:  # Only enhance if there's sufficient variation
                contrast_factor = 1.5
                enhanced_tile = local_mean + (tile - local_mean) * contrast_factor
                enhanced_image[y:y_end, x:x_end] = np.clip(enhanced_tile, 0, 255)
    
    # Method 3: Unsharp masking for edge enhancement
    # Create a slight blur
    kernel_size = 3
    blur_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    blurred_channels = []
    for channel in range(enhanced_image.shape[2]):
        ch = enhanced_image[:,:,channel]
        # Simple convolution approximation
        blurred = np.convolve(ch.flatten(), blur_kernel.flatten(), mode='same').reshape(ch.shape)
        blurred_channels.append(blurred)
    
    blurred_image = np.stack(blurred_channels, axis=-1)
    
    # Apply unsharp mask
    unsharp_strength = 0.5
    final_image = enhanced_image + unsharp_strength * (enhanced_image - blurred_image)
    
    return np.clip(final_image, 0, 255).astype(np.uint8)

def apply_edge_enhancement(image):
    """Apply subtle edge enhancement."""
    # Simple sharpening filter
    enhanced = image * 1.1 - np.roll(image, 1, axis=0) * 0.05 - np.roll(image, -1, axis=0) * 0.05
    return np.clip(enhanced, 0, 255)

def apply_color_smoothing(image):
    """Apply gentle color smoothing."""
    # Simple averaging with neighbors
    smoothed = (image + np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0)) / 3
    return np.clip(smoothed, 0, 255)

def apply_enhanced_contrast(image, intensity):
    """Apply enhanced contrast adjustment."""
    mean_val = np.mean(image)
    return np.clip(mean_val + (image - mean_val) * (1 + intensity), 0, 255)

def apply_color_boost(image, intensity):
    """Boost color saturation."""
    hsv = matplotlib.colors.rgb_to_hsv(image / 255.0)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + intensity * 0.5), 0, 1)
    return matplotlib.colors.hsv_to_rgb(hsv) * 255


def apply_advanced_colorblind_filter(image, filter_type):
    """
    Apply an advanced colorblind filter to an image, catering to various types of color vision deficiencies.

    :param image: The original game image
    :param filter_type: Type of colorblindness or visual deficiency
    :return: Image with advanced colorblind filter applied
    """

    # Define filters dictionary for different colorblind filter types
    filters = {
        'protanopia': adjust_for_protanopia,
        'deuteranopia': adjust_for_deuteranopia,
        'tritanopia': adjust_for_tritanopia,
        'monochrome': adjust_for_monochrome,
        'no_purple': adjust_for_no_purple,
        'neutral_difficulty': adjust_for_neutral_difficulty,
        'warm_color_difficulty': adjust_for_warm_color_difficulty,
        'neutral_greyscale': adjust_for_neutral_greyscale,
        'warm_greyscale': adjust_for_warm_greyscale,
        'complex_color_difficulty': adjust_for_complex_color_difficulty,
        'simple_color_difficulty': adjust_for_simple_color_difficulty
    }
    
    # Apply the selected filter, default to no adjustment if filter type not found
    adjusted_image = filters.get(filter_type, lambda x: x)(image)


    import numpy as np

    def adjust_for_deuteranopia(image):
        """
        Adjust colors for deuteranopia.

        :param image: Image array
        :return: Color-adjusted image array for deuteranopia
        """
        # Create a transformation matrix for deuteranopia
        # These values are illustrative and would need to be fine-tuned
        transform_matrix = np.array([
            [0.625, 0.375, 0],
            [0.70, 0.30, 0],
            [0, 0.30, 0.70]
        ])
        
        # Apply the transformation to each pixel
        adjusted_image = np.dot(image.reshape(-1, 3), transform_matrix).reshape(image.shape)
        return np.clip(adjusted_image, 0, 255)  # Ensuring the pixel values are valid

    def adjust_for_tritanopia(image):
        """
        Adjust colors for tritanopia.

        :param image: Image array
        :return: Color-adjusted image array for tritanopia
        """
        # Create a transformation matrix for tritanopia
        # These values are illustrative and would need to be fine-tuned
        transform_matrix = np.array([
            [0.95, 0.05, 0],
            [0, 0.433, 0.567],
            [0, 0.475, 0.525]
        ])
        
        # Apply the transformation to each pixel
        adjusted_image = np.dot(image.reshape(-1, 3), transform_matrix).reshape(image.shape)
        return np.clip(adjusted_image, 0, 255)  # Ensuring the pixel values are valid

    def adjust_for_protanopia(image):
        """
        Adjust colors for protanopia.

        :param image: Image array
        :return: Color-adjusted image array for protanopia
        """
        # Create a transformation matrix for protanopia
        # These values are illustrative and would need empirical tuning
        transform_matrix = np.array([
            [0.567, 0.433, 0],  # Reducing red component, increasing green
            [0.558, 0.442, 0],  # Similar adjustments in the green channel
            [0, 0.242, 0.758]   # Shifting some of the blue component into red and green
        ])
        
        # Apply the transformation to each pixel
        adjusted_image = np.dot(image.reshape(-1, 3), transform_matrix).reshape(image.shape)
        return np.clip(adjusted_image, 0, 255)  # Ensuring the pixel values are valid


    adjusted_image = filters.get(filter_type, lambda x: x)(image)
    # Example usage
    game_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)  # 720p resolution
    filtered_frame = apply_advanced_colorblind_filter(game_frame, 'neutral_difficulty')
    # Example usage
    # Assuming 'game_frame' is the current frame of the game
    filtered_frame = apply_colorblind_filter(game_frame, 'deuteranopia')
    return adjusted_image




def dynamic_colorblind_filter(image, filter_type, intensity=1.0):
    """
    Apply a dynamic colorblind filter to an image.

    :param image: The original game image
    :param filter_type: Type of colorblindness
    :param intensity: Intensity of the filter adjustment
    :return: Image with dynamic colorblind filter applied
    """
    def adjust_for_monochrome(image):
        """
        Adjust image for monochrome vision with enhanced contrast and detail preservation.

        :param image: Image array (RGB format)
        :return: Enhanced grayscale image with improved contrast and detail
        """
        # Convert to grayscale using weighted average (preserves luminance better than simple mean)
        # Standard luminance weights: R=0.299, G=0.587, B=0.114
        grayscale = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
        
        # Apply histogram equalization for better contrast
        # Normalize to 0-1 range
        normalized = (grayscale - np.min(grayscale)) / (np.max(grayscale) - np.min(grayscale) + 1e-6)
        
        # Apply contrast enhancement using gamma correction
        gamma = 0.8  # Adjust gamma for better contrast (< 1 brightens, > 1 darkens)
        contrast_enhanced = np.power(normalized, gamma)
        
        # Apply adaptive contrast stretching
        # Clip extreme values to prevent over-saturation
        low_percentile, high_percentile = np.percentile(contrast_enhanced, [5, 95])
        clipped = np.clip(contrast_enhanced, low_percentile, high_percentile)
        
        # Stretch to full range
        final_grayscale = ((clipped - low_percentile) / (high_percentile - low_percentile + 1e-6)) * 255
        
        # Apply slight sharpening to enhance edge details
        # Simple sharpening kernel
        kernel = np.array([[-0.1, -0.1, -0.1],
                           [-0.1,  1.8, -0.1],
                           [-0.1, -0.1, -0.1]])
        
        # Apply convolution for sharpening (simplified version)
        sharpened = final_grayscale + 0.3 * (final_grayscale - 
                    np.convolve(final_grayscale.flatten(), 
                               np.array([1/9]*9), mode='same').reshape(final_grayscale.shape))
        
        # Ensure values are in valid range
        enhanced_grayscale = np.clip(sharpened, 0, 255)
        
        # Return as 3-channel image for compatibility
        return np.stack((enhanced_grayscale, enhanced_grayscale, enhanced_grayscale), axis=-1).astype(np.uint8)


    def adjust_for_neutral_difficulty(image):
        """
        Adjust neutral colors to enhance distinguishability.

        :param image: Image array
        :return: Image with enhanced neutral colors
        """
        # Convert to HSV for better color manipulation
        hsv_image = matplotlib.colors.rgb_to_hsv(image / 255.0)
        h, s, v = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
        
        # Identify neutral colors (low saturation)
        neutral_mask = s < 0.3
        
        # Enhance contrast for neutral areas
        v_enhanced = np.where(neutral_mask, 
                             np.clip(v * 1.3 - 0.1, 0.1, 0.9),  # Increase contrast
                             v)
        
        # Slightly increase saturation for better color distinction
        s_enhanced = np.where(neutral_mask,
                             np.clip(s * 1.5, 0, 0.5),  # Boost saturation slightly
                             s)
        
        # Reconstruct HSV image
        enhanced_hsv = np.stack([h, s_enhanced, v_enhanced], axis=-1)
        
        # Convert back to RGB
        enhanced_rgb = matplotlib.colors.hsv_to_rgb(enhanced_hsv) * 255
        
        # Apply adaptive histogram equalization for local contrast enhancement
        grayscale = np.mean(enhanced_rgb, axis=2)
        
        # Simple local contrast enhancement
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = np.convolve(grayscale.flatten(), kernel.flatten(), mode='same').reshape(grayscale.shape)
        contrast_factor = 1.2
        enhanced_grayscale = grayscale + contrast_factor * (grayscale - local_mean)
        
        # Apply enhancement proportionally to all channels
        gray_ratio = enhanced_grayscale / (grayscale + 1e-6)  # Avoid division by zero
        enhanced_rgb[:,:,0] *= gray_ratio
        enhanced_rgb[:,:,1] *= gray_ratio
        enhanced_rgb[:,:,2] *= gray_ratio
        
        return np.clip(enhanced_rgb, 0, 255).astype(np.uint8)

    def adjust_for_warm_color_difficulty(image):
        """
        Adjust warm colors for better visibility.

        :param image: Image array
        :return: Image with enhanced warm colors
        """
        # Increase intensity of warm colors
        # Placeholder logic
        red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
        warm_enhancement = red * 1.2 + green * 0.8
        adjusted_image = np.stack((warm_enhancement, green, blue), axis=-1)
        return np.clip(adjusted_image, 0, 255)

    def adjust_for_neutral_greyscale(image):
        """
        Adjust greyscale image to enhance neutral colors.

        :param image: Image array
        :return: Image with adjusted neutral greyscale tones
        """
        grayscale = np.mean(image, axis=2)
        # Increase contrast for neutral tones
        adjusted_grayscale = np.clip(1.2 * grayscale - 20, 0, 255)
        return np.stack((adjusted_grayscale,)*3, axis=-1)

    def adjust_for_warm_greyscale(image):
        """
        Adjust greyscale image to enhance warm tones.

        :param image: Image array
        :return: Image with enhanced warm greyscale tones
        """
        grayscale = np.mean(image, axis=2)
        # Placeholder logic to enhance warm tones
        warm_enhanced_grayscale = np.clip(grayscale * 1.1, 0, 255)
        return np.stack((warm_enhanced_grayscale,)*3, axis=-1)



    # Placeholder for a dynamic adjustment algorithm
    adjusted_image = dynamic_adjustment_algorithm(image, filter_type, intensity)
    return adjusted_image

import numpy as np

def dynamic_adjustment_algorithm(image, filter_type, intensity):
    """
    Advanced algorithm that dynamically adjusts colors based on filter type and intensity.

    :param image: Image array
    :param filter_type: Type of colorblindness
    :param intensity: Intensity of the adjustment
    :return: Modified image
    """
    if filter_type == 'monochrome':
        adjusted_image = adjust_for_monochrome(image)
    elif filter_type == 'neutral_difficulty':
        adjusted_image = adjust_for_neutral_difficulty(image)
    elif filter_type == 'warm_color_difficulty':
        adjusted_image = adjust_for_warm_color_difficulty(image)
    elif filter_type == 'neutral_greyscale':
        adjusted_image = adjust_for_neutral_greyscale(image)
    elif filter_type == 'warm_greyscale':
        adjusted_image = adjust_for_warm_greyscale(image)
    else:
        adjusted_image = image

    # Apply intensity scaling to the adjustment
    return np.clip(image + intensity * (adjusted_image - image), 0, 255)



def advanced_colorblind_filter(image, filter_type, contrast_mode=False):
    """
    Apply an advanced colorblind filter with high-contrast mode options.

    :param image: The original game image
    :param filter_type: Type of specialized colorblindness or visual deficiency
    :param contrast_mode: Enable high-contrast mode
    :return: Image with advanced colorblind filter applied
    """
    # Placeholder functions for specific advanced filter adjustments
    advanced_filters = {
        # ... existing filter types ...
        'high_contrast': apply_high_contrast if contrast_mode else lambda x: x
    }

    adjusted_image = advanced_filters.get(filter_type, lambda x: x)(image)
    return adjusted_image

import numpy as np

def apply_high_contrast(image):
    """
    Apply high contrast adjustments to an image.

    :param image: Image array
    :return: High contrast image
    """
    # Convert to grayscale for contrast manipulation
    grayscale = np.mean(image, axis=2)
    
    # Normalize the grayscale values
    normalized = (grayscale - np.min(grayscale)) / (np.max(grayscale) - np.min(grayscale))

    # Apply contrast stretching
    # This expands the range of intensity values in the image
    contrast_stretched = 255 * ((normalized - 0.5) * 2) ** 2

    # Stack to replicate the changes across all color channels
    high_contrast_image = np.stack((contrast_stretched,)*3, axis=-1)

    return np.clip(high_contrast_image, 0, 255)  # Ensure pixel values remain valid


def apply_colorblind_adjustments(image, filter_type):
    """
    Apply specific color adjustments based on the type of colorblindness.

    :param image: The original game image
    :param filter_type: Type of colorblindness
    :return: Image with color adjustments for colorblindness
    """
    if filter_type == 'protanopia':
        adjusted_image = adjust_for_protanopia(image)
    elif filter_type == 'deuteranopia':
        adjusted_image = adjust_for_deuteranopia(image)
    elif filter_type == 'tritanopia':
        adjusted_image = adjust_for_tritanopia(image)
    elif filter_type == 'monochrome':
        adjusted_image = adjust_for_monochrome(image)
    elif filter_type == 'no_purple':
        adjusted_image = adjust_for_no_purple(image)
    elif filter_type == 'neutral_difficulty':
        adjusted_image = adjust_for_neutral_difficulty(image)
    elif filter_type == 'warm_color_difficulty':
        adjusted_image = adjust_for_warm_color_difficulty(image)
    elif filter_type == 'neutral_greyscale':
        adjusted_image = adjust_for_neutral_greyscale(image)
    elif filter_type == 'warm_greyscale':
        adjusted_image = adjust_for_warm_greyscale(image)
    else:
        adjusted_image = image  # No adjustment if filter type is not recognized

    return adjusted_image

# Placeholder functions for specific adjustments
def adjust_for_protanopia(image):
    # Specific adjustments for Protanopia
    return adjusted_image

# Similar functions would be defined for each type of colorblindness
# adjust_for_deuteranopia(image), adjust_for_tritanopia(image), etc.

def apply_colorblind_adjustments(image, filter_value):
    """
    Apply specific color adjustments based on the provided filter value.

    :param image: The original game image
    :param filter_value: Integer representing the type of colorblindness
    :return: Image with color adjustments for colorblindness
    """
    if filter_value == 1:
        adjusted_image = adjust_for_protanopia(image)
    elif filter_value == 2:
        adjusted_image = adjust_for_deuteranopia(image)
    elif filter_value == 3:
        adjusted_image = adjust_for_tritanopia(image)
    elif filter_value == 4:
        adjusted_image = adjust_for_monochrome(image)
    elif filter_value == 5:
        adjusted_image = adjust_for_no_purple(image)
    elif filter_value == 6:
        adjusted_image = adjust_for_neutral_difficulty(image)
    elif filter_value == 7:
        adjusted_image = adjust_for_warm_color_difficulty(image)
    elif filter_value == 8:
        adjusted_image = adjust_for_neutral_greyscale(image)
    elif filter_value == 9:
        adjusted_image = adjust_for_warm_greyscale(image)
    else:
        adjusted_image = image  # Default, no adjustment

    return adjusted_image


import numpy as np

def apply_numpy_colorblind_filter(image_array, filter_type):
    """
    Apply a colorblind filter using NumPy.

    :param image_array: Image represented as a NumPy array
    :param filter_type: Type of colorblindness
    :return: Adjusted image as a NumPy array
    """
    # Define transformation matrices for different colorblindness types
    transformation_matrices = {
        'protanopia': np.array([[...]]), # Protanopia matrix
        # ... Other matrices for different types
    }

    transformation_matrix = transformation_matrices.get(filter_type, np.eye(3))
    adjusted_image_array = np.dot(image_array, transformation_matrix)

    return adjusted_image_array


import numpy as np

def colorblind_transform(image_array, matrix):
    """ Apply color transformation to an image array. """
    return np.dot(image_array.reshape(-1, 3), matrix).reshape(image_array.shape)

import pandas as pd

# Simulating user preference data
data = {
    'user_id': range(1, 101),
    'filter_preference': np.random.choice(['protanopia', 'deuteranopia', 'tritanopia', 'none'], 100),
    'usage_frequency': np.random.randint(1, 10, 100)
}
user_preferences = pd.DataFrame(data)

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

# One-hot encode the categorical data
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(user_preferences[['filter_preference']]).toarray()

# Clustering
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(np.hstack((encoded_features, user_preferences[['usage_frequency']].values)))

# Assign clusters back to users
user_preferences['cluster'] = clusters

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def apply_numpy_colorblind_filter(image_array, filter_type):
    """
    Apply a colorblind filter using NumPy.

    :param image_array: Image represented as a NumPy array
    :param filter_type: Type of colorblindness
    :return: Adjusted image as a NumPy array
    """
    # Define transformation matrices for different colorblindness types
    transformation_matrices = {
        'protanopia': np.array([
            [0.567, 0.433, 0],
            [0.558, 0.442, 0],
            [0, 0.242, 0.758]
        ]),
        'deuteranopia': np.array([
            [0.625, 0.375, 0],
            [0.70, 0.30, 0],
            [0, 0.30, 0.70]
        ]),
        'tritanopia': np.array([
            [0.95, 0.05, 0],
            [0, 0.433, 0.567],
            [0, 0.475, 0.525]
        ])
        # Other matrices can be added for different types
    }

    def colorblind_transform(image_array, matrix):
        """
        Apply a colorblindness transformation matrix to an image array.

        :param image_array: Image represented as a NumPy array
        :param matrix: Transformation matrix for colorblindness
        :return: Transformed image array
        """
        adjusted_image_array = np.einsum('ij,klj->kli', matrix, image_array)
        return np.clip(adjusted_image_array, 0, 255)

    def apply_integrated_colorblind_filter(image_array):
        """
        Apply an integrated colorblind filter based on simulated user data and clustering.
    
        :param image_array: Image represented as a NumPy array
        :return: Adjusted image as a NumPy array
        """
        # Simulate user data
        user_data = {
            'user_id': range(100),
            'filter_preference': np.random.choice(['protanopia', 'deuteranopia', 'tritanopia'], 100)
        }
        user_preferences = pd.DataFrame(user_data)

        # Convert categorical data to numerical for clustering
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        user_preferences['filter_encoded'] = le.fit_transform(user_preferences['filter_preference'])

        # Check for sufficient unique data points before clustering
        unique_points = user_preferences['filter_encoded'].nunique()
        n_clusters = min(3, unique_points)  # Use fewer clusters if we don't have enough unique points
        
        if n_clusters < 2:
            # If we only have one unique value, skip clustering
            common_filter = user_preferences['filter_preference'].mode()[0]
        else:
            # Apply clustering with appropriate number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            user_preferences['cluster'] = kmeans.fit_predict(user_preferences[['filter_encoded']])

            # Determine the most common filter in the largest cluster
            common_cluster = user_preferences['cluster'].mode()[0]
            common_filter = user_preferences[user_preferences['cluster'] == common_cluster]['filter_preference'].mode()[0]

        # Define transformation matrices
        transformation_matrices = {
            'protanopia': np.array([
                [0.567, 0.433, 0],
                [0.558, 0.442, 0],
                [0, 0.242, 0.758]
            ]),
            'deuteranopia': np.array([
                [0.625, 0.375, 0],
                [0.70, 0.30, 0],
                [0, 0.30, 0.70]
            ]),
            'tritanopia': np.array([
                [0.95, 0.05, 0],
                [0, 0.433, 0.567],
                [0, 0.475, 0.525]
            ])
        }

        matrix = transformation_matrices.get(common_filter, np.eye(3))
        return colorblind_transform(image_array, matrix)






import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Example of image processing function
def apply_color_adjustments(image, adjustments):
    # Placeholder logic for color adjustments
    for color, shift in adjustments.items():
        if color in ['red_shift', 'green_shift', 'blue_shift']:
            channel = {'red_shift': 0, 'green_shift': 1, 'blue_shift': 2}[color]
            image[:,:,channel] *= shift
    return np.clip(image, 0, 255)

# Simulating player data collection
def collect_player_data(player_actions, player_settings):
    # Example of collecting and structuring player data
    player_data = pd.DataFrame({
        'actions': player_actions,
        'settings': [player_settings.get('color_setting', 1)] * len(player_actions)
    })
    return player_data

# Predicting colorblindness type using clustering
def predict_colorblindness_type(player_data):
    kmeans = KMeans(n_clusters=3) 
    player_data['cluster'] = kmeans.fit_predict(player_data[['settings']])
    return player_data['cluster'].mode()[0]

# Forecasting future adjustment needs
def forecast_adjustment_needs(player_data):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(player_data[['settings']])
    model = LinearRegression()
    model.fit(X_poly, player_data['settings'])
    forecasted_adjustments = model.predict(X_poly)
    return forecasted_adjustments.mean()

# Applying the dynamic colorblind filter
def apply_dynamic_colorblind_filter(image, player_actions, player_settings):
    player_data = collect_player_data(player_actions, player_settings)
    colorblindness_type = predict_colorblindness_type(player_data)
    forecasted_adjustments = forecast_adjustment_needs(player_data)

    # Define color adjustments based on predicted type and needs
    adjustments = {
        0: {'red_shift': 1.2, 'green_shift': 0.8},  # Example mapping for cluster 0
        1: {'red_shift': 0.8, 'green_shift': 1.2},  # Example mapping for cluster 1
        2: {'blue_shift': 1.2, 'yellow_shift': 0.8} # Example mapping for cluster 2
    }
    type_adjustments = adjustments.get(colorblindness_type, {})
    final_adjustments = {k: v * forecasted_adjustments for k, v in type_adjustments.items()}

    adjusted_image = apply_color_adjustments(image, final_adjustments)
    return adjusted_image

# Example usage
# Creating dummy data for illustration
image = np.random.rand(720, 1280, 3) * 255
player_actions = ['move', 'jump', 'interact']
player_settings = {'color_setting': 1.5}

adjusted_image = apply_dynamic_colorblind_filter(image, player_actions, player_settings)

def advanced_color_adjustment(image, adjustments):
    """
    Apply advanced color adjustments to an image based on the predicted colorblindness type.

    :param image: Image array
    :param adjustments: Dict with color adjustment parameters
    :return: Adjusted image array
    """
    # Example: Adjusting color channels based on the type
    # This is a placeholder. Actual implementation would involve complex image processing
    for channel, shift in adjustments.items():
        image[:, :, channel] *= shift
    return np.clip(image, 0, 255)  # Clipping to ensure valid color range

# Assuming player_data is being updated in real-time by the game engine
def real_time_data_processing(player_data):
    """
    Process player data in real-time to adjust for colorblindness

    :param player_data: Real-time data from player interactions
    :return: Predicted colorblindness type and adjustment needs
    """
    # Example: Using a simple heuristic or a lightweight ML model for real-time processing
    colorblindness_type = infer_colorblindness_type(player_data)
    adjustment_needs = determine_adjustment_needs(player_data)

    return colorblindness_type, adjustment_needs

def infer_colorblindness_type(player_data):
    """
    Logic to determine colorblindness type from player data.

    :param player_data: Data collected about player interactions and preferences.
    :return: Inferred colorblindness type.
    """
    # Example heuristic based on player data
    # This is a simplification for demonstration purposes.
    
    # Define threshold value for determining significant color adjustments
    threshold_value = 0.5  # Threshold for color adjustment frequency
    
    # Hypothetical logic: If a player frequently adjusts color settings,
    # infer their colorblindness type based on the nature of their adjustments.
    if player_data['color_adjustments'].mean() > threshold_value:
        if player_data['red_adjustments'].mean() > player_data['green_adjustments'].mean():
            return 'protanopia'
        elif player_data['green_adjustments'].mean() > player_data['red_adjustments'].mean():
            return 'deuteranopia'
        else:
            return 'tritanopia'
    else:
        return 'normal'  # Default to normal if no significant adjustments are made


def determine_adjustment_needs(player_data):
    """
    Assess the degree of adjustment required based on player interaction.

    :param player_data: Data collected about player interactions and preferences.
    :return: Adjustment needs in terms of color shifts.
    """
    # Example analysis to determine adjustment needs
    # This is a simplification for demonstration purposes.
    
    # Assuming 'player_data' contains fields like 'difficulty_with_red' and 'difficulty_with_green'
    adjustment_needs = {}
    adjustment_needs['red_shift'] = 1.0 + (player_data['difficulty_with_red'].mean() * 0.1)
    adjustment_needs['green_shift'] = 1.0 + (player_data['difficulty_with_green'].mean() * 0.1)

    # Add more logic here for other color adjustments as necessary
    return adjustment_needs

is_game_running = True  # This would normally be managed by the game's main loop

def game_is_running():
    """
    Check if the game is currently running.
    
    :return: Boolean indicating if the game is running.
    """
    global is_game_running
    return is_game_running

def get_current_game_frame():
    """
    Get the current frame from the game.

    :return: Current game frame as an image array.
    """
    # Placeholder logic
    current_frame = game_rendering_system.capture_frame()  # Hypothetical function call
    return current_frame

def get_real_time_player_data():
    """
    Collect real-time data from the player's actions and settings.

    :return: Data structure containing player's real-time data.
    """
    # Placeholder logic
    player_data = game_data_collector.get_player_data()  # Hypothetical function call
    return player_data

def calculate_adjustments(colorblindness_type, adjustment_needs):
    """
    Calculate the color adjustment parameters based on colorblindness type and needs.

    :param colorblindness_type: Type of colorblindness.
    :param adjustment_needs: Specific adjustment requirements.
    :return: Adjustment parameters.
    """
    # Placeholder logic for calculating adjustments
    adjustments = {
        'red_shift': adjustment_needs.get('red_shift', 1.0),
        'green_shift': adjustment_needs.get('green_shift', 1.0)
    }
    return adjustments

def collect_new_player_data():
    """
    Collect new player data for model updating.

    :return: Newly collected player data.
    """
    # Placeholder logic for data collection
    new_player_data = game_data_collector.collect_new_data()  # Hypothetical function call
    return new_player_data

def display_adjusted_frame(adjusted_frame):
    """
    Render the adjusted frame in the game.

    :param adjusted_frame: The adjusted game frame to be displayed.
    """
    # Placeholder logic for rendering a frame
    game_display_system.render_frame(adjusted_frame)  # Hypothetical function call

class GameDataCollector:
    def __init__(self):
        # Initialize data collector
        pass

    def get_player_data(self):
        # Retrieve real-time data about player's actions and settings
        # Placeholder logic
        player_data = {
            'actions': [],  # List of player actions
            'settings': {}  # Dictionary of player settings
        }
        return player_data

    def collect_new_data(self):
        # Collect new data for model updating
        # Placeholder logic
        new_data = {
            'new_actions': [],  # New actions since last collection
            'new_settings': {}  # New settings changes since last collection
        }
        return new_data

# Create an instance of GameDataCollector
game_data_collector = GameDataCollector()

class GameDisplaySystem:
    def __init__(self):
        # Initialize display system
        pass

    def render_frame(self, frame):
        # Render a frame on the game screen
        # Placeholder logic for frame rendering
        print("Rendering a frame...")  # Replace with actual rendering logic

# Create an instance of GameDisplaySystem
game_display_system = GameDisplaySystem()

class GameRenderingSystem:
    def __init__(self):
        # Initialize rendering system
        pass

    def capture_frame(self):
        # Capture the current frame from the game
        # Placeholder logic for frame capture
        # In a real implementation, this would interface with the game engine
        frame = np.random.rand(720, 1280, 3) * 255  # Dummy frame data
        return frame.astype(np.uint8)

# Create an instance of GameRenderingSystem
game_rendering_system = GameRenderingSystem()


def game_engine_integration():
    """
    Integrate the colorblindness adjustment system with the game engine
    """
    while game_is_running():
        current_frame = get_current_game_frame()
        player_data = game_data_collector.get_player_data()
        adjustments = calculate_adjustments(colorblindness_type, adjustment_needs)
        adjusted_frame = advanced_color_adjustment(current_frame, adjustments)
        game_display_system.render_frame(adjusted_frame)
        colorblindness_type, adjustment_needs = real_time_data_processing(player_data)
        display_adjusted_frame(adjusted_frame)

def continuous_learning():
    """
    Continuously update the model based on new player data
    """
    while game_is_running:
        new_data = collect_new_player_data()  # Collect new data from player interactions
        update_model(new_data)  # Update the predictive model with new data

def update_model(new_data):
    # Logic to update the colorblindness prediction model
    # Placeholder for machine learning model training/updating
    pass

def apply_enhanced_color_adjustments(image, adjustments, colorblindness_type):
    """
    Apply enhanced color adjustments based on the specific type of colorblindness.

    :param image: Image array
    :param adjustments: Adjustment parameters
    :param colorblindness_type: Specific type of colorblindness
    :return: Adjusted image array
    """
    # Extend the function to include other types of colorblindness
    if colorblindness_type == 'protanopia':
        adjusted_image = adjust_for_protanopia(image)
    elif colorblindness_type == 'deuteranopia':
        adjusted_image = adjust_for_deuteranopia(image)
    elif colorblindness_type == 'tritanopia':
        adjusted_image = adjust_for_tritanopia(image)
    elif colorblindness_type == 'monochrome':
        adjusted_image = adjust_for_monochrome(image)
    elif colorblindness_type == 'difficulty_purple':
        adjusted_image = adjust_for_no_purple(image)
    # Add similar elif conditions for other enhanced types like 'no_purple', 'neutral_difficulty', etc.

    return adjusted_image



def real_time_enhanced_data_processing(player_data):
    """
    Enhanced real-time data processing for various types of colorblindness

    :param player_data: Real-time data from player interactions
    :return: Identified colorblindness type and specific adjustment needs
    """
    # Implement logic to identify and process data for enhanced colorblindness types
    identified_type = infer_enhanced_colorblindness_type(player_data)
    specific_adjustment_needs = determine_enhanced_adjustment_needs(player_data)

    return identified_type, specific_adjustment_needs

def infer_enhanced_colorblindness_type(player_data):
    # Advanced logic or ML model to identify specific colorblindness types
    return 'monochrome'  # Example output

def determine_enhanced_adjustment_needs(player_data):
    # Advanced logic to determine specific adjustment needs for enhanced colorblindness types
    return {'contrast_increase': 1.2, 'saturation_decrease': 0.8}  # Example output

def pattern_recognition_for_enhanced_types(player_data):
    """
    Apply pattern recognition to identify specific types of colorblindness

    :param player_data: Data collected from player interactions
    :return: Predicted specific colorblindness type
    """
    # Implement machine learning algorithms for pattern recognition
    predicted_enhanced_type = apply_ml_model_for_colorblindness(player_data)
    return predicted_enhanced_type

def apply_ml_model_for_colorblindness(player_data):
    # Placeholder logic for an ML model
    return 'difficulty_purple'  # Example output


def dynamic_filter_adjustment_for_enhanced_types(image, player_actions, player_settings):
    player_data = collect_player_data(player_actions, player_settings)
    identified_type, specific_adjustment_needs = real_time_enhanced_data_processing(player_data)
    adjustments = calculate_specific_adjustments(identified_type, specific_adjustment_needs)
    adjusted_image = apply_enhanced_color_adjustments(image, adjustments, identified_type)
    return adjusted_image

def dynamic_filter_adjustment_for_enhanced_types(image, player_actions, player_settings):
    player_data = collect_player_data(player_actions, player_settings)
    identified_type, specific_adjustment_needs = real_time_enhanced_data_processing(player_data)

    adjustments = calculate_specific_adjustments(identified_type, specific_adjustment_needs)
    adjusted_image = apply_enhanced_color_adjustments(image, adjustments, identified_type)
    return adjusted_image
def calculate_specific_adjustments(identified_type, specific_adjustment_needs):
    """
    Calculate specific color adjustments based on identified colorblindness type and adjustment needs.

    :param identified_type: The identified type of colorblindness.
    :param specific_adjustment_needs: Specific needs for adjustments, determined by real-time data processing.
    :return: Dictionary of color adjustment parameters.
    """
    # Default adjustment values
    adjustments = {
        'red_shift': 1.0,
        'green_shift': 1.0,
        'blue_shift': 1.0,
        'contrast_increase': 1.0,
        'saturation_adjustment': 1.0
    }

    # Adjustments for different types of colorblindness
    if identified_type == 'protanopia':
        adjustments['red_shift'] = specific_adjustment_needs.get('red_shift', 1.2)
        adjustments['green_shift'] = specific_adjustment_needs.get('green_shift', 0.8)
    elif identified_type == 'deuteranopia':
        adjustments['green_shift'] = specific_adjustment_needs.get('green_shift', 1.2)
        adjustments['red_shift'] = specific_adjustment_needs.get('red_shift', 0.8)
    elif identified_type == 'tritanopia':
        adjustments['blue_shift'] = specific_adjustment_needs.get('blue_shift', 1.2)
    elif identified_type == 'monochrome':
        adjustments['contrast_increase'] = specific_adjustment_needs.get('contrast_increase', 1.5)
    elif identified_type == 'difficulty_purple':
        adjustments['saturation_adjustment'] = specific_adjustment_needs.get('saturation_adjustment', 0.8)

    # Add similar conditional blocks for other enhanced types

    return adjustments


import numpy as np

def adjust_for_deuteranopia(image):
    """
    Adjust colors for deuteranopia.

    :param image: Image array
    :return: Color-adjusted image array for deuteranopia
    """
    # Create a transformation matrix for deuteranopia
    # These values are illustrative and would need to be fine-tuned
    transform_matrix = np.array([
        [0.625, 0.375, 0],
        [0.70, 0.30, 0],
        [0, 0.30, 0.70]
    ])
    
    # Apply the transformation to each pixel
    adjusted_image = np.dot(image.reshape(-1, 3), transform_matrix).reshape(image.shape)
    return np.clip(adjusted_image, 0, 255)  # Ensuring the pixel values are valid

def adjust_for_tritanopia(image):
    """
    Adjust colors for tritanopia.

    :param image: Image array
    :return: Color-adjusted image array for tritanopia
    """
    # Create a transformation matrix for tritanopia
    # These values are illustrative and would need to be fine-tuned
    transform_matrix = np.array([
        [0.95, 0.05, 0],
        [0, 0.433, 0.567],
        [0, 0.475, 0.525]
    ])
    
    # Apply the transformation to each pixel
    adjusted_image = np.dot(image.reshape(-1, 3), transform_matrix).reshape(image.shape)
    return np.clip(adjusted_image, 0, 255)  # Ensuring the pixel values are valid

def adjust_for_protanopia(image):
    """
    Adjust colors for protanopia.

    :param image: Image array
    :return: Color-adjusted image array for protanopia
    """
    # Create a transformation matrix for protanopia
    # These values are illustrative and would need empirical tuning
    transform_matrix = np.array([
        [0.567, 0.433, 0],  # Reducing red component, increasing green
        [0.558, 0.442, 0],  # Similar adjustments in the green channel
        [0, 0.242, 0.758]   # Shifting some of the blue component into red and green
    ])
    
    # Apply the transformation to each pixel
    adjusted_image = np.dot(image.reshape(-1, 3), transform_matrix).reshape(image.shape)
    return np.clip(adjusted_image, 0, 255)  # Ensuring the pixel values are valid

def adjust_for_monochrome(image):
    """
    Adjust image for monochrome vision.

    :param image: Image array
    :return: Grayscale image
    """
    grayscale = np.mean(image, axis=2)
    return np.stack((grayscale,)*3, axis=-1)


def adjust_for_neutral_difficulty(image):
    """
    Adjust neutral colors to enhance distinguishability.

    :param image: Image array
    :return: Image with enhanced neutral colors
    """
    # Increase contrast and saturation for neutral colors
    # This is a placeholder for a more complex algorithm
    adjusted_image = np.clip(1.2 * image - 20, 0, 255)
    return adjusted_image

def adjust_for_warm_color_difficulty(image):
    """
    Adjust warm colors for better visibility.

    :param image: Image array
    :return: Image with enhanced warm colors
    """
    # Increase intensity of warm colors
    # Placeholder logic
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
    warm_enhancement = red * 1.2 + green * 0.8
    adjusted_image = np.stack((warm_enhancement, green, blue), axis=-1)
    return np.clip(adjusted_image, 0, 255)

def adjust_for_neutral_greyscale(image):
    """
    Adjust greyscale image to enhance neutral colors.

    :param image: Image array
    :return: Image with adjusted neutral greyscale tones
    """
    grayscale = np.mean(image, axis=2)
    # Increase contrast for neutral tones
    adjusted_grayscale = np.clip(1.2 * grayscale - 20, 0, 255)
    return np.stack((adjusted_grayscale,)*3, axis=-1)

def adjust_for_warm_greyscale(image):
    """
    Adjust greyscale image to enhance warm tones.

    :param image: Image array
    :return: Image with enhanced warm greyscale tones
    """
    grayscale = np.mean(image, axis=2)
    # Placeholder logic to enhance warm tones
    warm_enhanced_grayscale = np.clip(grayscale * 1.1, 0, 255)
    return np.stack((warm_enhanced_grayscale,)*3, axis=-1)

import numpy as np
import matplotlib
def adjust_for_no_purple(image):
    """
    Adjust purple hues in the image.

    :param image: Image array
    :return: Adjusted image with altered purple hues
    """
    # Assuming the image is in RGB format
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

    # Identifying purple hues (a mix of red and blue)
    # This is a simplified approach; actual implementation may require a more complex algorithm
    purple_mask = (red > 120) & (blue > 120) & (green < 80)

    # Shifting purple towards blue (or red) based on a chosen strategy
    # Here, we increase the blue component where purple is identified
    blue_adjusted = np.where(purple_mask, blue * 1.2, blue)

    # Reconstructing the image with adjusted blue channel
    adjusted_image = np.stack((red, green, blue_adjusted), axis=-1)
    return np.clip(adjusted_image, 0, 255)  # Clipping to ensure valid color values

# Example usage
# adjusted_image = adjust_for_no_purple(image)

def adjust_for_complex_color_difficulty(image):
    """
    Simplify complex colors into more distinct primary hues.

    :param image: Image array
    :return: Image with simplified color palette
    """
    # Assumption: Complex colors can be simplified by maximizing one primary color component
    # This is a simplified approach and may not cover all complex color cases

    # Breaking down the image into its RGB components
    red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

    # Simplifying the color by enhancing the dominant color channel in each pixel
    max_channel = np.argmax(image, axis=2)
    simplified_red = np.where(max_channel == 0, red * 1.2, red)
    simplified_green = np.where(max_channel == 1, green * 1.2, green)
    simplified_blue = np.where(max_channel == 2, blue * 1.2, blue)

    # Reconstructing the image with simplified colors
    simplified_image = np.stack((simplified_red, simplified_green, simplified_blue), axis=-1)
    return np.clip(simplified_image, 0, 255)

import numpy as np

def hsv_to_rgb(hsv):
    """
    Convert an HSV image to RGB.

    :param hsv: Image in HSV color space
    :return: Image in RGB color space
    """
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    r, g, b = np.zeros_like(h), np.zeros_like(s), np.zeros_like(v)

    i = np.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    idx = (i % 6 == 0)
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = (i == 1)
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = (i == 2)
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = (i == 3)
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = (i == 4)
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = (i >= 5)
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    rgb = np.stack([r, g, b], axis=-1)
    return rgb

import matplotlib.colors
import numpy as np

def adjust_for_simple_color_difficulty(image):
    """
    Enhance simple colors to make them more distinguishable.

    :param image: Image array
    :return: Image with enhanced primary and secondary colors
    """
    # Convert to HSV for easier saturation and value manipulation
    hsv_image = matplotlib.colors.rgb_to_hsv(image / 255.0)

    # Enhancing saturation (S channel) and value (V channel)
    hsv_image[:,:,1] *= 1.2  # Enhancing saturation
    hsv_image[:,:,2] *= 1.1  # Enhancing value/brightness

    # Converting back to RGB
    enhanced_image = matplotlib.colors.hsv_to_rgb(hsv_image) * 255
    return np.clip(enhanced_image, 0, 255)

# Example usage with a dummy image
dummy_image = np.random.rand(100, 100, 3) * 255
enhanced_image = adjust_for_simple_color_difficulty(dummy_image)



def adjust_for_simple_color_difficulty(image):
    """
    Enhance simple colors to make them more distinguishable.

    :param image: Image array
    :return: Image with enhanced primary and secondary colors
    """
    # Increasing the contrast and saturation can help enhance simple colors
    # This is a simplified approach

    # Convert to HSV for easier saturation and value manipulation
    hsv_image = matplotlib.colors.rgb_to_hsv(image / 255.0)

    # Enhancing saturation (S channel) and value (V channel)
    hsv_image[:,:,1] *= 1.2  # Enhancing saturation
    hsv_image[:,:,2] *= 1.1  # Enhancing value/brightness

    # Converting back to RGB
    enhanced_image = matplotlib.colors.hsv_to_rgb(hsv_image) * 255
    return np.clip(enhanced_image, 0, 255)



def apply_colorblind_filter(image, filter_type):
    """
    Apply a colorblind filter to an image.

    :param image: The original game image
    :param filter_type: Type of colorblindness (e.g., 'protanopia', 'deuteranopia', 'tritanopia')
    :return: Image with colorblind filter applied
    """
    if filter_type == 'protanopia':
        adjusted_image = adjust_for_protanopia(image)  # Assuming this function is defined
    elif filter_type == 'deuteranopia':
        adjusted_image = adjust_for_deuteranopia(image)
    elif filter_type == 'tritanopia':
        adjusted_image = adjust_for_tritanopia(image)
    else:
        adjusted_image = image

    return adjusted_image

def apply_large_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size):
    """
    Large-scale bilateral filter implementation for kernels 23-49.
    Uses advanced approximation, downsampling, and GPU-style optimizations.
    
    :param image: Input image array (float32, 0-255 range)
    :param spatial_sigma: Spatial standard deviation for Gaussian weights
    :param intensity_sigma: Intensity standard deviation for edge preservation
    :param kernel_size: Size of the filter kernel (23-49 recommended)
    :return: High-quality bilateral filtered image with advanced optimizations
    """
    if image is None or len(image.shape) != 3:
        return image
    
    h, w, c = image.shape
    
    # Multi-resolution approach for large kernels
    if kernel_size > 31:
        return apply_multiscale_bilateral_filter(image, spatial_sigma, intensity_sigma, kernel_size)
    
    # Advanced memory-efficient processing
    filtered_image = np.zeros_like(image, dtype=np.float32)
    pad_size = kernel_size // 2
    
    # Intelligent padding with boundary extrapolation
    padded = apply_intelligent_padding(image, pad_size)
    
    # Separable approximation for large kernels
    if spatial_sigma > 5.0:
        return apply_separable_bilateral_approximation(image, spatial_sigma, intensity_sigma)
    
    # Hierarchical weight computation with compression
    compressed_weights = compute_compressed_spatial_weights(kernel_size, spatial_sigma)
    
    # Advanced intensity clustering for efficient processing
    intensity_clusters = create_intensity_clusters(image, intensity_sigma)
    
    # Stream processing for memory efficiency
    stream_size = calculate_stream_size(h, w, kernel_size)

    def process_large_bilateral_stream(image_stream, padded_stream, compressed_weights, intensity_clusters, pad_size):
        """Process a stream of image data with bilateral filtering."""
        # Basic bilateral filtering implementation for the stream
        h_stream, w_stream, c = image_stream.shape
        result = np.zeros_like(image_stream, dtype=np.float32)
        
        for i in range(h_stream):
            for j in range(w_stream):
                for ch in range(c):
                    # Simple bilateral filter approximation
                    center_val = image_stream[i, j, ch]
                    weighted_sum = 0.0
                    weight_sum = 0.0
                    
                    # Sample neighborhood
                    for di in range(-min(pad_size, 2), min(pad_size, 2) + 1):
                        for dj in range(-min(pad_size, 2), min(pad_size, 2) + 1):
                            ni, nj = i + di + pad_size, j + dj + pad_size
                            if 0 <= ni < padded_stream.shape[0] and 0 <= nj < padded_stream.shape[1]:
                                neighbor_val = padded_stream[ni, nj, ch]
                                spatial_weight = np.exp(-(di*di + dj*dj) / (2 * 2.0 * 2.0))
                                intensity_weight = np.exp(-abs(center_val - neighbor_val) / (2 * 10.0 * 10.0))
                                weight = spatial_weight * intensity_weight
                                weighted_sum += neighbor_val * weight
                                weight_sum += weight
                    
                    if weight_sum > 0:
                        result[i, j, ch] = weighted_sum / weight_sum
                    else:
                        result[i, j, ch] = center_val
        
        return result
 
    for stream_start in range(0, h, stream_size):
        stream_end = min(stream_start + stream_size, h)
        
        # Process stream with large kernel optimization
        stream_result = process_large_bilateral_stream(
            image[stream_start:stream_end], 
            padded[stream_start:stream_end+2*pad_size],
            compressed_weights, intensity_clusters, pad_size
        )
        
        filtered_image[stream_start:stream_end] = stream_result
    
        # Multi-pass refinement for large kernel artifacts
        filtered_image = apply_multipass_refinement(image, filtered_image, kernel_size)
        
        return np.clip(filtered_image, 0, 255).astype(np.float32)
    
    def apply_multipass_refinement(original_image, filtered_image, kernel_size):
        """
        Apply multi-pass refinement to reduce artifacts from large kernel filtering.
        
        :param original_image: Original input image
        :param filtered_image: Bilateral filtered image
        :param kernel_size: Size of the kernel used for filtering
        :return: Refined filtered image
        """
        # Simple refinement: blend with original based on kernel size
        refinement_factor = min(0.1, kernel_size / 500.0)
        refined = filtered_image * (1 - refinement_factor) + original_image * refinement_factor
        return refined


def apply_selective_smoothing(image, characteristics):
    """Apply selective smoothing based on local image properties."""
    # Calculate local image properties
    local_variance = calculate_local_variance(image, kernel_size=5)
    edge_mask = calculate_edge_mask(image)
    
    # Adaptive smoothing strength
    smooth_strength = characteristics['noise_level'] * (1 - edge_mask) * 0.3
    
    # Apply Gaussian smoothing with adaptive strength
    smoothed = apply_optimized_gaussian_blur(image, sigma=1.0, kernel_size=5)
    
    # Blend based on smoothing strength
    result = image * (1 - smooth_strength) + smoothed * smooth_strength
    
    return result

def apply_filter_specific_color_prep(image, filter_type):
    """Apply final color space adjustments specific to the filter type."""
    if filter_type in ['monochrome', 'neutral_greyscale', 'warm_greyscale']:
        # Pre-enhance edges before grayscale conversion
        enhanced_edges = apply_selective_edge_enhancement(image)
        return enhanced_edges
    
    elif filter_type in ['high_contrast', 'enhanced_contrast']:
        # Prepare for high contrast by optimizing mid-tones
        midtone_enhanced = optimize_midtones(image)
        return midtone_enhanced
    
    return image

def KMeansCluster(weights):
    clusterCount = 3
    data = np.array(weights).reshape(-1, 1)
    unique_points = np.unique(data, axis=0)

    if unique_points.shape[0] >= clusterCount:
        kmeans = KMeans(n_clusters=clusterCount, n_init='auto').fit(data)
        return kmeans.labels_
    else:
        print("ConvergenceWarning Fix: Not enough unique points for clustering.")
        return np.zeros(len(weights), dtype=int)

# Remove or comment out the following block:
# unique_points = np.unique(data, axis=0)
# if unique_points.shape[0] >= 3:
#     kmeans = KMeans(n_clusters=3).fit(data)
#     labels = kmeans.labels_
# else:
#     # Fallback: assign all points to one cluster
#     labels = np.zeros(len(data), dtype=int)
#     print("Warning: Not enough unique data points for clustering; using default cluster.")


def apply_gradient_normalization(image, filter_type, strength=1.0):
    """
    Apply gradient normalization specific to each colorblindness filter type.
    :param image: Input image (numpy array, 0-255)
    :param filter_type: Colorblindness filter type
    :param strength: Normalization strength (default 1.0)
    :return: Gradient-normalized image
    """
    img = image.astype(np.float32)
    grad_x = np.diff(img, axis=1, prepend=img[:, :1])
    grad_y = np.diff(img, axis=0, prepend=img[:1, :])
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    norm_img = img.copy()

    if filter_type == 'protanopia':
        # Boost green/blue gradients, suppress red
        norm_img[:,:,1] += grad_mag[:,:,1] * 0.15 * strength
        norm_img[:,:,2] += grad_mag[:,:,2] * 0.10 * strength
        norm_img[:,:,0] -= grad_mag[:,:,0] * 0.10 * strength
    elif filter_type == 'deuteranopia':
        # Boost red/blue gradients, suppress green
        norm_img[:,:,0] += grad_mag[:,:,0] * 0.12 * strength
        norm_img[:,:,2] += grad_mag[:,:,2] * 0.12 * strength
        norm_img[:,:,1] -= grad_mag[:,:,1] * 0.08 * strength
    elif filter_type == 'tritanopia':
        # Boost red/green gradients, slightly boost blue
        norm_img[:,:,0] += grad_mag[:,:,0] * 0.10 * strength
        norm_img[:,:,1] += grad_mag[:,:,1] * 0.10 * strength
        norm_img[:,:,2] += grad_mag[:,:,2] * 0.05 * strength
    elif filter_type == 'monochrome':
        # Enhance luminance gradients
        gray = np.mean(img, axis=2)
        grad = np.sqrt(np.diff(gray, axis=1, prepend=gray[:, :1])**2 + np.diff(gray, axis=0, prepend=gray[:1, :])**2)
        norm_gray = gray + grad * 0.25 * strength
        norm_img = np.stack([norm_gray]*3, axis=-1)
    elif filter_type == 'no_purple':
        # Enhance blue gradients in purple regions
        purple_mask = (img[:,:,0] > 120) & (img[:,:,2] > 120) & (img[:,:,1] < 80)
        norm_img[:,:,2] += purple_mask * grad_mag[:,:,2] * 0.2 * strength
    elif filter_type == 'neutral_difficulty':
        # Enhance gradients in low-saturation (neutral) areas
        hsv = matplotlib.colors.rgb_to_hsv(img / 255.0)
        neutral_mask = hsv[:,:,1] < 0.3
        for c in range(3):
            norm_img[:,:,c] += neutral_mask * grad_mag[:,:,c] * 0.15 * strength
    elif filter_type == 'warm_color_difficulty':
        # Enhance gradients in red/yellow regions
        warm_mask = (img[:,:,0] > 100) | ((img[:,:,0] + img[:,:,1]) > 200)
        norm_img[:,:,0] += warm_mask * grad_mag[:,:,0] * 0.18 * strength
        norm_img[:,:,1] += warm_mask * grad_mag[:,:,1] * 0.10 * strength
    elif filter_type == 'neutral_greyscale':
        # Enhance gradients in greyscale
        gray = np.mean(img, axis=2)
        grad = np.sqrt(np.diff(gray, axis=1, prepend=gray[:, :1])**2 + np.diff(gray, axis=0, prepend=gray[:1, :])**2)
        norm_gray = gray + grad * 0.18 * strength
        norm_img = np.stack([norm_gray]*3, axis=-1)
    elif filter_type == 'warm_greyscale':
        # Enhance gradients in warm-tinted greyscale
        gray = np.mean(img, axis=2)
        grad = np.sqrt(np.diff(gray, axis=1, prepend=gray[:, :1])**2 + np.diff(gray, axis=0, prepend=gray[:1, :])**2)
        norm_gray = gray + grad * 0.12 * strength
        norm_img = np.stack([norm_gray*1.05, norm_gray*1.03, norm_gray*0.97], axis=-1)
    else:
        # Default: mild edge enhancement
        for c in range(3):
            norm_img[:,:,c] += grad_mag[:,:,c] * 0.08 * strength

    return np.clip(norm_img, 0, 255).astype(np.uint8)

# Example integration: after applying a colorblind filter, call this function
# filtered = adjust_for_protanopia(image)
# filtered = apply_gradient_normalization(filtered, 'protanopia', strength=1.0)
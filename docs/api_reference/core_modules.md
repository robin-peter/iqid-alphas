# API Reference: Core Modules

This document provides detailed API documentation for the core IQID-Alphas modules.

## Table of Contents

- [IQID Processing Object](#iqid-processing-object)
- [IQID Helper Functions](#iqid-helper-functions)
- [Image Alignment](#image-alignment)
- [DPK (Dose Point Kernel) Functions](#dpk-dose-point-kernel-functions)
- [Specification and Validation](#specification-and-validation)

## IQID Processing Object

### `src.core.iqid.process_object`

The main processing class for iQID image analysis.

#### Class: `IQIDProcessor`

```python
class IQIDProcessor:
    """Main processor for iQID image analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the processor with configuration."""
```

#### Methods

##### `load_image(image_path: str) -> np.ndarray`

Loads an image from the specified path.

**Parameters:**
- `image_path` (str): Path to the image file

**Returns:**
- `np.ndarray`: Loaded image array

**Raises:**
- `FileNotFoundError`: If image file doesn't exist
- `ValueError`: If image format is not supported

**Example:**
```python
processor = IQIDProcessor(config)
image = processor.load_image("path/to/image.tif")
```

##### `preprocess_image(image: np.ndarray, **kwargs) -> np.ndarray`

Applies preprocessing steps to the input image.

**Parameters:**
- `image` (np.ndarray): Input image
- `**kwargs`: Additional preprocessing parameters

**Returns:**
- `np.ndarray`: Preprocessed image

**Example:**
```python
preprocessed = processor.preprocess_image(
    image, 
    gaussian_blur_sigma=1.0,
    median_filter_size=3
)
```

##### `segment_tissue(image: np.ndarray, method: str = 'otsu') -> np.ndarray`

Segments tissue regions from the image.

**Parameters:**
- `image` (np.ndarray): Input image
- `method` (str): Segmentation method ('otsu', 'li', 'triangle', 'adaptive')

**Returns:**
- `np.ndarray`: Binary segmentation mask

**Example:**
```python
tissue_mask = processor.segment_tissue(image, method='otsu')
```

##### `segment_activity(image: np.ndarray, **kwargs) -> np.ndarray`

Segments activity/hotspot regions from the image.

**Parameters:**
- `image` (np.ndarray): Input image
- `**kwargs`: Segmentation parameters

**Returns:**
- `np.ndarray`: Binary activity mask

##### `quantify_regions(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]`

Quantifies intensity values within segmented regions.

**Parameters:**
- `image` (np.ndarray): Input image
- `mask` (np.ndarray): Binary mask defining regions

**Returns:**
- `Dict[str, float]`: Quantification results with metrics

**Example:**
```python
results = processor.quantify_regions(image, tissue_mask)
# Returns: {'mean_intensity': 1250.5, 'total_activity': 50000, ...}
```

## IQID Helper Functions

### `src.core.iqid.helper`

Utility functions for image processing and analysis.

#### Functions

##### `normalize_image(image: np.ndarray, method: str = 'minmax') -> np.ndarray`

Normalizes image intensity values.

**Parameters:**
- `image` (np.ndarray): Input image
- `method` (str): Normalization method ('minmax', 'zscore', 'quantile')

**Returns:**
- `np.ndarray`: Normalized image

##### `apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray`

Applies Gaussian blur to reduce noise.

**Parameters:**
- `image` (np.ndarray): Input image
- `sigma` (float): Standard deviation for Gaussian kernel

**Returns:**
- `np.ndarray`: Blurred image

##### `enhance_contrast(image: np.ndarray, method: str = 'clahe') -> np.ndarray`

Enhances image contrast.

**Parameters:**
- `image` (np.ndarray): Input image
- `method` (str): Enhancement method ('clahe', 'histogram_eq', 'adaptive')

**Returns:**
- `np.ndarray`: Contrast-enhanced image

##### `remove_background(image: np.ndarray, method: str = 'rolling_ball', radius: int = 50) -> np.ndarray`

Removes background from image.

**Parameters:**
- `image` (np.ndarray): Input image
- `method` (str): Background removal method
- `radius` (int): Radius for background estimation

**Returns:**
- `np.ndarray`: Background-corrected image

## Image Alignment

### `src.core.iqid.align`

Functions for image registration and alignment.

#### Functions

##### `register_images(fixed_image: np.ndarray, moving_image: np.ndarray, method: str = 'phase_correlation') -> Tuple[np.ndarray, Dict]`

Registers two images and returns the transformation.

**Parameters:**
- `fixed_image` (np.ndarray): Reference image
- `moving_image` (np.ndarray): Image to be aligned
- `method` (str): Registration method

**Returns:**
- `Tuple[np.ndarray, Dict]`: Transformed image and transformation parameters

##### `apply_transformation(image: np.ndarray, transform_params: Dict) -> np.ndarray`

Applies transformation parameters to an image.

**Parameters:**
- `image` (np.ndarray): Input image
- `transform_params` (Dict): Transformation parameters

**Returns:**
- `np.ndarray`: Transformed image

##### `calculate_alignment_quality(fixed: np.ndarray, aligned: np.ndarray) -> Dict[str, float]`

Calculates quality metrics for image alignment.

**Parameters:**
- `fixed` (np.ndarray): Reference image
- `aligned` (np.ndarray): Aligned image

**Returns:**
- `Dict[str, float]`: Quality metrics

## DPK (Dose Point Kernel) Functions

### `src.core.iqid.dpk`

Functions for dose point kernel calculations and analysis.

#### Functions

##### `calculate_dpk(image: np.ndarray, kernel_params: Dict) -> np.ndarray`

Calculates dose point kernel for the image.

**Parameters:**
- `image` (np.ndarray): Input activity image
- `kernel_params` (Dict): DPK parameters

**Returns:**
- `np.ndarray`: Dose distribution

##### `generate_kernel(size: int, alpha_energy: float, tissue_type: str = 'soft') -> np.ndarray`

Generates a dose point kernel for specific alpha energy.

**Parameters:**
- `size` (int): Kernel size in pixels
- `alpha_energy` (float): Alpha particle energy in MeV
- `tissue_type` (str): Tissue type for density correction

**Returns:**
- `np.ndarray`: Dose point kernel

## Specification and Validation

### `src.core.iqid.spec`

Functions for data validation and specification checking.

#### Functions

##### `validate_image_format(image_path: str) -> bool`

Validates if image file format is supported.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `bool`: True if format is supported

##### `validate_config(config: Dict) -> Tuple[bool, List[str]]`

Validates configuration parameters.

**Parameters:**
- `config` (Dict): Configuration dictionary

**Returns:**
- `Tuple[bool, List[str]]`: Validation status and error messages

##### `check_image_compatibility(image1: np.ndarray, image2: np.ndarray) -> bool`

Checks if two images are compatible for processing.

**Parameters:**
- `image1`, `image2` (np.ndarray): Images to compare

**Returns:**
- `bool`: True if images are compatible

## Error Handling

All functions raise appropriate exceptions:

- `ValueError`: For invalid parameters or data
- `FileNotFoundError`: For missing files
- `RuntimeError`: For processing errors
- `MemoryError`: For insufficient memory

## Configuration Integration

All core modules integrate with the configuration system:

```python
from src.core.iqid.process_object import IQIDProcessor
import json

# Load configuration
with open('configs/iqid_pipeline_config.json', 'r') as f:
    config = json.load(f)

# Initialize processor
processor = IQIDProcessor(config)
```

## Performance Considerations

- **Memory Usage**: Large images may require chunked processing
- **Processing Time**: Complex segmentation methods take longer
- **Parallel Processing**: Most functions support batch processing
- **Caching**: Results can be cached for repeated operations

## Version Compatibility

- Minimum Python version: 3.8
- Recommended Python version: 3.11
- NumPy version: >= 1.20.0
- SciPy version: >= 1.7.0
- scikit-image version: >= 0.18.0

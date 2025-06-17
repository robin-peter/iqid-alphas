# API Reference: Pipeline Classes

This document provides detailed API documentation for the IQID-Alphas pipeline classes.

## Table of Contents

- [SimpleiQIDPipeline](#simpleiqdpipeline)
- [iQIDProcessingPipeline](#iqidprocessingpipeline)
- [CombinedHEiQIDPipeline](#combinedheiqidpipeline)
- [Configuration Management](#configuration-management)
- [Batch Processing](#batch-processing)

## SimpleiQIDPipeline

### `pipelines.simplified_iqid_pipeline.SimpleiQIDPipeline`

A streamlined pipeline for basic iQID image processing.

#### Class Definition

```python
class SimpleiQIDPipeline:
    """Simplified pipeline for iQID image processing with essential steps."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
```

#### Constructor Parameters

- `config_path` (str): Path to configuration JSON file

#### Methods

##### `process_single_image(image_path: str, output_dir: str) -> Dict[str, Any]`

Processes a single iQID image through the complete pipeline.

**Parameters:**
- `image_path` (str): Path to the input image
- `output_dir` (str): Directory for output files

**Returns:**
- `Dict[str, Any]`: Processing results and metrics

**Example:**
```python
pipeline = SimpleiQIDPipeline('configs/iqid_pipeline_config.json')
results = pipeline.process_single_image('data/image.tif', 'outputs/')
```

##### `process_batch(input_dir: str, output_dir: str) -> Dict[str, Any]`

Processes multiple images in batch mode.

**Parameters:**
- `input_dir` (str): Directory containing input images
- `output_dir` (str): Directory for output files

**Returns:**
- `Dict[str, Any]`: Batch processing summary

##### `validate_inputs(image_path: str) -> bool`

Validates input image before processing.

**Parameters:**
- `image_path` (str): Path to input image

**Returns:**
- `bool`: True if validation passes

## iQIDProcessingPipeline

### `pipelines.iqid_only_pipeline.iQIDProcessingPipeline`

Advanced pipeline for comprehensive iQID image analysis.

#### Class Definition

```python
class iQIDProcessingPipeline:
    """Advanced iQID processing pipeline with full feature set."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
```

#### Methods

##### `full_analysis(image_path: str, output_dir: str, **kwargs) -> Dict[str, Any]`

Performs complete analysis including segmentation, quantification, and DPK.

**Parameters:**
- `image_path` (str): Path to input image
- `output_dir` (str): Output directory
- `**kwargs`: Additional processing parameters

**Returns:**
- `Dict[str, Any]`: Comprehensive analysis results

**Example:**
```python
pipeline = iQIDProcessingPipeline('configs/iqid_pipeline_config.json')
results = pipeline.full_analysis(
    'data/image.tif', 
    'outputs/',
    enable_dpk=True,
    generate_visualizations=True
)
```

##### `segment_and_quantify(image: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict]`

Performs segmentation and quantification in one step.

**Parameters:**
- `image` (np.ndarray): Input image
- `**kwargs`: Segmentation parameters

**Returns:**
- `Tuple[np.ndarray, Dict]`: Segmentation mask and quantification results

##### `calculate_dose_distribution(activity_image: np.ndarray, **kwargs) -> np.ndarray`

Calculates dose distribution using dose point kernels.

**Parameters:**
- `activity_image` (np.ndarray): Activity distribution image
- `**kwargs`: DPK parameters

**Returns:**
- `np.ndarray`: Dose distribution map

##### `generate_report(results: Dict, output_path: str) -> None`

Generates comprehensive analysis report.

**Parameters:**
- `results` (Dict): Analysis results
- `output_path` (str): Path for output report

## CombinedHEiQIDPipeline

### `pipelines.combined_he_iqid_pipeline.CombinedHEiQIDPipeline`

Pipeline for combined H&E and iQID image analysis with alignment.

#### Class Definition

```python
class CombinedHEiQIDPipeline:
    """Combined pipeline for H&E and iQID image co-analysis."""
    
    def __init__(self, config_path: str):
        """Initialize combined pipeline."""
```

#### Methods

##### `process_sample(he_path: str, iqid_path: str, output_dir: str) -> Dict[str, Any]`

Processes a paired H&E and iQID sample.

**Parameters:**
- `he_path` (str): Path to H&E image
- `iqid_path` (str): Path to iQID image
- `output_dir` (str): Output directory

**Returns:**
- `Dict[str, Any]`: Combined analysis results

**Example:**
```python
pipeline = CombinedHEiQIDPipeline('configs/combined_pipeline_config.json')
results = pipeline.process_sample(
    'data/he_image.tif',
    'data/iqid_image.tif',
    'outputs/sample_001/'
)
```

##### `align_images(he_image: np.ndarray, iqid_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]`

Aligns H&E and iQID images.

**Parameters:**
- `he_image` (np.ndarray): H&E image
- `iqid_image` (np.ndarray): iQID image

**Returns:**
- `Tuple[np.ndarray, np.ndarray, Dict]`: Aligned images and transformation parameters

##### `combined_segmentation(he_image: np.ndarray, iqid_image: np.ndarray) -> Dict[str, np.ndarray]`

Performs segmentation using both image modalities.

**Parameters:**
- `he_image` (np.ndarray): H&E image
- `iqid_image` (np.ndarray): iQID image

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary of segmentation masks

##### `cross_modal_analysis(he_results: Dict, iqid_results: Dict) -> Dict[str, Any]`

Performs cross-modal analysis between H&E and iQID results.

**Parameters:**
- `he_results` (Dict): H&E analysis results
- `iqid_results` (Dict): iQID analysis results

**Returns:**
- `Dict[str, Any]`: Cross-modal analysis results

##### `process_batch_samples(input_dir: str, output_dir: str) -> Dict[str, Any]`

Processes multiple paired samples in batch mode.

**Parameters:**
- `input_dir` (str): Directory with paired samples
- `output_dir` (str): Output directory

**Returns:**
- `Dict[str, Any]`: Batch processing summary

## Configuration Management

### Configuration Loading

All pipelines support configuration management:

```python
import json

# Load configuration
with open('config_path.json', 'r') as f:
    config = json.load(f)

# Override specific parameters
config['processing']['gaussian_blur_sigma'] = 1.5

# Initialize pipeline with modified config
pipeline = SimpleiQIDPipeline(config)
```

### Dynamic Configuration Updates

```python
# Update configuration during runtime
pipeline.update_config({
    'segmentation': {
        'tissue': {'method': 'li'}
    }
})
```

## Batch Processing

### Batch Configuration

All pipelines support batch processing with progress tracking:

```python
# Batch processing with progress callback
def progress_callback(current, total, sample_name):
    print(f"Processing {sample_name}: {current}/{total}")

results = pipeline.process_batch(
    input_dir='data/batch/',
    output_dir='outputs/batch/',
    progress_callback=progress_callback,
    max_workers=4  # Parallel processing
)
```

### Batch Results Structure

```python
{
    'summary': {
        'total_samples': 10,
        'successful': 9,
        'failed': 1,
        'processing_time': 1200.5
    },
    'individual_results': {
        'sample_001': {...},
        'sample_002': {...},
        ...
    },
    'failed_samples': ['sample_005'],
    'batch_statistics': {
        'mean_processing_time': 120.05,
        'mean_tissue_area': 15000.2,
        ...
    }
}
```

## Error Handling

### Exception Types

All pipelines raise specific exceptions:

```python
from pipelines.exceptions import (
    PipelineConfigError,
    ImageProcessingError,
    AlignmentError,
    SegmentationError
)

try:
    results = pipeline.process_single_image(image_path, output_dir)
except PipelineConfigError as e:
    print(f"Configuration error: {e}")
except ImageProcessingError as e:
    print(f"Processing error: {e}")
```

### Error Recovery

```python
# Enable error recovery mode
pipeline.set_error_recovery(True)

# Process with automatic retry
results = pipeline.process_with_retry(
    image_path,
    output_dir,
    max_retries=3,
    retry_delay=1.0
)
```

## Performance Optimization

### Memory Management

```python
# Enable memory optimization
pipeline.enable_memory_optimization(
    chunk_size=1024,
    use_memory_mapping=True,
    garbage_collect_frequency=10
)
```

### Parallel Processing

```python
# Configure parallel processing
pipeline.set_parallel_config(
    max_workers=4,
    use_processes=True,  # vs threads
    chunk_size='auto'
)
```

## Logging and Monitoring

### Pipeline Logging

```python
import logging

# Configure pipeline logging
pipeline.configure_logging(
    level=logging.INFO,
    log_file='pipeline.log',
    include_performance_metrics=True
)
```

### Progress Monitoring

```python
# Monitor processing progress
monitor = pipeline.create_progress_monitor()
results = pipeline.process_batch(
    input_dir,
    output_dir,
    progress_monitor=monitor
)

# Get processing statistics
stats = monitor.get_statistics()
```

## Integration Examples

### Custom Pipeline Extension

```python
class CustomPipeline(SimpleiQIDPipeline):
    """Custom pipeline with additional processing steps."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.custom_processor = CustomProcessor()
    
    def process_single_image(self, image_path: str, output_dir: str) -> Dict:
        # Call parent method
        results = super().process_single_image(image_path, output_dir)
        
        # Add custom processing
        custom_results = self.custom_processor.process(results)
        results.update(custom_results)
        
        return results
```

### Pipeline Chaining

```python
# Chain multiple pipelines
simple_pipeline = SimpleiQIDPipeline(config1)
advanced_pipeline = iQIDProcessingPipeline(config2)

# Process with first pipeline
intermediate_results = simple_pipeline.process_single_image(image_path, temp_dir)

# Continue with second pipeline
final_results = advanced_pipeline.process_results(
    intermediate_results,
    output_dir
)
```

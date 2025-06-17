# User Guide: Getting Started with IQID-Alphas

## Introduction

Welcome to IQID-Alphas, a comprehensive pipeline for quantitative imaging analysis. This guide will help you get started with processing your iQID and H&E images.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher (Python 3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for large images)
- **Storage**: 10GB free space for outputs
- **OS**: Linux, macOS, or Windows

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd iqid-alphas
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python evaluation/scripts/production_validation.py
   ```

## Understanding Your Data

### Supported File Formats
- **iQID Images**: TIFF format (.tif, .tiff)
- **H&E Images**: TIFF, PNG, JPEG formats
- **Batch Processing**: Directory structure with organized samples

### Data Organization
Organize your data in the following structure:
```
your_data/
├── sample_001/
│   ├── iqid/
│   │   └── image.tif
│   └── he/
│       └── image.tif
├── sample_002/
│   ├── iqid/
│   │   └── image.tif
│   └── he/
│       └── image.tif
└── ...
```

## Basic Usage

### 1. Simple iQID Processing

For basic iQID image processing:

```python
from pipelines.simplified_iqid_pipeline import SimpleiQIDPipeline

# Initialize pipeline
pipeline = SimpleiQIDPipeline()

# Process a single image
results = pipeline.process_sample('path/to/your/iqid_image.tif')

# Access results
segmentation_mask = results['segmentation_mask']
processed_image = results['processed_image']
metrics = results['metrics']
```

### 2. Advanced iQID Processing

For advanced processing with decay correction and detailed analysis:

```python
from pipelines.iqid_only_pipeline import iQIDProcessingPipeline

# Initialize pipeline with custom config
pipeline = iQIDProcessingPipeline('configs/iqid_pipeline_config.json')

# Process with additional analysis
results = pipeline.process_sample('path/to/your/iqid_image.tif')

# Access advanced results
decay_corrected = results['decay_corrected']
activity_map = results['activity_map']
quantitative_metrics = results['quantitative_metrics']
```

### 3. Combined H&E + iQID Processing

For multi-modal processing with alignment:

```python
from pipelines.combined_he_iqid_pipeline import CombinedHEiQIDPipeline

# Initialize combined pipeline
pipeline = CombinedHEiQIDPipeline('configs/combined_pipeline_config.json')

# Process paired images
results = pipeline.process_sample(
    iqid_path='path/to/iqid_image.tif',
    he_path='path/to/he_image.tif'
)

# Access combined results
aligned_images = results['aligned_images']
combined_mask = results['combined_mask']
registration_metrics = results['registration_metrics']
```

## Batch Processing

### Processing Multiple Samples

```python
from pipelines.combined_he_iqid_pipeline import CombinedHEiQIDPipeline
import os

pipeline = CombinedHEiQIDPipeline()

# Define your data directory
data_dir = 'path/to/your/data'
output_dir = 'path/to/outputs'

# Process all samples in directory
for sample_dir in os.listdir(data_dir):
    sample_path = os.path.join(data_dir, sample_dir)
    if os.path.isdir(sample_path):
        iqid_path = os.path.join(sample_path, 'iqid', 'image.tif')
        he_path = os.path.join(sample_path, 'he', 'image.tif')
        
        if os.path.exists(iqid_path) and os.path.exists(he_path):
            print(f"Processing {sample_dir}...")
            results = pipeline.process_sample(iqid_path, he_path)
            
            # Save results
            sample_output_dir = os.path.join(output_dir, sample_dir)
            pipeline.save_results(results, sample_output_dir)
```

## Understanding Results

### Output Structure
Each processed sample generates:
```
outputs/
├── sample_001/
│   ├── segmentation/
│   │   ├── tissue_mask.tif
│   │   ├── activity_mask.tif
│   │   └── combined_mask.tif
│   ├── alignment/
│   │   ├── aligned_he.tif
│   │   ├── aligned_iqid.tif
│   │   └── registration_overlay.png
│   ├── analysis/
│   │   ├── quantitative_metrics.json
│   │   ├── activity_map.tif
│   │   └── visualization.png
│   └── reports/
│       ├── processing_summary.json
│       └── quality_metrics.json
```

### Key Output Files

1. **Segmentation Masks**:
   - `tissue_mask.tif`: Binary mask of tissue regions
   - `activity_mask.tif`: Binary mask of activity regions
   - `combined_mask.tif`: Combined tissue and activity mask

2. **Aligned Images**:
   - `aligned_he.tif`: H&E image aligned to iQID
   - `aligned_iqid.tif`: iQID image (reference)
   - `registration_overlay.png`: Visual overlay of alignment

3. **Analysis Results**:
   - `quantitative_metrics.json`: Numerical measurements
   - `activity_map.tif`: Quantitative activity distribution
   - `visualization.png`: Summary visualization

## Configuration

### Basic Configuration

Create or modify configuration files in the `configs/` directory:

```json
{
  "processing": {
    "gaussian_blur_sigma": 1.0,
    "threshold_method": "otsu",
    "morphology_kernel_size": 5,
    "min_object_size": 100
  },
  "alignment": {
    "max_translation": 50,
    "correlation_threshold": 0.7,
    "pyramid_levels": 3
  },
  "output": {
    "save_intermediate": true,
    "output_format": "tiff",
    "compression": "lzw"
  },
  "visualization": {
    "create_overlays": true,
    "save_plots": true,
    "dpi": 300
  }
}
```

### Configuration Parameters

#### Processing Parameters
- `gaussian_blur_sigma`: Smoothing intensity (0.5-2.0)
- `threshold_method`: Segmentation method ("otsu", "li", "triangle")
- `morphology_kernel_size`: Morphological operation size (3-10)
- `min_object_size`: Minimum object size in pixels (50-500)

#### Alignment Parameters
- `max_translation`: Maximum allowed translation in pixels
- `correlation_threshold`: Minimum correlation for successful alignment
- `pyramid_levels`: Number of pyramid levels for alignment

#### Output Parameters
- `save_intermediate`: Save intermediate processing steps
- `output_format`: File format for outputs ("tiff", "png")
- `compression`: Compression method for TIFF files

## Quality Control

### Validating Results

After processing, validate your results:

```python
# Run quality checks
from evaluation.scripts import comprehensive_pipeline_evaluator

evaluator = comprehensive_pipeline_evaluator.ComprehensivePipelineEvaluator()
results = evaluator.run_comprehensive_evaluation()

# Check overall score
if results['overall_score'] >= 80:
    print("✅ Results are high quality")
elif results['overall_score'] >= 60:
    print("⚠️ Results are acceptable but may need review")
else:
    print("❌ Results need improvement")
```

### Manual Quality Checks

1. **Segmentation Quality**:
   - Check if tissue boundaries are accurately detected
   - Verify activity regions make biological sense
   - Look for over-segmentation or under-segmentation

2. **Alignment Quality**:
   - Examine registration overlay images
   - Check for proper feature alignment
   - Verify no significant distortions

3. **Quantitative Metrics**:
   - Review activity measurements for reasonableness
   - Check for consistent measurements across similar samples
   - Validate against known controls if available

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```
   Error: No module named 'pipelines'
   Solution: Ensure you're running from the iqid-alphas root directory
   ```

2. **Memory Issues**:
   ```
   Error: Out of memory
   Solution: Process smaller batches or reduce image resolution
   ```

3. **Alignment Failures**:
   ```
   Warning: Alignment correlation below threshold
   Solution: Check image quality and overlap between H&E and iQID
   ```

4. **File Format Issues**:
   ```
   Error: Cannot read image file
   Solution: Ensure images are in supported formats (TIFF recommended)
   ```

### Getting Help

1. **Check Logs**: Review output logs for detailed error messages
2. **Validation Reports**: Run validation scripts to identify issues
3. **Test Data**: Try processing with provided test data first
4. **Configuration**: Verify your configuration parameters are appropriate

## Next Steps

After completing this guide:

1. **Process Test Data**: Try the pipeline with provided test samples
2. **Configure for Your Data**: Adjust parameters for your specific images
3. **Batch Processing**: Scale up to process your full dataset
4. **Validation**: Regularly validate results for quality assurance

For advanced usage and customization, see:
- [Configuration Guide](configuration.md)
- [Advanced Usage](advanced_usage.md)
- [API Reference](../api_reference/)

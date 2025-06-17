# Naming Conventions Standard for IQID-Alphas

## Overview

This document defines the consistent naming conventions used throughout the IQID-Alphas project to ensure code readability, maintainability, and professional standards.

## General Principles

1. **Consistency**: All similar entities should follow the same naming pattern
2. **Clarity**: Names should be descriptive and unambiguous
3. **Professional**: Follow Python PEP 8 and industry best practices
4. **Domain-specific**: Use appropriate terminology for medical imaging and quantitative analysis

## File and Directory Naming

### Python Files
- **Format**: `snake_case.py`
- **Examples**: 
  - ✅ `iqid_processing_pipeline.py`
  - ✅ `combined_he_iqid_pipeline.py`
  - ✅ `tissue_segmentation_utils.py`
  - ❌ `iQIDPipeline.py`
  - ❌ `CombinedPipeline.py`

### Configuration Files
- **Format**: `descriptive_name_config.json`
- **Examples**:
  - ✅ `iqid_processing_config.json`
  - ✅ `combined_pipeline_config.json`
  - ✅ `segmentation_config.json`
  - ❌ `config1.json`
  - ❌ `test.json`

### Test Files
- **Format**: `test_feature_description.py`
- **Examples**:
  - ✅ `test_iqid_pipeline.py`
  - ✅ `test_image_segmentation.py`
  - ✅ `test_batch_processing.py`
  - ❌ `testPipeline.py`
  - ❌ `test1.py`

### Documentation Files
- **Format**: `UPPERCASE.md` for root-level docs, `lowercase.md` for subdirectories
- **Examples**:
  - ✅ `README.md`
  - ✅ `CONTRIBUTING.md`
  - ✅ `user_guides/getting_started.md`
  - ✅ `technical/architecture.md`

### Directories
- **Format**: `snake_case` for all directories
- **Examples**:
  - ✅ `src/core/iqid/`
  - ✅ `docs/user_guides/`
  - ✅ `evaluation/scripts/`
  - ❌ `srcCore/`
  - ❌ `userGuides/`

## Class Naming

### Pipeline Classes
- **Format**: `DescriptivePipeline` (PascalCase ending with "Pipeline")
- **Standard Names**:
  - ✅ `IQIDProcessingPipeline` (not `iQIDProcessingPipeline`)
  - ✅ `CombinedHEIQIDPipeline` (not `CombinedHEiQIDPipeline`)
  - ✅ `SimpleIQIDPipeline` (not `SimpleiQIDPipeline`)
  - ✅ `BatchProcessingPipeline`

### Utility Classes
- **Format**: `DescriptiveProcessor` or `DescriptiveAnalyzer`
- **Examples**:
  - ✅ `ImageSegmentationProcessor`
  - ✅ `TissueActivityAnalyzer`
  - ✅ `AlignmentValidator`

### Data Classes
- **Format**: `DescriptiveData` or `DescriptiveResult`
- **Examples**:
  - ✅ `ProcessingResult`
  - ✅ `SegmentationData`
  - ✅ `AlignmentMetrics`

## Method Naming

### Processing Methods
- **Format**: `process_specific_task()`
- **Standard Names**:
  - ✅ `process_sample()` - for single sample processing
  - ✅ `process_batch()` - for batch processing
  - ✅ `process_image_pair()` - for paired image processing
  - ❌ `process_pair()` (ambiguous)
  - ❌ `processSample()` (camelCase)

### Analysis Methods
- **Format**: `analyze_specific_feature()`
- **Examples**:
  - ✅ `analyze_tissue_distribution()`
  - ✅ `analyze_activity_colocalization()`
  - ✅ `calculate_quantitative_metrics()`

### Validation Methods
- **Format**: `validate_specific_aspect()`
- **Examples**:
  - ✅ `validate_input_data()`
  - ✅ `validate_alignment_quality()`
  - ✅ `validate_segmentation_results()`

### Utility Methods
- **Format**: `action_object()` or `get_object()` or `set_object()`
- **Examples**:
  - ✅ `load_configuration()`
  - ✅ `save_results()`
  - ✅ `get_image_properties()`
  - ✅ `set_processing_parameters()`

## Variable Naming

### Image Variables
- **Format**: Descriptive names indicating image type and processing stage
- **Standard Names**:
  - ✅ `iqid_image`, `he_image` (raw images)
  - ✅ `processed_iqid`, `processed_he` (processed images)
  - ✅ `aligned_iqid`, `aligned_he` (aligned images)
  - ✅ `segmented_tissue`, `segmented_activity` (segmentation masks)
  - ❌ `img1`, `img2` (non-descriptive)
  - ❌ `iQID`, `HE` (inconsistent capitalization)

### Path Variables
- **Format**: `object_path` or `object_directory`
- **Examples**:
  - ✅ `iqid_image_path`, `he_image_path`
  - ✅ `output_directory`, `config_file_path`
  - ✅ `sample_directory`, `results_path`

### Configuration Variables
- **Format**: `category_parameter_name`
- **Examples**:
  - ✅ `segmentation_threshold`
  - ✅ `alignment_max_translation`
  - ✅ `processing_gaussian_sigma`

### Result Variables
- **Format**: Descriptive names for metrics and results
- **Examples**:
  - ✅ `alignment_correlation`
  - ✅ `segmentation_accuracy`
  - ✅ `processing_metrics`
  - ✅ `quantitative_results`

## Constant Naming

### Configuration Constants
- **Format**: `UPPERCASE_WITH_UNDERSCORES`
- **Examples**:
  - ✅ `DEFAULT_GAUSSIAN_SIGMA = 1.0`
  - ✅ `MAX_IMAGE_SIZE = 4096`
  - ✅ `SUPPORTED_IMAGE_FORMATS = ['.tif', '.tiff', '.png']`

### File Extensions and Formats
- **Examples**:
  - ✅ `TIFF_EXTENSION = '.tif'`
  - ✅ `CONFIG_FORMAT = 'json'`
  - ✅ `LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'`

## Function Naming

### Processing Functions
- **Format**: `process_specific_task()` or `perform_specific_action()`
- **Examples**:
  - ✅ `process_iqid_image()`
  - ✅ `perform_tissue_segmentation()`
  - ✅ `align_image_pair()`

### Utility Functions
- **Format**: `action_object()` or `calculate_metric()`
- **Examples**:
  - ✅ `load_image_data()`
  - ✅ `save_processing_results()`
  - ✅ `calculate_alignment_score()`

## Parameter Naming

### Method Parameters
- **Format**: Descriptive snake_case names
- **Standard Parameters**:
  - ✅ `iqid_image_path` (not `iqid_path` or `image1`)
  - ✅ `he_image_path` (not `he_path` or `image2`)
  - ✅ `output_directory` (not `output_dir` or `out_path`)
  - ✅ `config_file_path` (not `config` or `cfg`)
  - ✅ `sample_identifier` (not `sample_id` or `id`)

### Configuration Parameters
- **Format**: `category.parameter_name` in JSON
- **Examples**:
```json
{
  "processing": {
    "gaussian_blur_sigma": 1.0,
    "median_filter_size": 3,
    "intensity_normalization_enabled": true
  },
  "segmentation": {
    "tissue_threshold_method": "otsu",
    "activity_threshold_percentile": 95,
    "minimum_object_size": 100
  },
  "alignment": {
    "maximum_translation_pixels": 50,
    "correlation_threshold": 0.7,
    "pyramid_levels": 3
  }
}
```

## Abbreviations and Acronyms

### Standardized Abbreviations
- **IQID**: Always capitalized (not iqid, iQID, or IQiD)
- **HE**: Always capitalized for H&E staining (not he, He, or h&e)
- **QC**: Quality Control
- **ROI**: Region of Interest
- **DPK**: Dose Point Kernel

### Medical/Scientific Terms
- **tissue** (lowercase unless at start of sentence)
- **activity** (lowercase unless at start of sentence)
- **segmentation** (not segmentaion)
- **colocalization** (not co-localization)
- **quantitative** (not quantatative)

## Output File Naming

### Result Files
- **Format**: `sample_id_process_type_timestamp.extension`
- **Examples**:
  - ✅ `sample_001_segmentation_tissue_20250617.tif`
  - ✅ `sample_001_alignment_overlay_20250617.png`
  - ✅ `sample_001_quantitative_metrics_20250617.json`

### Log Files
- **Format**: `process_type_YYYYMMDD.log`
- **Examples**:
  - ✅ `iqid_processing_20250617.log`
  - ✅ `batch_validation_20250617.log`
  - ✅ `pipeline_evaluation_20250617.log`

## Configuration Section Naming

### Standard Configuration Sections
```json
{
  "pipeline_info": {
    "name": "iqid_processing_pipeline",
    "version": "1.0.0",
    "description": "IQID image processing pipeline"
  },
  "input_settings": {
    "supported_formats": [".tif", ".tiff"],
    "required_metadata": ["pixel_size", "acquisition_time"]
  },
  "processing_parameters": {
    "preprocessing": { ... },
    "segmentation": { ... },
    "quantitative_analysis": { ... }
  },
  "output_settings": {
    "save_intermediate_results": true,
    "output_formats": { ... },
    "directory_structure": { ... }
  },
  "quality_control": {
    "validation_thresholds": { ... },
    "diagnostic_outputs": true
  }
}
```

## Documentation Naming

### Section Headers
- **Format**: Title Case for main headers, Sentence case for subheaders
- **Examples**:
  - ✅ `# IQID Processing Pipeline`
  - ✅ `## Getting Started`
  - ✅ `### Installation requirements`

### Code Examples
- **Format**: Descriptive variable names even in examples
- **Examples**:
```python
# ✅ Good example
iqid_pipeline = IQIDProcessingPipeline()
processing_results = iqid_pipeline.process_sample(sample_path)

# ❌ Poor example  
pipeline = Pipeline()
results = pipeline.process(path)
```

## Migration Guidelines

### Updating Existing Code
1. **Gradual Migration**: Update one module at a time
2. **Backward Compatibility**: Maintain old method names temporarily with deprecation warnings
3. **Documentation Updates**: Update all documentation to reflect new naming
4. **Test Updates**: Update all tests to use new naming conventions
5. **Configuration Migration**: Provide migration scripts for configuration files

### Deprecation Process
```python
import warnings

def process_pair(self, *args, **kwargs):
    warnings.warn(
        "process_pair is deprecated, use process_image_pair instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self.process_image_pair(*args, **kwargs)
```

## Validation and Enforcement

### Automated Checks
- Use linting tools to enforce naming conventions
- Implement pre-commit hooks for naming validation
- Regular code reviews focusing on naming consistency

### Style Guide Integration
- Integrate with existing Python style guides (PEP 8)
- Use tools like `pylint`, `flake8`, or `black` for enforcement
- Configure IDEs to highlight naming convention violations

This naming convention standard should be followed for all new code and gradually applied to existing code through refactoring efforts.

# Configuration Guide

This guide explains how to configure the IQID-Alphas pipeline for your specific data and requirements.

## Configuration Files

The pipeline uses JSON configuration files located in the `configs/` directory:

- `iqid_pipeline_config.json`: iQID-only processing configuration
- `combined_pipeline_config.json`: Combined H&E + iQID processing
- `test_config.json`: Configuration for testing and validation

## Configuration Structure

### Basic Structure

```json
{
  "pipeline": {
    "name": "combined_he_iqid_pipeline",
    "version": "1.0"
  },
  "processing": {
    // Image processing parameters
  },
  "segmentation": {
    // Segmentation parameters
  },
  "alignment": {
    // Image alignment parameters
  },
  "analysis": {
    // Quantitative analysis parameters
  },
  "output": {
    // Output configuration
  },
  "visualization": {
    // Visualization settings
  }
}
```

## Processing Parameters

### Preprocessing Settings

```json
"processing": {
  "gaussian_blur_sigma": 1.0,          // Gaussian blur intensity (0.5-3.0)
  "median_filter_size": 3,             // Median filter kernel size (3, 5, 7)
  "normalize_intensity": true,         // Normalize image intensity
  "clip_percentile": [1, 99],          // Intensity clipping percentiles
  "enhance_contrast": false,           // Apply contrast enhancement
  "background_subtraction": {
    "enabled": true,
    "method": "rolling_ball",          // "rolling_ball", "top_hat"
    "radius": 50                       // Background subtraction radius
  }
}
```

#### Parameter Guidelines

- **gaussian_blur_sigma**: 
  - Low (0.5-1.0): Minimal smoothing, preserves fine details
  - Medium (1.0-2.0): Balanced smoothing for most applications
  - High (2.0-3.0): Strong smoothing for noisy images

- **median_filter_size**: 
  - 3: Light noise reduction
  - 5: Standard noise reduction
  - 7: Strong noise reduction (may blur edges)

- **clip_percentile**: 
  - [1, 99]: Standard clipping for most images
  - [0.5, 99.5]: Conservative clipping
  - [2, 98]: Aggressive clipping for high dynamic range

## Segmentation Parameters

### Tissue Segmentation

```json
"segmentation": {
  "tissue": {
    "method": "otsu",                  // "otsu", "li", "triangle", "adaptive"
    "morphology": {
      "opening_size": 3,               // Morphological opening kernel
      "closing_size": 5,               // Morphological closing kernel
      "remove_small_objects": 100      // Minimum object size (pixels)
    },
    "hole_filling": true,              // Fill holes in segmented objects
    "watershed": {
      "enabled": false,                // Apply watershed segmentation
      "min_distance": 10               // Minimum distance between peaks
    }
  },
  "activity": {
    "method": "adaptive",              // Segmentation method for activity
    "threshold_factor": 1.5,           // Threshold multiplier
    "min_activity_size": 50,           // Minimum activity region size
    "max_activity_size": 10000         // Maximum activity region size
  }
}
```

#### Segmentation Method Selection

- **otsu**: Best for bimodal histograms with clear separation
- **li**: Good for images with varying background
- **triangle**: Effective for images with skewed histograms
- **adaptive**: Local thresholding for varying illumination

### Activity Detection

```json
"activity_detection": {
  "percentile_threshold": 95,          // Activity threshold percentile
  "local_maxima": {
    "min_distance": 5,                 // Minimum distance between peaks
    "threshold_abs": 0.1,              // Absolute threshold for peaks
    "threshold_rel": 0.8               // Relative threshold for peaks
  },
  "region_growing": {
    "tolerance": 0.2,                  // Region growing tolerance
    "connectivity": 8                  // Pixel connectivity (4 or 8)
  }
}
```

## Alignment Parameters

### Image Registration

```json
"alignment": {
  "method": "phase_correlation",       // "phase_correlation", "feature_based"
  "max_translation": 50,               // Maximum allowed translation (pixels)
  "max_rotation": 5,                   // Maximum allowed rotation (degrees)
  "correlation_threshold": 0.7,        // Minimum correlation for success
  "pyramid_levels": 3,                 // Multi-scale pyramid levels
  "interpolation": "bilinear",         // "nearest", "bilinear", "bicubic"
  "edge_detection": {
    "enabled": true,
    "sigma": 1.0,                      // Edge detection sigma
    "low_threshold": 0.1,              // Canny low threshold
    "high_threshold": 0.2              // Canny high threshold
  }
}
```

#### Alignment Method Selection

- **phase_correlation**: 
  - Fast and robust for translation-only alignment
  - Good for images with similar content
  - Limited to rigid transformations

- **feature_based**: 
  - More flexible, handles rotation and scaling
  - Better for images with different contrasts
  - Slower but more accurate for complex alignments

### Quality Control

```json
"alignment_qc": {
  "min_overlap_area": 0.7,             // Minimum overlap fraction
  "max_rmse": 5.0,                     // Maximum root mean square error
  "check_mutual_information": true,     // Validate using mutual information
  "save_diagnostics": true             // Save alignment diagnostic images
}
```

## Analysis Parameters

### Quantitative Analysis

```json
"analysis": {
  "measurements": {
    "total_activity": true,            // Measure total activity
    "activity_density": true,          // Calculate activity per unit area
    "regional_analysis": true,         // Analyze different tissue regions
    "colocalization": true             // H&E-iQID colocalization analysis
  },
  "statistics": {
    "percentiles": [25, 50, 75, 90, 95], // Activity percentiles to calculate
    "histogram_bins": 256,             // Number of histogram bins
    "spatial_resolution": 1.0          // Spatial resolution (μm/pixel)
  },
  "decay_correction": {
    "enabled": true,
    "half_life": 9.9,                  // Half-life in days (for Ac-225)
    "acquisition_time": 60,            // Acquisition time in minutes
    "decay_constant": 0.07             // Decay constant (1/day)
  }
}
```

## Output Configuration

### File Output Settings

```json
"output": {
  "base_directory": "./outputs",       // Base output directory
  "save_intermediate": true,           // Save intermediate processing steps
  "file_format": {
    "images": "tiff",                  // "tiff", "png", "jpeg"
    "masks": "tiff",                   // Format for binary masks
    "data": "json",                    // "json", "csv", "hdf5"
    "compression": "lzw"               // TIFF compression ("none", "lzw", "zip")
  },
  "naming_convention": {
    "use_timestamps": false,           // Include timestamps in filenames
    "prefix": "",                      // Filename prefix
    "suffix": ""                       // Filename suffix
  },
  "quality_control": {
    "save_qc_images": true,            // Save quality control visualizations
    "save_logs": true,                 // Save detailed processing logs
    "log_level": "INFO"                // Logging level ("DEBUG", "INFO", "WARNING")
  }
}
```

### Directory Structure

```json
"directory_structure": {
  "segmentation": "segmentation",      // Segmentation results subdirectory
  "alignment": "alignment",            // Alignment results subdirectory
  "analysis": "analysis",              // Analysis results subdirectory
  "visualization": "visualization",    // Visualization subdirectory
  "reports": "reports",                // Reports subdirectory
  "logs": "logs"                       // Logs subdirectory
}
```

## Visualization Parameters

### Plot and Image Settings

```json
"visualization": {
  "create_overlays": true,             // Create overlay visualizations
  "save_plots": true,                  // Save matplotlib plots
  "interactive_plots": false,          // Create interactive plots
  "figure_settings": {
    "dpi": 300,                        // Figure DPI for saved images
    "format": "png",                   // Figure format ("png", "pdf", "svg")
    "bbox_inches": "tight",            // Bounding box setting
    "transparent": false               // Transparent background
  },
  "colormap": {
    "activity": "hot",                 // Colormap for activity visualization
    "tissue": "gray",                  // Colormap for tissue visualization
    "overlay": "viridis"               // Colormap for overlay visualization
  },
  "display_settings": {
    "show_colorbar": true,             // Show colorbars on plots
    "show_scale_bar": true,            // Show scale bars
    "show_annotations": true,          // Show text annotations
    "font_size": 12                    // Default font size
  }
}
```

## Advanced Configuration

### Performance Optimization

```json
"performance": {
  "multiprocessing": {
    "enabled": true,                   // Enable multiprocessing
    "n_jobs": -1,                      // Number of parallel jobs (-1 for all cores)
    "chunk_size": "auto"               // Processing chunk size
  },
  "memory_management": {
    "max_image_size": 4096,            // Maximum image dimension
    "tile_processing": true,           // Process large images in tiles
    "tile_size": 1024,                 // Tile size for large images
    "overlap": 128                     // Tile overlap in pixels
  },
  "caching": {
    "enabled": true,                   // Enable result caching
    "cache_directory": "./cache",      // Cache directory
    "max_cache_size": "1GB"            // Maximum cache size
  }
}
```

### Debug and Development

```json
"debug": {
  "save_debug_images": false,          // Save intermediate debug images
  "verbose_logging": false,            // Enable verbose logging
  "profile_performance": false,        // Profile pipeline performance
  "validate_inputs": true,             // Validate input parameters
  "fail_fast": true                    // Stop on first error
}
```

## Configuration Examples

### High-Quality Processing

For highest quality results (slower processing):

```json
{
  "processing": {
    "gaussian_blur_sigma": 0.5,
    "median_filter_size": 3,
    "normalize_intensity": true,
    "clip_percentile": [0.5, 99.5]
  },
  "segmentation": {
    "tissue": {
      "method": "li",
      "morphology": {
        "opening_size": 2,
        "closing_size": 3,
        "remove_small_objects": 50
      }
    }
  },
  "alignment": {
    "pyramid_levels": 4,
    "correlation_threshold": 0.8,
    "interpolation": "bicubic"
  }
}
```

### Fast Processing

For faster processing (lower quality):

```json
{
  "processing": {
    "gaussian_blur_sigma": 2.0,
    "median_filter_size": 5,
    "normalize_intensity": false
  },
  "segmentation": {
    "tissue": {
      "method": "otsu",
      "morphology": {
        "opening_size": 5,
        "closing_size": 7,
        "remove_small_objects": 200
      }
    }
  },
  "alignment": {
    "pyramid_levels": 2,
    "correlation_threshold": 0.6,
    "interpolation": "bilinear"
  }
}
```

### Noisy Data

For processing noisy or low-quality images:

```json
{
  "processing": {
    "gaussian_blur_sigma": 2.0,
    "median_filter_size": 7,
    "enhance_contrast": true,
    "background_subtraction": {
      "enabled": true,
      "method": "rolling_ball",
      "radius": 100
    }
  },
  "segmentation": {
    "tissue": {
      "method": "adaptive",
      "morphology": {
        "opening_size": 5,
        "closing_size": 7,
        "remove_small_objects": 500
      }
    }
  }
}
```

## Validation and Testing

### Configuration Validation

Test your configuration with:

```python
from pipelines.combined_he_iqid_pipeline import CombinedHEiQIDPipeline

# Load and validate configuration
pipeline = CombinedHEiQIDPipeline('path/to/your/config.json')

# Test with sample data
results = pipeline.process_sample('test_iqid.tif', 'test_he.tif')
```

### Parameter Optimization

For optimal results:

1. **Start with default parameters**
2. **Process a few representative samples**
3. **Evaluate results quality**
4. **Adjust parameters based on results**
5. **Re-test and iterate**

## Best Practices

1. **Backup configurations**: Keep copies of working configurations
2. **Document changes**: Note why parameters were changed
3. **Test incrementally**: Change one parameter at a time
4. **Validate results**: Always check output quality after changes
5. **Use version control**: Track configuration changes over time

For more advanced configuration options, see the [Technical Documentation](../technical/) and [API Reference](../api_reference/).

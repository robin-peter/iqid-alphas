# Visualization Guidelines and Standards

## Overview

The IQID-Alphas project follows strict visualization guidelines to ensure consistency, reproducibility, and professional quality across all generated plots and figures.

## Key Principles

### 1. Consistency
- **Standardized Color Palettes**: Use predefined color schemes across all visualizations
- **Uniform Styling**: Consistent fonts, sizes, and layouts
- **Reproducible Parameters**: All visualization parameters are configurable and version-controlled

### 2. Clarity
- **Informative Titles**: All plots have descriptive, bold titles
- **Clear Labels**: Axis labels with appropriate units
- **Legends**: When multiple data series are present
- **Grid Lines**: Optional, subtle grid lines for better readability

### 3. Professional Quality
- **High Resolution**: 300 DPI for all saved figures
- **Publication Ready**: Vector formats available when needed
- **Accessible Colors**: Color-blind friendly palettes
- **Clean Aesthetics**: Minimal unnecessary elements

## Implementation

### Configuration-Driven Approach

All visualization parameters are controlled through configuration files:

```json
{
  "visualization": {
    "figure_size": [15, 10],
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl",
    "title_fontsize": 14,
    "label_fontsize": 12,
    "show_grid": true,
    "grid_alpha": 0.3
  }
}
```

### Class-Based Architecture

- **BaseVisualizer**: Abstract base class ensuring consistent interface
- **Specialized Visualizers**: Purpose-built classes for different visualization types
- **Configuration Management**: Centralized parameter management
- **Error Handling**: Robust error management and logging

### Code Structure

```python
# Improved architecture
class BaseVisualizer(ABC):
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self._setup_plotting_style()
    
    @abstractmethod
    def create_visualization(self, *args, **kwargs) -> plt.Figure:
        pass
    
    def save_figure(self, fig: plt.Figure, output_path: str) -> None:
        # Standardized saving with consistent parameters
        pass
```

## Visualization Types

### 1. Value Range Analysis
- **Purpose**: Analyze pixel value distributions with outlier detection
- **Features**: Raw vs clipped histograms, percentile analysis, outlier statistics
- **Configuration**: Outlier percentiles, histogram bins, color schemes

### 2. Segmentation Results
- **Purpose**: Display segmentation masks and overlays
- **Features**: Original/mask/overlay triplets, multiple mask visualization
- **Configuration**: Overlay transparency, colormap selection, contour options

### 3. Alignment Results
- **Purpose**: Show image registration quality
- **Features**: Before/after overlays, difference maps, registration metrics
- **Configuration**: Overlay colors, alpha values, metric displays

### 4. Pipeline Integration
- **Purpose**: Comprehensive pipeline stage visualization
- **Features**: Stage-specific plots, summary dashboards, progress tracking
- **Configuration**: Output organization, file formats, logging levels

## Best Practices

### Code Organization

```python
# ✅ Good: Clear class structure
class ValueRangeVisualizer(BaseVisualizer):
    def create_visualization(self, images: Dict[str, np.ndarray]) -> plt.Figure:
        # Validate inputs
        # Analyze data
        # Create visualization
        # Return figure
        pass

# ❌ Avoid: Monolithic functions
def create_all_plots(data, options, paths, formats, ...):
    # Too many responsibilities
    pass
```

### Error Handling

```python
# ✅ Good: Comprehensive error handling
try:
    analysis_results = self.analyzer.analyze_image_collection(images)
    if not analysis_results:
        raise ValueError("No images could be analyzed")
except Exception as e:
    logger.error(f"Analysis failed: {e}")
    return None

# ❌ Avoid: Silent failures
analysis_results = self.analyzer.analyze_image_collection(images)
# No error checking
```

### Configuration Management

```python
# ✅ Good: Configuration-driven parameters
@dataclass
class VisualizationConfig:
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 300
    color_palette: str = 'husl'
    
    @classmethod
    def from_json(cls, config_path: str) -> 'VisualizationConfig':
        # Load from file with fallback to defaults
        pass

# ❌ Avoid: Hard-coded values
plt.figure(figsize=(15, 10))  # Hard-coded size
plt.savefig('output.png', dpi=300)  # Hard-coded DPI
```

### Documentation

```python
# ✅ Good: Comprehensive docstrings
def create_visualization(self, images: Dict[str, np.ndarray], 
                        output_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive value range visualization.
    
    Args:
        images: Dictionary of image_name -> image_array
        output_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValueError: If no images can be analyzed
        TypeError: If images is not a dictionary
    """

# ❌ Avoid: Minimal or missing documentation
def plot_stuff(data, path=None):
    # Creates some plots
    pass
```

## Pipeline Integration

### Stage-Based Visualization

Each pipeline stage can generate appropriate visualizations:

```python
# Pipeline integration
visualizer = PipelineVisualizer(config_path='configs/visualization_config.json')

# Stage-specific visualizations
preprocessing_plots = visualizer.visualize_pipeline_stage(
    'preprocessing', preprocessing_data, sample_name
)

segmentation_plots = visualizer.visualize_pipeline_stage(
    'segmentation', segmentation_data, sample_name
)

# Comprehensive summary
summary_dashboard = visualizer.create_pipeline_summary(
    complete_results, sample_name
)
```

### Output Organization

```
outputs/
├── visualizations/
│   ├── sample_001/
│   │   ├── preprocessing/
│   │   │   ├── value_range_analysis.png
│   │   │   └── preprocessing_comparison.png
│   │   ├── segmentation/
│   │   │   ├── segmentation_result.png
│   │   │   └── segmentation_quality.png
│   │   ├── alignment/
│   │   │   ├── alignment_result.png
│   │   │   └── alignment_metrics.png
│   │   └── pipeline_summary_dashboard.png
│   └── visualization_log.json
```

## Quality Assurance

### Automated Testing

```python
def test_visualization_system():
    """Test the visualization system with sample data."""
    config = VisualizationConfig()
    visualizer = ValueRangeVisualizer(config)
    
    # Generate test data
    test_images = create_test_images()
    
    # Test visualization creation
    fig = visualizer.create_visualization(test_images)
    assert fig is not None
    
    # Test configuration loading
    config_from_file = VisualizationConfig.from_json('test_config.json')
    assert config_from_file.dpi == 300
```

### Validation Checklist

- [ ] **Configuration Loading**: Test with valid and invalid config files
- [ ] **Error Handling**: Verify graceful handling of invalid inputs
- [ ] **Output Quality**: Check resolution, format, and file sizes
- [ ] **Reproducibility**: Same inputs produce identical outputs
- [ ] **Memory Management**: Large datasets don't cause memory issues

## Migration from Old Code

### Common Issues Fixed

1. **Hard-coded Paths**: Moved to configuration files
2. **Inconsistent Styling**: Standardized through base classes
3. **Poor Error Handling**: Added comprehensive exception management
4. **Memory Issues**: Implemented efficient data handling
5. **Code Duplication**: Created reusable components

### Migration Steps

1. **Update Imports**:
   ```python
   # Old
   from value_range_visualization import ValueRangeVisualizer
   
   # New
   from visualization.improved_visualization import ValueRangeVisualizer
   ```

2. **Use Configuration**:
   ```python
   # Old
   visualizer = ValueRangeVisualizer(figsize=(15, 10))
   
   # New
   config = VisualizationConfig.from_json('configs/visualization_config.json')
   visualizer = ValueRangeVisualizer(config)
   ```

3. **Update Method Calls**:
   ```python
   # Old
   create_comprehensive_visualization(images, output_path, outlier_percentiles, is_16bit)
   
   # New
   create_visualization(images, output_path)  # Parameters in config
   ```

## Future Enhancements

### Planned Features

1. **Interactive Visualizations**: Plotly integration for web dashboards
2. **Animation Support**: Progress animations for batch processing
3. **3D Visualizations**: Volume rendering for 3D datasets
4. **Export Formats**: SVG, PDF, and interactive HTML outputs
5. **Template System**: Customizable visualization templates

### Performance Optimizations

1. **Lazy Loading**: Only load visualization modules when needed
2. **Caching**: Cache computed analysis results
3. **Parallel Processing**: Multi-threaded visualization generation
4. **Memory Efficiency**: Streaming processing for large datasets

## Conclusion

The improved visualization system provides:

- **Consistency**: Standardized appearance across all visualizations
- **Flexibility**: Configurable parameters for different use cases
- **Maintainability**: Clean, well-documented code architecture
- **Extensibility**: Easy to add new visualization types
- **Quality**: Publication-ready outputs with proper error handling

All new visualization code should follow these guidelines to maintain the high quality and consistency of the IQID-Alphas project.

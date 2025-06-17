# API Reference: Visualization System

This document provides detailed API documentation for the IQID-Alphas visualization system.

## Table of Contents

- [Visualization Classes](#visualization-classes)
- [Configuration System](#configuration-system)
- [Pipeline Integration](#pipeline-integration)
- [Custom Visualizations](#custom-visualizations)
- [Export and Reporting](#export-and-reporting)

## Visualization Classes

### `src.visualization.improved_visualization.VisualizationManager`

Main class for managing all visualization operations.

#### Class Definition

```python
class VisualizationManager:
    """Central manager for all visualization operations."""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional configuration file."""
```

#### Constructor Parameters

- `config_path` (str, optional): Path to visualization configuration JSON file

#### Methods

##### `create_image_visualization(image: np.ndarray, title: str = None, **kwargs) -> Figure`

Creates a publication-quality image visualization.

**Parameters:**
- `image` (np.ndarray): Input image array
- `title` (str, optional): Plot title
- `**kwargs`: Additional visualization parameters

**Returns:**
- `matplotlib.figure.Figure`: Generated figure

**Example:**
```python
viz_manager = VisualizationManager('configs/visualization_config.json')
fig = viz_manager.create_image_visualization(
    image,
    title="iQID Image Analysis",
    colormap="viridis",
    show_colorbar=True,
    add_scale_bar=True
)
```

##### `create_overlay_visualization(base_image: np.ndarray, overlay: np.ndarray, **kwargs) -> Figure`

Creates overlay visualization for masks and segmentations.

**Parameters:**
- `base_image` (np.ndarray): Base image
- `overlay` (np.ndarray): Overlay image or mask
- `**kwargs`: Overlay parameters

**Returns:**
- `matplotlib.figure.Figure`: Generated figure

**Example:**
```python
fig = viz_manager.create_overlay_visualization(
    base_image=raw_image,
    overlay=tissue_mask,
    overlay_alpha=0.5,
    overlay_colormap="Reds",
    title="Tissue Segmentation"
)
```

##### `create_comparison_visualization(images: List[np.ndarray], titles: List[str], **kwargs) -> Figure`

Creates side-by-side comparison visualization.

**Parameters:**
- `images` (List[np.ndarray]): List of images to compare
- `titles` (List[str]): Titles for each image
- `**kwargs`: Comparison parameters

**Returns:**
- `matplotlib.figure.Figure`: Generated figure

##### `create_analysis_dashboard(results: Dict, **kwargs) -> Figure`

Creates comprehensive analysis dashboard.

**Parameters:**
- `results` (Dict): Analysis results dictionary
- `**kwargs`: Dashboard parameters

**Returns:**
- `matplotlib.figure.Figure`: Dashboard figure

### `src.visualization.improved_visualization.ImageVisualizer`

Specialized class for individual image visualizations.

#### Methods

##### `plot_with_colorbar(image: np.ndarray, **kwargs) -> Tuple[Figure, Axes]`

Creates image plot with customizable colorbar.

**Parameters:**
- `image` (np.ndarray): Input image
- `**kwargs`: Plotting parameters

**Returns:**
- `Tuple[Figure, Axes]`: Figure and axes objects

##### `add_scale_bar(axes: Axes, pixel_size: float, **kwargs) -> None`

Adds scale bar to existing plot.

**Parameters:**
- `axes` (matplotlib.axes.Axes): Target axes
- `pixel_size` (float): Physical size per pixel
- `**kwargs`: Scale bar parameters

##### `add_annotations(axes: Axes, annotations: List[Dict], **kwargs) -> None`

Adds annotations to existing plot.

**Parameters:**
- `axes` (matplotlib.axes.Axes): Target axes
- `annotations` (List[Dict]): List of annotation dictionaries
- `**kwargs`: Annotation parameters

## Configuration System

### Configuration Structure

The visualization system uses JSON configuration files:

```json
{
  "default_style": {
    "figure_size": [12, 8],
    "dpi": 300,
    "font_family": "DejaVu Sans",
    "font_size": 12
  },
  "colormaps": {
    "default": "viridis",
    "tissue": "Blues",
    "activity": "Reds",
    "overlay": "jet"
  },
  "image_settings": {
    "interpolation": "bilinear",
    "aspect": "equal",
    "origin": "upper"
  },
  "export": {
    "formats": ["png", "pdf", "svg"],
    "dpi": 300,
    "bbox_inches": "tight",
    "transparent": false
  }
}
```

### Configuration Management

#### Loading Configuration

```python
from src.visualization.improved_visualization import load_config

# Load configuration
config = load_config('configs/visualization_config.json')

# Initialize with configuration
viz_manager = VisualizationManager(config=config)
```

#### Runtime Configuration Updates

```python
# Update specific configuration values
viz_manager.update_config({
    'default_style': {
        'figure_size': [16, 10],
        'dpi': 600
    }
})
```

## Pipeline Integration

### `src.visualization.pipeline_visualization.PipelineVisualizer`

Specialized visualizer for pipeline integration.

#### Class Definition

```python
class PipelineVisualizer:
    """Visualizer integrated with pipeline processing stages."""
    
    def __init__(self, pipeline_config: Dict, viz_config: Dict = None):
        """Initialize with pipeline and visualization configurations."""
```

#### Methods

##### `visualize_processing_stage(stage_name: str, data: Dict, **kwargs) -> Figure`

Creates visualizations for specific processing stages.

**Parameters:**
- `stage_name` (str): Name of processing stage
- `data` (Dict): Stage data and results
- `**kwargs`: Visualization parameters

**Returns:**
- `matplotlib.figure.Figure`: Generated figure

**Example:**
```python
pipeline_viz = PipelineVisualizer(pipeline_config, viz_config)

# Visualize segmentation stage
fig = pipeline_viz.visualize_processing_stage(
    'segmentation',
    {
        'original_image': image,
        'tissue_mask': mask,
        'quantification': results
    }
)
```

##### `create_pipeline_summary(pipeline_results: Dict, **kwargs) -> Figure`

Creates comprehensive pipeline summary visualization.

**Parameters:**
- `pipeline_results` (Dict): Complete pipeline results
- `**kwargs`: Summary parameters

**Returns:**
- `matplotlib.figure.Figure`: Summary figure

##### `generate_processing_report(results: Dict, output_path: str, **kwargs) -> None`

Generates complete processing report with visualizations.

**Parameters:**
- `results` (Dict): Processing results
- `output_path` (str): Output file path
- `**kwargs`: Report parameters

## Custom Visualizations

### Creating Custom Visualizers

```python
from src.visualization.improved_visualization import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    """Custom visualizer for specialized analysis."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
    
    def create_custom_plot(self, data: Dict, **kwargs) -> Figure:
        """Create custom visualization."""
        fig, ax = self.create_figure(**kwargs)
        
        # Custom plotting logic
        self.plot_custom_data(ax, data)
        
        # Apply styling
        self.apply_styling(fig, ax)
        
        return fig
    
    def plot_custom_data(self, ax: Axes, data: Dict) -> None:
        """Custom data plotting logic."""
        # Implementation specific to your needs
        pass
```

### Visualization Plugins

```python
# Register custom visualization plugin
@viz_manager.register_plugin('custom_analysis')
def custom_analysis_viz(data: Dict, **kwargs) -> Figure:
    """Custom analysis visualization plugin."""
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    # Custom visualization logic
    ax.plot(data['x'], data['y'])
    ax.set_title(kwargs.get('title', 'Custom Analysis'))
    
    return fig

# Use the plugin
fig = viz_manager.create_visualization('custom_analysis', data)
```

## Export and Reporting

### Export Functions

#### `export_figure(figure: Figure, filepath: str, **kwargs) -> None`

Exports figure to file with various format options.

**Parameters:**
- `figure` (matplotlib.figure.Figure): Figure to export
- `filepath` (str): Output file path
- `**kwargs`: Export parameters

**Example:**
```python
from src.visualization.improved_visualization import export_figure

export_figure(
    figure,
    'outputs/analysis_plot.png',
    dpi=300,
    bbox_inches='tight',
    transparent=True
)
```

#### `export_multiple_figures(figures: Dict[str, Figure], output_dir: str, **kwargs) -> None`

Exports multiple figures with consistent naming.

**Parameters:**
- `figures` (Dict[str, Figure]): Dictionary of figures with names
- `output_dir` (str): Output directory
- `**kwargs`: Export parameters

### Report Generation

#### `generate_visualization_report(results: Dict, output_path: str, **kwargs) -> None`

Generates comprehensive visualization report.

**Parameters:**
- `results` (Dict): Analysis results
- `output_path` (str): Output file path
- `**kwargs`: Report parameters

**Example:**
```python
generate_visualization_report(
    pipeline_results,
    'outputs/analysis_report.html',
    include_interactive=True,
    template='comprehensive'
)
```

## Advanced Features

### Interactive Visualizations

```python
# Create interactive visualization
from src.visualization.interactive import InteractiveVisualizer

interactive_viz = InteractiveVisualizer(config)
widget = interactive_viz.create_interactive_plot(
    image,
    features=['zoom', 'pan', 'measure', 'annotate']
)
```

### Animation Support

```python
# Create animation from time series data
from src.visualization.animation import AnimationCreator

animator = AnimationCreator(config)
animation = animator.create_time_series_animation(
    image_sequence,
    output_path='outputs/time_series.gif',
    fps=2,
    add_timestamp=True
)
```

### 3D Visualizations

```python
# Create 3D visualization for volume data
from src.visualization.volume import VolumeVisualizer

volume_viz = VolumeVisualizer(config)
fig = volume_viz.create_volume_rendering(
    volume_data,
    threshold=0.5,
    colormap='viridis',
    add_isosurface=True
)
```

## Error Handling

### Visualization Exceptions

```python
from src.visualization.exceptions import (
    VisualizationError,
    ConfigurationError,
    ExportError
)

try:
    fig = viz_manager.create_image_visualization(image)
except VisualizationError as e:
    print(f"Visualization error: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Graceful Degradation

```python
# Enable graceful degradation for missing dependencies
viz_manager.enable_graceful_degradation(True)

# Will use fallback visualization if advanced features unavailable
fig = viz_manager.create_visualization(
    data,
    preferred_method='advanced',
    fallback_method='basic'
)
```

## Performance Optimization

### Memory Management

```python
# Configure memory-efficient visualization
viz_manager.set_memory_optimization(
    enable_lazy_loading=True,
    max_image_size=(2048, 2048),
    use_compression=True
)
```

### Batch Processing

```python
# Batch visualization creation
figures = viz_manager.create_batch_visualizations(
    image_list,
    visualization_type='comparison',
    output_dir='outputs/batch_viz/',
    parallel_processing=True,
    max_workers=4
)
```

## Integration Examples

### Pipeline Integration

```python
# Integrate with processing pipeline
class EnhancedPipeline(SimpleiQIDPipeline):
    def __init__(self, config_path: str, viz_config_path: str = None):
        super().__init__(config_path)
        self.visualizer = PipelineVisualizer(
            self.config,
            viz_config_path
        )
    
    def process_with_visualization(self, image_path: str, output_dir: str):
        # Process image
        results = self.process_single_image(image_path, output_dir)
        
        # Generate visualizations
        viz_results = self.visualizer.create_pipeline_summary(results)
        
        # Export visualizations
        self.visualizer.export_results(viz_results, output_dir)
        
        return results
```

### Custom Analysis Integration

```python
# Custom analysis with integrated visualization
def analyze_with_visualization(image_path: str, config: Dict):
    # Load and process image
    image = load_image(image_path)
    results = analyze_image(image, config)
    
    # Create visualizations
    viz_manager = VisualizationManager(config['visualization'])
    
    # Generate analysis plots
    plots = {
        'overview': viz_manager.create_image_visualization(image),
        'segmentation': viz_manager.create_overlay_visualization(
            image, results['mask']
        ),
        'quantification': viz_manager.create_analysis_dashboard(results)
    }
    
    return results, plots
```

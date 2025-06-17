# IQID-Alphas

**🎉 PRODUCTION READY - Version 1.0.0 ✅**

A modern, comprehensive Python package for quantitative imaging analysis using iQID (Imaging Quantitative ID) cameras and H&E histology images. This system provides end-to-end processing from raw data to publication-ready analyses with a clean, modular architecture.

**Authors:** Robin Peter, Brian Miller  
**Production Release:** December 2024

## 🚀 Quick Start

```bash
# Clone and install
git clone <repository-url>
cd iqid-alphas
pip install -r requirements.txt

# Simple usage with the new API
python -c "
import iqid_alphas
pipeline = iqid_alphas.SimplePipeline()
results = pipeline.process('path/to/data')
"

# Or use the advanced pipeline
python -c "
import iqid_alphas
pipeline = iqid_alphas.AdvancedPipeline()
results = pipeline.process('path/to/data', 'path/to/config.json')
"
```

## 📋 What's New in Version 1.0.0

- ✅ **Modern Package Structure**: Clean, importable `iqid_alphas` Python package
- ✅ **Simple API**: Easy-to-use classes with sensible defaults
- ✅ **Modular Design**: Core, pipelines, and visualization components
- ✅ **Production Ready**: Comprehensive error handling and validation
- ✅ **Documentation**: Complete API reference and examples
- ✅ **Backwards Compatibility**: Legacy scripts preserved in archive/

## 📚 Papers and Citations
- (2022, Sci Rep, initial methods): https://doi.org/10.1038/s41598-022-22664-5
- (2024, Sci Rep, 3D sub-organ dosimetry and TCP): https://doi.org/10.1038/s41598-024-70417-3

Permanent DOI of the initial repository release in 2022: [![DOI](https://zenodo.org/badge/540307496.svg)](https://zenodo.org/badge/latestdoi/540307496)

## 📦 Package Structure

The repository is now organized as a modern Python package:

```
iqid-alphas/
├── iqid_alphas/                 # Main Python package
│   ├── __init__.py             # Package API
│   ├── core/                   # Core processing modules
│   │   ├── processor.py        # Main IQIDProcessor class
│   │   ├── alignment.py        # Image alignment utilities
│   │   └── segmentation.py     # Segmentation algorithms
│   ├── pipelines/              # Processing pipelines
│   │   ├── simple.py          # Simple processing pipeline
│   │   ├── advanced.py        # Advanced processing pipeline
│   │   └── combined.py        # Combined H&E-iQID pipeline
│   ├── utils/                  # Utility functions
│   └── visualization/          # Plotting and visualization
│       └── plotter.py         # Visualizer class
│
├── examples/                   # Usage examples
├── tests/                      # Test suite
├── docs/                       # Documentation
├── configs/                    # Configuration files
├── data/                       # Sample data directory
├── archive/                    # Legacy code (preserved)
└── src/                        # Legacy source (preserved)
```

## 🔧 API Reference

### Core Components

```python
import iqid_alphas

# Core processor for iQID data
processor = iqid_alphas.IQIDProcessor()
processed_data = processor.process(image_data)

# Image alignment
aligner = iqid_alphas.ImageAligner()
aligned_images = aligner.align(image1, image2)

# Image segmentation
segmenter = iqid_alphas.ImageSegmenter()
segments = segmenter.segment(image)
```

### Processing Pipelines

```python
# Simple pipeline - minimal configuration
simple = iqid_alphas.SimplePipeline()
results = simple.process(data_path)

# Advanced pipeline - full control
advanced = iqid_alphas.AdvancedPipeline()
results = advanced.process(data_path, config_path)

# Combined H&E-iQID pipeline
combined = iqid_alphas.CombinedPipeline()
results = combined.process(iqid_path, he_path)
```

### Visualization

```python
# Create publication-quality plots
viz = iqid_alphas.Visualizer()
viz.plot_activity_map(data)
viz.plot_dose_distribution(dose_data)
viz.save_figure('output.png')
```
## 📖 Examples and Usage

### Basic Processing Example

```python
import iqid_alphas
import numpy as np

# Load and process iQID data
processor = iqid_alphas.IQIDProcessor()
image_data = np.load('sample_data.npy')
processed = processor.process(image_data)

# Visualize results
viz = iqid_alphas.Visualizer()
viz.plot_activity_map(processed)
viz.show()
```

### Pipeline Processing Example

```python
import iqid_alphas

# Simple processing
pipeline = iqid_alphas.SimplePipeline()
results = pipeline.process('/path/to/data')

# Advanced processing with custom config
advanced = iqid_alphas.AdvancedPipeline()
results = advanced.process('/path/to/data', '/path/to/config.json')

# Access results
activity_map = results['activity_map']
dose_distribution = results['dose_distribution']
```

### Configuration Example

```python
# Create custom configuration
config = {
    'processing': {
        'smooth_sigma': 1.0,
        'threshold': 0.1
    },
    'visualization': {
        'colormap': 'viridis',
        'save_plots': True
    }
}

pipeline = iqid_alphas.AdvancedPipeline(config=config)
```

## 🧪 Testing and Validation

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python tests/test_core.py
python tests/test_pipelines.py
python tests/test_visualization.py
```

### Validation Scripts
```bash
# Validate installation
python scripts/validate_installation.py

# Run production checks
python evaluation/scripts/production_validation.py
```

## 🏗️ Legacy Support

### Accessing Legacy Functionality
The original modules are preserved and can still be accessed:

```python
# Access original iqid modules
import sys
sys.path.append('src/core')
from iqid import align, dpk, helper, process_object

# Use original automation scripts
python archive/automate_processing.py
```

### Migration Guide
- **Old**: `from iqid import align` → **New**: `import iqid_alphas; aligner = iqid_alphas.ImageAligner()`
- **Old**: `automate_processing.py` → **New**: `iqid_alphas.SimplePipeline()`
- **Old**: Manual configuration → **New**: JSON config files in `configs/`

## 🔧 Dependencies

Core requirements are listed in `requirements.txt`. Key dependencies include:
- **numpy** (≥1.20.2) - Numerical computing
- **opencv-python** (≥4.0.1) - Computer vision operations  
- **scikit-image** (≥0.18.1) - Image processing algorithms
- **scipy** (≥1.6.2) - Scientific computing
- **PyStackReg** (≥0.2.5) - Image registration
- **matplotlib** - Visualization and plotting
- **tifffile** - TIFF image handling

Optional dependencies:
- **becquerel** - Gamma spectroscopy features
- **jupyter** - Interactive notebooks
- **pytest** - Testing framework

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes in the `iqid_alphas/` package
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request

### Development Setup
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests before submitting
python -m pytest tests/
```

## 📄 License and Citation

This project is licensed under the terms specified in `LICENSE.txt`.

If you use this software in your research, please cite:
- (2022, Sci Rep, initial methods): https://doi.org/10.1038/s41598-022-22664-5
- (2024, Sci Rep, 3D sub-organ dosimetry and TCP): https://doi.org/10.1038/s41598-024-70417-3

Permanent DOI: [![DOI](https://zenodo.org/badge/540307496.svg)](https://zenodo.org/badge/latestdoi/540307496)

## 📞 Support

For questions, issues, or data access requests:
- Open an issue on GitHub
- Contact the authors: Robin Peter, Brian Miller
- Check the documentation in `docs/`

---

**Version 1.0.0** - December 2024  
**Production Ready** ✅

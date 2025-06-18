# IQID-Alphas Advanced Batch Processing CLI

## Overview

The IQID-Alphas CLI provides a powerful command-line interface for batch processing of iQID and H&E imaging data. It unifies all processing workflows into a single, easy-to-use tool with advanced features like:

- **Unified Interface**: Single command for all pipeline types (simple, advanced, combined)
- **Automatic Data Discovery**: Intelligent detection and pairing of iQID and H&E files
- **Batch Processing**: Process hundreds of samples with progress monitoring
- **Quality Control**: Built-in validation and quality assessment
- **Flexible Configuration**: JSON-based configuration with validation
- **Comprehensive Reporting**: Detailed processing summaries and error tracking

## Installation & Setup

The CLI is included with the IQID-Alphas package. No additional installation is required.

## Usage

### Method 1: Module Execution
```bash
python -m iqid_alphas.cli [command] [options]
```

### Method 2: Standalone Script
```bash
./iqid-cli.py [command] [options]
```

## Commands

### 1. Process Command

Process a batch of samples using the specified pipeline.

```bash
# Basic usage
python -m iqid_alphas.cli process --data /path/to/data --config configs/simple.json

# Advanced pipeline with custom output
python -m iqid_alphas.cli process \
    --data /path/to/ucsf_data \
    --config configs/cli_batch_config.json \
    --pipeline advanced \
    --output results/batch_2024_06_18

# Quick test with limited samples
python -m iqid_alphas.cli process \
    --data /path/to/test_data \
    --config configs/cli_quick_config.json \
    --max-samples 5 \
    --verbose
```

#### Options:
- `--data` (required): Path to data directory
- `--config` (required): Path to configuration file
- `--pipeline`: Pipeline type (`simple`, `advanced`, `combined`)
- `--output`: Output directory for results
- `--max-samples`: Maximum number of samples to process
- `--verbose`: Enable verbose logging

### 2. Discover Command

Discover and analyze available data files.

```bash
# Basic discovery
python -m iqid_alphas.cli discover --data /path/to/data

# Save discovery results
python -m iqid_alphas.cli discover \
    --data /path/to/ucsf_data \
    --output discovery_results.json
```

#### Options:
- `--data` (required): Path to data directory
- `--output`: Save discovery results to JSON file

### 3. Config Command

Configuration file management.

```bash
# Create default configuration
python -m iqid_alphas.cli config --create configs/my_config.json

# Validate existing configuration
python -m iqid_alphas.cli config --validate configs/existing_config.json
```

#### Options:
- `--create`: Create default configuration file
- `--validate`: Validate configuration file

## Configuration Files

### Basic Configuration Structure
```json
{
  "processing": {
    "normalize": true,
    "remove_outliers": true,
    "gaussian_filter": true,
    "filter_sigma": 1.0
  },
  "segmentation": {
    "method": "otsu",
    "min_size": 100,
    "remove_small_objects": true
  },
  "alignment": {
    "method": "phase_correlation",
    "max_shift": 50,
    "subpixel_precision": true
  },
  "visualization": {
    "save_plots": true,
    "output_dir": "results",
    "dpi": 300,
    "format": "png"
  },
  "quality_control": {
    "enable_validation": true,
    "min_alignment_score": 0.5,
    "max_processing_time": 300
  }
}
```

### Pre-configured Files
- `configs/cli_batch_config.json`: Full-featured batch processing
- `configs/cli_quick_config.json`: Fast processing for testing

## Data Discovery

The CLI automatically discovers and pairs data files based on:

### Supported File Formats
- **iQID files**: `.tif`, `.tiff`, `.npy`
- **H&E files**: `.tif`, `.tiff`, `.png`, `.jpg`

### Naming Pattern Recognition
The CLI recognizes files containing keywords:
- **iQID keywords**: `iqid`, `mBq`, `corr`, `aligned`
- **H&E keywords**: `he`, `h&e`, `hematoxylin`, `eosin`, `histology`

### Sample Pairing
Automatic pairing based on filename similarity using:
- Base filename extraction
- Pattern matching
- Similarity scoring

## Pipeline Types

### Simple Pipeline
- Basic iQID processing
- Preprocessing and segmentation
- Fast execution
- Suitable for initial analysis

```bash
python -m iqid_alphas.cli process \
    --data /path/to/data \
    --config configs/simple.json \
    --pipeline simple
```

### Advanced Pipeline
- Comprehensive iQID analysis
- Quality metrics calculation
- Advanced visualization
- Detailed reporting

```bash
python -m iqid_alphas.cli process \
    --data /path/to/data \
    --config configs/advanced.json \
    --pipeline advanced
```

### Combined Pipeline
- Joint iQID + H&E processing
- Cross-modal alignment
- Tissue analysis
- Integrated visualization

```bash
python -m iqid_alphas.cli process \
    --data /path/to/data \
    --config configs/combined.json \
    --pipeline combined
```

## Output Structure

### Batch Processing Results
```
results/
├── batch_results.json          # Processing summary
├── sample_001/                 # Individual sample results
│   ├── processed_images/
│   ├── visualizations/
│   └── quality_metrics.json
├── sample_002/
│   └── ...
└── summary_report.html         # Overall batch report
```

### Result Files
- **batch_results.json**: Complete processing summary
- **sample_*/**: Individual sample processing results
- **quality_metrics.json**: Quality assessment for each sample
- **visualizations/**: Generated plots and figures

## Quality Control

### Automatic Validation
- File format validation
- Processing pipeline checks
- Quality score calculation
- Error detection and reporting

### Quality Metrics
- Alignment quality scores
- Processing success rates
- Execution time tracking
- Memory usage monitoring

## Error Handling

### Robust Processing
- Continue processing on individual failures
- Detailed error logging
- Graceful degradation
- Recovery mechanisms

### Error Reporting
- Individual sample error tracking
- Comprehensive error logs
- Processing summary with failure analysis
- Debugging information with `--verbose`

## Examples

### Real-World UCSF Data Processing
```bash
# Process all UCSF samples with advanced pipeline
python -m iqid_alphas.cli process \
    --data /path/to/ucsf_data \
    --config configs/cli_batch_config.json \
    --pipeline advanced \
    --output results/ucsf_batch_$(date +%Y%m%d) \
    --verbose

# Quick validation run with first 10 samples
python -m iqid_alphas.cli process \
    --data /path/to/ucsf_data \
    --config configs/cli_quick_config.json \
    --max-samples 10 \
    --output results/validation_run
```

### Development and Testing
```bash
# Discover available data
python -m iqid_alphas.cli discover \
    --data /path/to/test_data \
    --output test_data_inventory.json

# Create and validate custom configuration
python -m iqid_alphas.cli config --create configs/custom.json
python -m iqid_alphas.cli config --validate configs/custom.json

# Test processing with custom config
python -m iqid_alphas.cli process \
    --data /path/to/test_data \
    --config configs/custom.json \
    --max-samples 3 \
    --verbose
```

## Performance Considerations

### Batch Size Optimization
- Process samples in appropriate chunks
- Monitor memory usage
- Adjust `max_samples` for system capacity

### Parallel Processing
- Future enhancement for multi-core processing
- Currently processes samples sequentially
- Configurable through batch_processing settings

## Troubleshooting

### Common Issues

#### Data Discovery Problems
```bash
# Check data structure
python -m iqid_alphas.cli discover --data /path/to/data

# Verify file naming patterns match expected keywords
```

#### Configuration Issues
```bash
# Validate configuration
python -m iqid_alphas.cli config --validate configs/your_config.json

# Create working default configuration
python -m iqid_alphas.cli config --create configs/working_config.json
```

#### Processing Failures
```bash
# Run with verbose logging
python -m iqid_alphas.cli process ... --verbose

# Start with smaller batch
python -m iqid_alphas.cli process ... --max-samples 1
```

### Getting Help
```bash
# General help
python -m iqid_alphas.cli --help

# Command-specific help
python -m iqid_alphas.cli process --help
python -m iqid_alphas.cli discover --help
python -m iqid_alphas.cli config --help
```

## Integration with Existing Workflows

The CLI is designed to complement existing IQID-Alphas workflows:

- **UCSF Batch Processing**: Replaces and enhances `examples/ucsf_consolidated/run_batch_processing.py`
- **Pipeline Integration**: Uses existing pipeline classes without modification
- **Configuration Compatibility**: Works with existing configuration files
- **Result Format**: Compatible with existing analysis scripts

## Future Enhancements

- **Parallel Processing**: Multi-core batch processing
- **Cloud Integration**: AWS/Azure processing support
- **Real-time Monitoring**: Web dashboard for batch processing
- **Advanced Scheduling**: Cron-compatible batch scheduling
- **Plugin System**: Custom pipeline plugins

## Contributing

To extend the CLI functionality:

1. Add new commands in `iqid_alphas/cli.py`
2. Follow the existing command structure pattern
3. Add comprehensive help documentation
4. Include example usage in this documentation
5. Test with various data scenarios

The CLI is designed to be the primary interface for production batch processing while maintaining full compatibility with the existing IQID-Alphas ecosystem.

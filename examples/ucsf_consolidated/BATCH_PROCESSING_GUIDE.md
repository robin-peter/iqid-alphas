# UCSF Batch Processing System

## Overview

The UCSF Batch Processing System is an enhanced workflow that automatically discovers and processes ALL available samples in the UCSF dataset, generating comprehensive visualizations and quality assessments for each sample plus detailed summary analysis across the entire dataset.

## Key Features

### 🔍 Automatic Sample Discovery
- Scans the UCSF data structure to find all available sample pairs
- Matches H&E and iQID data across different locations (DataPush1, ReUpload)
- Reports sample statistics (kidney vs tumor, left vs right, data availability)

### 📊 Individual Sample Processing
- **Path 1**: iQID Raw → Aligned processing with quality validation
- **Path 2**: Aligned iQID + H&E coregistration with tissue analysis
- Comprehensive quality assessment for each sample
- Individual visualization dashboard per sample

### 📈 Comprehensive Summary Analysis
- Batch processing overview dashboard
- Quality metrics distribution analysis  
- Sample comparison charts
- Processing performance analysis
- Statistical summary across all samples

### 🎯 Quality Assessment
- Alignment quality scores
- Registration quality metrics
- Processing efficiency measures
- Data completeness validation
- Outlier detection and flagging

## Quick Start

### Basic Usage
```bash
cd examples/ucsf_consolidated

# Process all available samples
python run_batch_processing.py

# Quick test with first 3 samples
python run_batch_processing.py --quick

# Demo mode with mock data
python run_batch_processing.py --demo
```

### Advanced Options
```bash
# Process specific number of samples
python run_batch_processing.py --samples 10

# Use custom configuration
python run_batch_processing.py --config configs/custom_config.json

# Enable verbose logging
python run_batch_processing.py --verbose
```

## Generated Outputs

### Directory Structure
```
outputs/
├── batch_processing/           # Main processing results
│   ├── {sample_key}/          # Individual sample results
│   │   ├── path1_iqid_alignment/
│   │   └── path2_coregistration/
│   └── batch_processing_results.json
├── batch_visualizations/       # Individual sample visualizations
│   └── {sample_key}/
│       ├── {sample_key}_quality_dashboard.png
│       ├── {sample_key}_alignment_quality.png
│       └── {sample_key}_activity_histogram.png
└── batch_summary/             # Summary analysis
    ├── batch_processing_dashboard.png
    ├── quality_metrics_summary.png
    ├── sample_comparison_radar.png
    ├── processing_performance_analysis.png
    ├── batch_processing_summary.md
    └── quality_metrics.csv
```

### Key Visualizations

#### Summary Visualizations
1. **Batch Processing Dashboard** (`batch_processing_dashboard.png`)
   - Processing results pie chart (success/failure)
   - Quality score distribution histogram
   - Sample type distribution (kidney/tumor)
   - Processing timeline
   - Quality metrics heatmap
   - Statistical summary panel
   - Success/failure overview

2. **Quality Metrics Summary** (`quality_metrics_summary.png`)
   - Alignment vs registration quality scatter plot
   - Overall quality distribution
   - Quality vs processing efficiency
   - Quality metrics box plots

3. **Sample Comparison Charts** (`sample_comparison_radar.png`)
   - Radar charts for top-performing samples
   - Multi-dimensional quality comparison
   - Performance benchmarking

4. **Processing Performance Analysis** (`processing_performance_analysis.png`)
   - Processing time distribution
   - Time vs quality correlation
   - Cumulative processing time
   - Efficiency summary statistics

#### Individual Sample Visualizations
1. **Quality Dashboard** (`{sample}_quality_dashboard.png`)
   - Path 1 alignment quality metrics
   - Path 2 registration quality metrics
   - Activity distribution pie chart
   - Processing time comparison
   - Quality radar chart
   - Summary statistics panel

2. **Alignment Quality** (`{sample}_alignment_quality.png`)
   - Frame-by-frame correlation plot
   - Quality threshold lines
   - Temporal quality assessment

3. **Activity Histogram** (`{sample}_activity_histogram.png`)
   - Activity value distribution
   - Statistical markers (mean, median)
   - Data distribution analysis

## Configuration

### Batch Processing Configuration
The system uses `configs/ucsf_batch_config.json` for comprehensive configuration:

```json
{
    "batch_processing": {
        "max_parallel_samples": 4,
        "continue_on_error": true,
        "quality_thresholds": {
            "minimum_alignment_score": 0.6,
            "minimum_registration_score": 0.5,
            "minimum_overall_score": 0.55
        }
    },
    "comprehensive_visualization": {
        "individual_sample_plots": {
            "quality_dashboard": true,
            "alignment_quality_over_time": true,
            "activity_distribution_histogram": true
        },
        "summary_visualizations": {
            "batch_overview_dashboard": true,
            "quality_metrics_summary": true,
            "sample_comparison_charts": true,
            "processing_performance_analysis": true
        }
    }
}
```

### Quality Control
```json
{
    "quality_control": {
        "metrics": {
            "alignment_correlation_threshold": 0.7,
            "registration_quality_threshold": 0.6,
            "signal_to_noise_threshold": 5.0,
            "tissue_coverage_threshold": 0.3
        },
        "statistical_analysis": {
            "compute_batch_statistics": true,
            "outlier_detection": true,
            "correlation_analysis": true
        }
    }
}
```

## Architecture

### Core Components

1. **UCSFBatchProcessor** (`ucsf_batch_processor.py`)
   - Main batch processing orchestrator
   - Extends UCSFConsolidatedWorkflow
   - Handles sample discovery and processing coordination

2. **Sample Discovery**
   - Uses UCSFDataMatcher to find all available samples
   - Supports both real UCSF data and mock data for testing
   - Reports comprehensive sample statistics

3. **Processing Pipeline**
   - Path 1: iQID Raw → Aligned processing
   - Path 2: Aligned iQID + H&E coregistration
   - Quality assessment and validation
   - Error handling and recovery

4. **Visualization Engine**
   - Individual sample visualization creation
   - Summary dashboard generation
   - Statistical analysis and plotting
   - Configurable output formats

5. **Quality Assessment**
   - Multi-dimensional quality scoring
   - Statistical analysis across samples
   - Outlier detection and flagging
   - Performance benchmarking

## Usage Examples

### Programmatic Usage
```python
from ucsf_batch_processor import UCSFBatchProcessor

# Initialize batch processor
processor = UCSFBatchProcessor('configs/ucsf_batch_config.json')

# Run batch processing
results = processor.run_batch_processing(max_samples=5)

# Access results
sample_results = results['sample_results']
quality_metrics = results['quality_metrics']
statistical_summary = results['statistical_summary']
```

### Integration with Existing Workflows
```python
# Use with existing pipeline components
from iqid_alphas.pipelines.advanced import AdvancedPipeline
from ucsf_batch_processor import UCSFBatchProcessor

# Combine batch processing with advanced pipeline
processor = UCSFBatchProcessor('config.json')
advanced_pipeline = AdvancedPipeline()

# Process samples with enhanced analysis
results = processor.run_batch_processing()
```

## Quality Metrics

### Individual Sample Metrics
- **Alignment Score**: Composite score from correlation, displacement, SNR
- **Registration Score**: Based on registration quality, tissue coverage, heterogeneity
- **Overall Quality Score**: Combined alignment and registration assessment
- **Processing Efficiency**: Samples processed per unit time
- **Data Completeness**: Availability and quality of input data

### Batch Statistics
- Mean, standard deviation, range of quality scores
- Processing time statistics and efficiency metrics
- Success/failure rates and error analysis
- Cross-sample correlation analysis
- Distribution analysis and outlier detection

## Testing and Validation

### Quick Test
```bash
# Run test with mock data
python test_batch_processing.py
```

### Validation Steps
1. **Sample Discovery**: Verify all expected samples are found
2. **Processing Quality**: Check alignment and registration metrics
3. **Visualization Creation**: Ensure all plots are generated correctly
4. **Statistical Analysis**: Validate summary statistics and correlations
5. **Output Organization**: Verify proper file structure and naming

## Performance Considerations

### Processing Efficiency
- Typical processing time: 1-3 minutes per sample
- Memory usage: ~2-4 GB for typical dataset
- Disk space: ~50-100 MB per sample output

### Scalability
- Supports datasets with hundreds of samples
- Configurable parallel processing (future enhancement)
- Memory-efficient data handling
- Progressive result saving

## Troubleshooting

### Common Issues
1. **No Samples Found**: Check UCSF data path configuration
2. **Processing Failures**: Review individual sample error logs
3. **Visualization Errors**: Verify matplotlib/seaborn installation
4. **Memory Issues**: Reduce max_parallel_samples in config
5. **Missing Outputs**: Check file permissions and disk space

### Debug Information
- Comprehensive logging with configurable levels
- Individual sample processing logs
- Error details and stack traces
- Performance timing information

## Future Enhancements

### Planned Features
- Parallel sample processing for improved speed
- Interactive visualization dashboard
- Advanced statistical analysis and machine learning integration
- Export to different formats (Excel, PowerPoint, etc.)
- Integration with external analysis tools

### Extensibility
- Plugin architecture for custom analysis modules
- Configurable visualization templates
- Custom quality metrics definition
- Integration with external databases

## Integration with IQID-Alphas Package

The batch processing system is fully integrated with the IQID-Alphas package:

- Uses core processing modules (IQIDProcessor, ImageAligner, etc.)
- Leverages pipeline architecture (SimplePipeline, AdvancedPipeline)
- Follows configuration-driven approach
- Maintains quality standards and scientific rigor
- Supports the complete evaluation framework

This ensures consistency with the overall package design and maintains compatibility with existing workflows while providing enhanced batch processing capabilities.

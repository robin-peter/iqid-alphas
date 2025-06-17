# UCSF Workflow Examples

This documentation covers the comprehensive UCSF workflows available in the `examples/` directory for processing UCSF iQID and H&E histology data.

## Overview

The UCSF workflows demonstrate complete end-to-end processing of iQID and H&E histology data, specifically designed for UCSF dataset characteristics and requirements. We provide both individual workflow components and a consolidated workflow for comprehensive processing.

## Consolidated Workflow (Recommended)

### UCSF Consolidated Workflow
**Location:** `examples/ucsf_consolidated/`

**Purpose:** Comprehensive workflow that processes UCSF data through two main paths using the actual UCSF data structure.

**Data Structure Support:**
- **DataPush1**: Contains aligned iQID and H&E data
- **ReUpload**: Contains iQID raw data and preprocessed results  
- **Visualization**: Contains visualization results

**Processing Paths:**
1. **Path 1**: iQID Raw → Aligned processing
2. **Path 2**: Aligned iQID + H&E coregistration

**Key Features:**
- Works with actual UCSF data paths (`../data/UCSF-Collab/data/`)
- Organized intermediate and output file structure
- Comprehensive quality assessment and reporting
- Mock data fallback for testing without real data
- Integrated visualization workflow
- Complete logging and error handling

**Quick Start:**
```bash
cd examples/ucsf_consolidated
python ucsf_consolidated_workflow.py
```

**Documentation:** See `examples/ucsf_consolidated/README.md` for detailed usage.

## Individual Workflow Components

### Available Workflows

### 1. iQID Raw Data Alignment Workflow
**Location:** `examples/ucsf_workflows/workflow1_iqid_alignment.py`

**Purpose:** Process raw iQID image stacks through complete alignment pipeline for UCSF datasets.

**Key Features:**
- Handles UCSF-specific iQID data formats
- Automatic quality assessment and outlier detection
- Configurable reference frame selection
- Sub-pixel accuracy alignment
- Comprehensive metrics and reporting

**Configuration:** `examples/ucsf_workflows/configs/iqid_alignment_config.json`

### 2. H&E-iQID Co-registration Workflow
**Location:** `examples/ucsf_workflows/workflow2_he_iqid_coregistration.py`

**Purpose:** Co-register H&E histology images with aligned iQID data for tissue-specific analysis.

**Key Features:**
- Multi-modal image registration (H&E to iQID)
- Automatic tissue segmentation from H&E
- Activity quantification per tissue region
- Statistical analysis and heterogeneity metrics
- Publication-quality visualizations

**Configuration:** `examples/ucsf_workflows/configs/he_iqid_config.json`

### 3. Complete Analysis Pipeline
**Location:** `examples/ucsf_workflows/run_complete_pipeline.py`

**Purpose:** Orchestrate both workflows for complete end-to-end analysis with integrated reporting.

**Key Features:**
- Sequential execution of both workflows
- Comprehensive quality assessment
- Integrated reporting and visualization
- Error handling and recovery
- Human-readable summary generation

## Data Requirements

### UCSF Actual Data Structure (for Consolidated Workflow)
The consolidated workflow works with the actual UCSF data structure containing kidney and tumor samples:

```
../data/UCSF-Collab/data/
├── DataPush1/                   # Aligned iQID and H&E data
│   ├── HE/                     # H&E Stained Images
│   │   ├── 3D scans/           # 3D volumetric H&E data
│   │   │   ├── Kidney/         # Kidney samples (ID + laterality: L/R)
│   │   │   └── Tumor/          # Tumor samples (ID + measurement)
│   │   └── Sequential sections (10um)/  # 2D sequential sections
│   │       ├── Kidney/         # Sequential kidney sections
│   │       └── Tumor/          # Sequential tumor sections
│   └── iQID/                   # iQID Images (corrected/aligned)
│       ├── 3D scans/           # 3D volumetric iQID data
│       │   ├── Kidney/         # mBq_corr_*.tif files
│       │   └── Tumor/          # mBq_corr_*.tif files
│       └── Sequential sections/ # Sequential iQID sections
├── ReUpload/                   # Raw iQID and processing pipeline
│   └── iQID_reupload/
│       ├── 3D scans/           # 3D processing pipeline
│       │   ├── Kidney/         # Per-sample processing
│       │   │   └── [SampleID]/ # 0_*_iqid_event_image.tif (raw)
│       │   │       ├── 1_segmented/    # mBq_*.tif (segmented)
│       │   │       ├── 2_aligned/     # mBq_corr_*.tif (aligned)
│       │   │       └── Visualization/ # testAnimation_iQID.gif
│       │   └── Tumor/          # Similar structure
│       └── Sequential scans/   # Sequential processing pipeline
└── Visualization/              # Analysis results
    └── [SampleID]/            # testAnimation_both.gif, etc.
```

**Important**: All UCSF data directories are READ-ONLY. The workflow only reads from these locations and stores all results in organized subdirectories within the workflow folder.

### Legacy Example Data Structure (for Individual Workflows)
```
examples/ucsf_workflows/data/
├── raw_iqid/                    # Raw iQID TIFF stacks
│   ├── sample01_frame001.tif
│   ├── sample01_frame002.tif
│   └── ...
└── he_histology/               # H&E histology images
    ├── sample01_he.jpg
    └── ...
```

### Expected File Formats
- **iQID Data:** TIFF files (`.tif`, `.tiff`)
- **H&E Images:** JPEG or TIFF files (`.jpg`, `.jpeg`, `.tif`, `.tiff`)
- **Metadata:** JSON files (optional)

## Usage Examples

### Consolidated Workflow (Recommended)

#### Complete Workflow
```bash
cd examples/ucsf_consolidated
python ucsf_consolidated_workflow.py
```

#### Individual Processing Paths
```bash
# Run only Path 1 (iQID Raw → Aligned)
python ucsf_consolidated_workflow.py --path 1

# Run only Path 2 (Aligned iQID + H&E Coregistration)
python ucsf_consolidated_workflow.py --path 2
```

#### Data Validation
```bash
# Validate UCSF data paths
python ucsf_consolidated_workflow.py --validate-only

# Run complete validation and testing
python run_and_validate.py
```

#### Programmatic Usage
```python
from ucsf_consolidated_workflow import UCSFConsolidatedWorkflow

# Initialize workflow
workflow = UCSFConsolidatedWorkflow('configs/ucsf_data_config.json')

# Run complete workflow
results = workflow.run_complete_workflow()

# Access results
path1_results = results['path1_results']
path2_results = results['path2_results']
viz_results = results['visualization_results']
```

### Individual Workflows (Legacy)

#### Quick Start
```bash
cd examples/ucsf_workflows
python run_complete_pipeline.py
```

### Custom Configuration
```python
from workflow1_iqid_alignment import UCSFiQIDWorkflow

# Create custom configuration
custom_config = {
    "alignment": {
        "method": "phase_correlation",
        "reference_frame": "middle",
        "convergence_threshold": 1e-6
    },
    "quality_control": {
        "correlation_threshold": 0.7
    }
}

# Initialize workflow with custom config
workflow = UCSFiQIDWorkflow()
workflow.config.update(custom_config)

# Run workflow
results = workflow.run_complete_workflow("data/raw_iqid")
```

### Accessing Results
```python
# After running complete pipeline
results = run_complete_pipeline()

# Access aligned iQID data
aligned_stack = results['workflow1_results']['aligned_stack']

# Access tissue analysis
tissue_data = results['workflow2_results']['analysis_data']['tissue_activity_data']

# Access comprehensive summary
summary = results['comprehensive_summary']
```

## Output Structure

### Consolidated Workflow Outputs
```
examples/ucsf_consolidated/
├── intermediate/                    # Organized intermediate files
│   ├── path1_iqid_alignment/       # Path 1 intermediate results
│   ├── path2_coregistration/       # Path 2 intermediate results
│   └── visualization/              # Visualization intermediate files
├── outputs/                        # Final output files
│   ├── path1_aligned_iqid/         # Path 1 final results
│   ├── path2_coregistered/         # Path 2 final results
│   └── visualization/              # Final visualizations
├── logs/                           # Processing logs with timestamps
└── reports/                        # Comprehensive JSON reports
    └── consolidated_workflow_results_*.json
```

### Individual Workflow Outputs (Legacy)
```
examples/ucsf_workflows/outputs/
├── iqid_aligned/                    # Workflow 1 outputs
│   ├── aligned_iqid_stack.npy     # Main aligned data
│   ├── processing_report.json      # Alignment metrics
│   └── visualizations/             # Quality plots
├── he_iqid_analysis/               # Workflow 2 outputs
│   ├── tissue_activity_data.csv    # Quantitative data
│   ├── quantitative_analysis.json  # Complete analysis
│   └── visualizations/             # Overlay images
└── complete_analysis/              # Master pipeline outputs
    ├── comprehensive_analysis_report.json
    └── analysis_summary.txt
```

### Intermediate Files
```
examples/ucsf_workflows/intermediate/
├── iqid_alignment/                 # Workflow 1 intermediate
│   ├── preprocessed_stack.npy
│   ├── aligned_stack.npy
│   └── alignment_metrics.json
├── he_iqid_coregistration/        # Workflow 2 intermediate
│   ├── preprocessed_he.npy
│   ├── registered_he_gray.npy
│   └── segmentation/
└── *.log                          # Processing logs
```

## Configuration Options

### Workflow 1 Configuration
```json
{
  "preprocessing": {
    "gaussian_blur_sigma": 1.0,
    "normalize": true,
    "background_correction": true
  },
  "alignment": {
    "method": "phase_correlation",
    "reference_frame": "middle",
    "max_iterations": 100
  },
  "quality_control": {
    "correlation_threshold": 0.7,
    "max_displacement": 50
  }
}
```

### Workflow 2 Configuration
```json
{
  "registration": {
    "method": "feature_based",
    "feature_detector": "SIFT",
    "transformation_type": "affine"
  },
  "segmentation": {
    "tissue_segmentation_method": "otsu",
    "min_tissue_area": 500
  },
  "analysis": {
    "quantify_activity_per_tissue": true,
    "calculate_dose_metrics": true
  }
}
```

## Quality Assessment

### Automatic Quality Metrics
- **Alignment Correlation:** Measures frame-to-frame alignment quality
- **Registration Quality:** H&E to iQID registration assessment
- **Tissue Detection:** Validation of tissue segmentation
- **Activity Quantification:** Signal-to-noise ratio analysis

### Quality Thresholds
- **Good Quality:** Correlation > 0.7, multiple tissue regions detected
- **Fair Quality:** Correlation > 0.5, some tissue regions detected
- **Poor Quality:** Correlation < 0.5, manual review recommended

## Integration with iqid_alphas Package

### Using Package Components
```python
import iqid_alphas

# The workflows use the main package components
processor = iqid_alphas.IQIDProcessor()
aligner = iqid_alphas.ImageAligner()
segmenter = iqid_alphas.ImageSegmenter()
visualizer = iqid_alphas.Visualizer()

# Workflows can also be accessed as pipelines
advanced_pipeline = iqid_alphas.AdvancedPipeline()
combined_pipeline = iqid_alphas.CombinedPipeline()
```

### Extending Workflows
1. **Custom Processing Steps:** Add new analysis methods
2. **Additional Metrics:** Implement domain-specific calculations
3. **Enhanced Visualization:** Create custom plot types
4. **Integration Points:** Connect with external analysis tools

## Performance Considerations

### Memory Usage
- Large iQID stacks may require chunked processing
- Enable memory-efficient mode for limited resources
- Monitor intermediate file sizes

### Processing Time
- Typical processing time: 2-10 minutes per sample
- Depends on image size and number of frames
- Parallel processing available for batch analyses

### Storage Requirements
- Raw data: ~100MB per sample
- Intermediate files: ~200MB per sample
- Final outputs: ~50MB per sample

## Troubleshooting

### Common Issues
1. **Data Not Found:** Ensure data is in correct directories
2. **Low Registration Quality:** Adjust preprocessing parameters
3. **No Tissue Detected:** Modify segmentation thresholds
4. **Memory Errors:** Enable chunked processing

### Debug Information
- Check processing logs in `intermediate/`
- Examine intermediate outputs
- Validate configuration parameters
- Review quality metrics

## Best Practices

### Data Preparation
1. **Consistent Naming:** Use systematic file naming conventions
2. **Quality Control:** Pre-screen data for artifacts
3. **Metadata:** Include acquisition parameters when available
4. **Backup:** Maintain copies of raw data

### Analysis Workflow
1. **Configuration Review:** Validate parameters before processing
2. **Quality Assessment:** Check intermediate results
3. **Documentation:** Record analysis parameters and decisions
4. **Validation:** Compare results with manual analysis when possible

## Research Applications

### Dosimetry Studies
- Quantify radiation dose distribution in tissue
- Calculate dose-volume histograms
- Assess spatial dose heterogeneity

### Biodistribution Analysis
- Map radiopharmaceutical uptake patterns
- Correlate activity with tissue morphology
- Track temporal changes in distribution

### Clinical Translation
- Validate imaging biomarkers
- Support treatment planning
- Enable personalized therapy approaches

---

For more information, see the complete workflow documentation in `examples/ucsf_workflows/README.md`.

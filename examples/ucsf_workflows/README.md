# UCSF iQID-H&E Analysis Workflows - Updated for Real UCSF Data

This directory contains comprehensive workflows specifically designed for processing UCSF iQID and H&E histology datasets. The workflows have been **updated to use the actual UCSF data structure** and enforce a **readonly policy** for data safety.

## 🔄 **Recent Updates (v2.0.0)**

### ✅ **Configuration Updates**
- **Real UCSF Data Integration**: Configurations updated to match actual UCSF directory structure
- **Readonly Policy Enforcement**: All outputs written outside readonly UCSF data directory  
- **Enhanced Safety**: Clear separation between readonly inputs and writable outputs
- **Hierarchical Data Paths**: Detailed sub-path organization for DataPush1, ReUpload, Visualization, and Metadata
- **Path Validation**: Updated file patterns to match actual UCSF naming conventions

### ✅ **Code Updates (June 2025)**
- **Dynamic Path Resolution**: All workflows now read data paths from hierarchical configs instead of hardcoded paths
- **Dataset Auto-Selection**: Workflows automatically select the first available dataset (DataPush1, ReUpload, etc.)
- **Readonly Compliance**: All Python files check and respect readonly policies from configs
- **Safety Features**: Enhanced output isolation and data protection
- **Graceful Fallbacks**: Workflows use simulated data if real UCSF data not available

### ✅ **Validation & Testing**
- **Config Validation**: All updated configurations pass validation script (`validate_configs.py`)
- **Code Validation**: All Python files compile and initialize successfully
- **Integration Ready**: Workflows are ready for use with real UCSF data structure

## 📁 Directory Structure

```
ucsf_workflows/
├── README.md                                    # This file (updated)
├── run_complete_pipeline.py                    # Master workflow orchestrator (updated)
├── workflow1_iqid_alignment.py                # iQID raw → aligned processing (updated)
├── workflow2_he_iqid_coregistration.py       # H&E + iQID co-registration (updated)
├── configs/                                   # Configuration files
│   ├── iqid_alignment_config.json            # Workflow 1 config (updated v2.0.0)
│   └── he_iqid_config.json                   # Workflow 2 config (updated v2.0.0)
├── validate_configs.py                       # Config validation script (new)
├── CONFIG_UPDATE_SUMMARY.md                  # Config change details (new)
├── CODE_UPDATE_SUMMARY.md                    # Code change details (new)
├── data/                                      # Input data directory
│   ├── raw_iqid/                             # Raw iQID TIFF stacks
│   └── he_histology/                         # H&E histology images
├── intermediate/                              # Intermediate processing files
│   ├── iqid_alignment/                       # Workflow 1 intermediate files
│   ├── he_iqid_coregistration/              # Workflow 2 intermediate files
│   └── *.log                                 # Processing logs
└── outputs/                                  # Final outputs
    ├── iqid_aligned/                         # Aligned iQID outputs
    ├── he_iqid_analysis/                     # Co-registration analysis
    └── complete_analysis/                    # Comprehensive reports
```

## 🔬 Workflows Overview

### Workflow 1: iQID Raw Data Alignment
**File:** `workflow1_iqid_alignment.py`

**Purpose:** Process raw iQID image stacks through complete alignment pipeline.

**Pipeline Steps:**
1. **Load Raw Data** - Import UCSF iQID TIFF stacks
2. **Preprocessing** - Noise reduction, background correction, normalization
3. **Quality Assessment** - Frame quality checks and outlier detection
4. **Inter-frame Alignment** - Register all frames to reference frame
5. **Quality Control** - Validate alignment results
6. **Output Generation** - Save aligned stack and metrics

**Key Features:**
- Automatic reference frame selection (first, middle, or max intensity)
- Sub-pixel accuracy alignment
- Comprehensive quality metrics
- Outlier frame detection
- Detailed logging and intermediate file saving

### Workflow 2: H&E-iQID Co-registration
**File:** `workflow2_he_iqid_coregistration.py`

**Purpose:** Co-register H&E histology with aligned iQID data for tissue analysis.

**Pipeline Steps:**
1. **Load Aligned iQID** - Import results from Workflow 1
2. **Load H&E Images** - Import corresponding histology images
3. **H&E Preprocessing** - Stain normalization, artifact removal
4. **Registration** - Align H&E to iQID coordinate system
5. **Tissue Segmentation** - Identify tissue regions from H&E
6. **Activity Mapping** - Map iQID activity to tissue regions
7. **Quantitative Analysis** - Calculate tissue-specific activity metrics

**Key Features:**
- Multiple registration methods (feature-based, intensity-based)
- Automatic tissue segmentation
- Activity quantification per tissue region
- Statistical analysis and heterogeneity metrics
- Publication-quality visualization outputs

### Master Pipeline: Complete Analysis
**File:** `run_complete_pipeline.py`

**Purpose:** Orchestrate both workflows for complete end-to-end analysis.

**Features:**
- Sequential execution of both workflows
- Comprehensive quality assessment
- Integrated reporting
- Human-readable summary generation
- Error handling and recovery

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure iqid_alphas package is available
cd /path/to/iqid-alphas
python -c "import iqid_alphas; print('Package available')"
```

### Running Individual Workflows

**Option 1: Run Complete Pipeline (Recommended)**
```bash
cd examples/ucsf_workflows
python run_complete_pipeline.py
```

**Option 2: Run Individual Workflows**
```bash
# Step 1: iQID Alignment
python workflow1_iqid_alignment.py

# Step 2: H&E Co-registration (requires Step 1 output)
python workflow2_he_iqid_coregistration.py
```

### Data Preparation

**For Real UCSF Data:**
1. Place raw iQID TIFF files in `data/raw_iqid/`
2. Place H&E histology images in `data/he_histology/`
3. Update config files with actual file patterns and parameters

**For Demonstration:**
- Workflows automatically generate simulated data if real data is not available
- Simulated data mimics realistic UCSF dataset characteristics

## ⚙️ Configuration

### Workflow 1 Configuration (`configs/iqid_alignment_config.json`)

Key parameters:
```json
{
  "preprocessing": {
    "gaussian_blur_sigma": 1.0,
    "background_correction": true,
    "noise_threshold": 0.1
  },
  "alignment": {
    "method": "phase_correlation",
    "reference_frame": "middle",
    "convergence_threshold": 1e-6
  },
  "quality_control": {
    "correlation_threshold": 0.7,
    "max_displacement": 50
  }
}
```

### Workflow 2 Configuration (`configs/he_iqid_config.json`)

Key parameters:
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
    "calculate_dose_metrics": true,
    "heterogeneity_metrics": ["coefficient_of_variation", "entropy"]
  }
}
```

## 🗂️ **Updated Configuration for Real UCSF Data (v2.0.0)**

### UCSF Data Structure Mapping

**Source Data Locations (READ-ONLY):**
```
../data/UCSF-Collab/data/
├── DataPush1/
│   ├── H&E/                    # H&E histology images
│   └── iQID/
│       ├── 3D scans/           # 3D iQID scans  
│       └── Sequential sections/ # Sequential iQID sections
└── ReUpload/
    ├── H&E_reupload/           # Additional H&E images
    └── iQID_reupload/
        ├── 3D scans/           # Additional 3D scans
        └── Sequential scans/    # Additional sequential scans
```

**Output Directories (WRITABLE):**
```
examples/ucsf_workflows/
├── outputs/
│   ├── he_iqid_coregistration/ # H&E-iQID co-registration results
│   └── iqid_aligned/          # iQID alignment results
├── intermediate/              # Intermediate processing files
├── logs/                     # Workflow execution logs
└── reports/                  # Analysis reports
```

### Updated Configuration Features

**H&E-iQID Config (`he_iqid_config.json`):**
- **Real Data Sources**: Direct paths to UCSF H&E and iQID directories
- **Readonly Policy**: All outputs written to local writable directories
- **Tissue Patterns**: Updated patterns for kidney (`D*M*(P*)_[LR]`) and tumor (`D* M* tumor *um`) samples
- **Safety Warnings**: Clear readonly policy enforcement

**iQID Alignment Config (`iqid_alignment_config.json`):**
- **UCSF File Patterns**: Matches actual UCSF naming conventions
- **Multiple Sources**: Supports both DataPush1 and ReUpload directories
- **Quality Validation**: Enhanced quality control for UCSF data characteristics
- **Timestamped Outputs**: Prevents accidental overwrites

### Running with Real UCSF Data

```bash
# Ensure UCSF data is mounted at ../data/UCSF-Collab/data/
ls -la ../data/UCSF-Collab/data/

# Run workflows (will automatically use updated configs)
python run_complete_pipeline.py

# Or run individually:
python workflow1_iqid_alignment.py --config configs/iqid_alignment_config.json
python workflow2_he_iqid_coregistration.py --config configs/he_iqid_config.json
```

### Running with Simulated Data

```bash
# Force simulation mode (bypass data checks)
python run_complete_pipeline.py --simulate

# Or for individual workflows:
python workflow1_iqid_alignment.py --simulate
python workflow2_he_iqid_coregistration.py --simulate
```

## 📊 Output Files

### Workflow 1 Outputs (`outputs/iqid_aligned/`)
- `aligned_iqid_stack.npy` - Aligned iQID image stack
- `processing_report.json` - Alignment metrics and quality assessment
- `visualizations/` - Alignment quality plots and preview images

### Workflow 2 Outputs (`outputs/he_iqid_analysis/`)
- `tissue_activity_data.csv` - Quantitative data per tissue region
- `quantitative_analysis.json` - Complete analysis results
- `visualizations/` - Overlay images and analysis plots
- `comprehensive_analysis_report.json` - Detailed technical report

### Complete Pipeline Outputs (`outputs/complete_analysis/`)
- `comprehensive_analysis_report.json` - Combined results from both workflows
- `analysis_summary.txt` - Human-readable summary report

### Intermediate Files
- `intermediate/iqid_alignment/` - Workflow 1 intermediate processing files
- `intermediate/he_iqid_coregistration/` - Workflow 2 intermediate files
- `intermediate/*.log` - Detailed processing logs

## 🔍 Quality Assessment

The workflows include comprehensive quality assessment:

### Workflow 1 Quality Metrics
- **Frame Correlation** - Measures alignment quality between frames
- **Displacement Tracking** - Monitors inter-frame movement
- **Intensity Stability** - Checks for acquisition artifacts
- **Outlier Detection** - Identifies problematic frames

### Workflow 2 Quality Metrics
- **Registration Correlation** - H&E to iQID alignment quality
- **Tissue Coverage** - Percentage of image containing tissue
- **Activity Signal-to-Noise** - Quality of activity detection
- **Segmentation Validation** - Tissue region identification accuracy

## 🎯 Use Cases

### Research Applications
- **Dosimetry Studies** - Quantify radiation dose distribution in tissue
- **Biodistribution Analysis** - Map radiopharmaceutical uptake patterns
- **Heterogeneity Assessment** - Measure spatial activity variation
- **Correlation Studies** - Relate activity to tissue morphology

### Clinical Applications
- **Treatment Planning** - Inform targeted radiotherapy protocols
- **Therapy Monitoring** - Track treatment response over time
- **Biomarker Development** - Identify predictive imaging features
- **Quality Assurance** - Validate imaging and analysis protocols

## 🛠️ Customization

### Adding New Analysis Metrics
1. Extend the analysis functions in `workflow2_he_iqid_coregistration.py`
2. Update configuration files to include new parameters
3. Modify output generation to include new metrics

### Integrating Different Registration Methods
1. Add new registration algorithms to the `ImageAligner` class
2. Update configuration options
3. Implement quality assessment for new methods

### Custom Visualization
1. Extend the `Visualizer` class with new plot types
2. Update workflow visualization functions
3. Configure output formats and styling

## 📝 Troubleshooting

### Common Issues

**Issue: "No iQID data found"**
- Solution: Ensure data is in `data/raw_iqid/` or workflows will use simulated data

**Issue: "Registration correlation too low"**
- Solution: Adjust registration parameters in config files
- Check H&E image quality and preprocessing settings

**Issue: "No tissue regions detected"**
- Solution: Modify segmentation thresholds in configuration
- Verify H&E image preprocessing is appropriate

### Performance Optimization
- **Memory Usage** - Enable chunked processing for large datasets
- **Processing Speed** - Use multiprocessing for independent operations
- **Storage** - Disable intermediate file saving if space is limited

## 📚 References

### UCSF iQID Technology
- Scientific Reports papers on iQID methodology
- UCSF imaging protocols and standards

### Image Processing Methods
- Phase correlation alignment algorithms
- Feature-based registration techniques
- Tissue segmentation approaches

## 📞 Support

For questions about these workflows:
1. Check the processing logs in `intermediate/`
2. Review configuration parameters
3. Examine intermediate outputs for debugging
4. Contact the IQID-Alphas development team

---

**Last Updated:** June 2025  
**Version:** 1.0.0  
**Compatibility:** IQID-Alphas v1.0.0+

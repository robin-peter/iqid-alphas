# UCSF Consolidated Workflow

This directory contains a comprehensive, consolidated workflow for processing UCSF iQID and H&E data using the modernized `iqid_alphas` package. The workflow supports two main processing paths based on the actual UCSF data structure.

## 📁 UCSF Data Structure

The workflow is designed to work with the actual UCSF data located at `../data/UCSF-Collab/data/` with the following detailed structure:

### DataPush1 - Aligned iQID and H&E Data
```
../data/UCSF-Collab/data/DataPush1/
├── HE/                          # H&E Stained Images
│   ├── 3D scans/               # 3D volumetric H&E data
│   │   ├── Kidney/             # Kidney samples by ID and laterality
│   │   │   ├── D1M1/           # Individual sample ID
│   │   │   │   ├── L/          # Left kidney (L) or Right (R)
│   │   │   │   │   ├── P1.tif  # TIFF image slices
│   │   │   │   │   ├── P2.tif
│   │   │   │   │   └── ...
│   │   │   └── D7M1/...
│   │   └── Tumor/              # Tumor samples by ID and measurement
│   │       ├── D1 M1 tumor 200um/
│   │       │   ├── P1.tif
│   │       │   └── ...
│   │       └── ...
│   └── Sequential sections (10um)/  # 2D sequential sections
│       ├── Kidney/             # Sequential kidney sections
│       │   ├── *_Lower_*.tif   # Lower sections
│       │   └── *_Upper_*.tif   # Upper sections
│       └── Tumor/              # Sequential tumor sections
│           ├── *_Lower_*.tif
│           └── *_Upper_*.tif
└── iQID/                       # iQID Images (aligned/corrected)
    ├── 3D scans/              # 3D volumetric iQID data
    │   ├── Kidney/            # Kidney samples
    │   │   ├── D1M1(P1)_L/    # ID + laterality
    │   │   │   ├── mBq_corr_1.tif  # Corrected iQID in mBq units
    │   │   │   ├── mBq_corr_2.tif
    │   │   │   └── ...
    │   │   └── ...
    │   └── Tumor/             # Tumor samples
    │       ├── D1M1/          # Sample ID
    │       │   ├── mBq_corr_1.tif
    │       │   └── ...
    │       └── ...
    └── Sequential sections/   # Sequential iQID sections
        ├── Kidney/           
        └── Tumor/
```

### ReUpload - Raw iQID and Processing Pipeline
```
../data/UCSF-Collab/data/ReUpload/
└── iQID_reupload/
    ├── 3D scans/              # 3D iQID processing pipeline
    │   ├── Kidney/
    │   │   └── D1M1(P1)_L/    # Individual sample
    │   │       ├── 0_*_iqid_event_image.tif    # RAW iQID event data
    │   │       ├── 1_segmented/                # Segmentation results
    │   │       │   ├── mBq_1.tif
    │   │       │   └── ...
    │   │       ├── 2_aligned/                  # Manually aligned results
    │   │       │   ├── mBq_corr_1.tif
    │   │       │   └── ...
    │   │       ├── LR_ims/                     # Left/Right images (optional)
    │   │       │   ├── mBq_1.tif
    │   │       │   └── ...
    │   │       └── Visualization/              # Visual QC (optional)
    │   │           └── testAnimation_iQID.gif
    │   └── Tumor/
    │       └── [similar structure]
    └── Sequential scans/      # Sequential processing pipeline
        ├── Kidney/
        └── Tumor/
```

### Visualization - Analysis Results
```
../data/UCSF-Collab/data/Visualization/
├── D1M1/                      # Sample-specific visualizations
│   ├── testAnimation_both.gif     # Combined iQID+H&E animation
│   ├── testAnimation_HE.gif       # H&E-only animation
│   └── testAnimation_iQID_full.gif # Full iQID animation
└── [other samples]/
```

### Additional Files
```
../data/UCSF-Collab/data/
├── iqid_data.txt              # iQID measurement data
├── readme.docx                # Dataset documentation
├── metadata_TJU.csv           # Clinical/experimental metadata
└── quantitative_notes.docx    # Quantitative analysis notes
```

⚠️ **CRITICAL: All UCSF data directories are READ-ONLY!** The workflow only reads from these locations and stores all processing results in organized subdirectories within the workflow folder.

## 🚀 Processing Paths

### Path 1: iQID Raw → Aligned Processing
- **Input**: Raw iQID event data from `ReUpload/iQID_reupload/*/0_*_iqid_event_image.tif`
- **Reference**: Manual alignments from `ReUpload/iQID_reupload/*/2_aligned/mBq_corr_*.tif`
- **Output**: Validated aligned iQID data ready for analysis
- **Processing Steps**:
  1. **Load Raw iQID Events**: Process raw event images from both 3D and sequential scans
  2. **Preprocess Event Data**: Clean and normalize event data
  3. **Segment Tissues**: Identify kidney vs tumor tissue regions
  4. **Align Sequences**: Temporal and spatial alignment of event sequences
  5. **Validate Against Manual**: Compare with manually aligned reference data
  6. **Save Aligned Data**: Export validated aligned sequences

### Path 2: Aligned iQID + H&E Coregistration
- **Input**: 
  - Aligned iQID from `DataPush1/iQID/*/mBq_corr_*.tif`
  - H&E images from `DataPush1/HE/*/P*.tif` (3D) or `*_Lower_*.tif/*_Upper_*.tif` (sequential)
- **Output**: Coregistered iQID-H&E data pairs with tissue-specific quantification
- **Processing Steps**:
  1. **Load Aligned iQID**: Process corrected iQID data from DataPush1
  2. **Load H&E Images**: Process H&E stained tissue images
  3. **Extract Features**: Multi-modal feature extraction for registration
  4. **Registration Alignment**: Mutual information-based coregistration
  5. **Tissue Segmentation**: H&E-guided tissue region identification
  6. **Activity Quantification**: Tissue-specific iQID activity measurements
  7. **Quality Validation**: Validate coregistration accuracy
  8. **Save Coregistered Data**: Export aligned pairs with quantitative data

### Visualization Workflow
- **Input**: Results from both processing paths + reference visualizations from `Visualization/`
- **Output**: Comprehensive visualizations and quality reports
- **Features**:
  - **Sample-Specific Analysis**: Individual sample processing and QC
  - **Comparison Plots**: Before/after processing comparisons
  - **Tissue Activity Maps**: Quantitative activity distribution visualizations
  - **Quality Assessment**: Registration accuracy and processing quality metrics
  - **Interactive Visualizations**: GIF animations and overlay images
  - **Statistical Reporting**: Cross-sample analysis and heterogeneity assessment

## 📂 Directory Structure

```
ucsf_consolidated/
├── configs/
│   └── ucsf_data_config.json        # Main configuration file
├── intermediate/                     # Organized intermediate files
│   ├── path1_iqid_alignment/        # Path 1 intermediate results
│   ├── path2_coregistration/        # Path 2 intermediate results
│   └── visualization/               # Visualization intermediate files
├── outputs/                         # Final output files
│   ├── path1_aligned_iqid/          # Path 1 final results
│   ├── path2_coregistered/          # Path 2 final results
│   └── visualization/               # Final visualizations
├── logs/                            # Processing logs
├── reports/                         # Comprehensive reports
├── ucsf_consolidated_workflow.py    # Main workflow script
├── test_consolidated_workflow.py    # Test suite
└── README.md                        # This file
```

## 🔧 Configuration

The main configuration is in `configs/ucsf_data_config.json` and includes:

- **Data Paths**: References to actual UCSF data locations
- **Workflow Parameters**: Processing settings for each path
- **Output Organization**: How results are organized and stored
- **Quality Thresholds**: Criteria for processing validation

## 🚀 Usage

### NEW: Batch Processing (Recommended)
Process ALL available samples with comprehensive visualization and quality assessment:

```bash
cd examples/ucsf_consolidated

# Process all samples with comprehensive visualizations
python run_batch_processing.py

# Quick demo with first 3 samples only
python run_batch_processing.py --quick

# Demo mode with mock data (for testing without real UCSF data)
python run_batch_processing.py --demo

# Process specific number of samples
python run_batch_processing.py --samples 10

# Enable verbose logging
python run_batch_processing.py --verbose
```

**Batch Processing Features:**
- 🔍 **Automatic Sample Discovery**: Finds all available UCSF samples
- 📊 **Individual Sample Visualization**: Quality dashboard, alignment plots, activity maps
- 📈 **Comprehensive Summary**: Batch overview, quality metrics, statistical analysis
- 🎯 **Quality Assessment**: Alignment scores, registration quality, processing efficiency
- 📁 **Organized Outputs**: Structured results with individual and summary visualizations

**Generated Visualizations:**
- Batch processing dashboard with overall statistics
- Quality metrics summary and distribution analysis
- Sample comparison radar charts
- Processing performance analysis
- Individual sample quality dashboards
- Alignment quality over time plots
- Activity distribution histograms

### Individual Sample Processing (Original Workflow)

### Run Complete Workflow
```bash
cd examples/ucsf_consolidated
python ucsf_consolidated_workflow.py
```

### Run Individual Paths
```bash
# Run only Path 1 (iQID Raw → Aligned)
python ucsf_consolidated_workflow.py --path 1

# Run only Path 2 (Aligned iQID + H&E Coregistration)  
python ucsf_consolidated_workflow.py --path 2
```

### Validate Data Paths Only
```bash
python ucsf_consolidated_workflow.py --validate-only
```

### Custom Configuration
```bash
python ucsf_consolidated_workflow.py --config custom_config.json
```

## 🧪 Testing

Run the test suite to validate the workflow:

```bash
python test_consolidated_workflow.py
```

The test suite includes:
- Unit tests for individual components
- Integration tests for complete workflows
- Mock data testing (when real data is not available)
- Error handling validation

## 📊 Output Organization

All intermediate and output files are organized in dedicated subdirectories:

### Intermediate Files (`intermediate/`)
- `path1_iqid_alignment/`: Temporary files from iQID alignment processing
- `path2_coregistration/`: Temporary files from coregistration processing  
- `visualization/`: Temporary visualization files

### Output Files (`outputs/`)
- `path1_aligned_iqid/`: Final aligned iQID data
- `path2_coregistered/`: Final coregistered iQID-H&E pairs
- `visualization/`: Final plots, overlays, and reports

### Logs and Reports
- `logs/`: Detailed processing logs with timestamps
- `reports/`: Comprehensive JSON reports with processing statistics

## 🔍 Quality Assessment

The workflow includes comprehensive quality assessment:

### Path 1 Quality Metrics
- Alignment score (spatial consistency)
- Temporal consistency across frames
- Signal-to-noise ratio improvements
- Processing completion statistics

### Path 2 Quality Metrics
- Registration accuracy
- Mutual information between modalities
- Correlation coefficients
- Validation against manual alignments

### Visualization Quality
- Plot generation success rates
- Overlay accuracy assessments
- Report completeness checks

## 🛠️ Integration with iqid_alphas Package

The workflow leverages the modernized `iqid_alphas` package components:

- **Core Modules**: `processor.py`, `alignment.py`, `segmentation.py`
- **Pipeline Classes**: `advanced.py` for complex processing workflows
- **Visualization**: `plotter.py` for comprehensive visualization
- **Utils**: Helper functions for data handling and validation

## 📈 Performance Considerations

- **Memory Management**: Processes large datasets in chunks
- **Parallel Processing**: Utilizes multiple cores when available
- **Intermediate File Cleanup**: Configurable retention policies
- **Progress Tracking**: Detailed logging and progress reporting

## 🚨 Error Handling

- **Data Validation**: Checks for required input files
- **Graceful Degradation**: Continues processing when possible
- **Mock Data Fallback**: Creates test data when real data unavailable
- **Comprehensive Logging**: Detailed error reporting and debugging

## 📝 Example Results

After successful execution, you'll find:

1. **Aligned iQID Data**: High-quality aligned sequences in `outputs/path1_aligned_iqid/`
2. **Coregistered Pairs**: iQID-H&E aligned pairs in `outputs/path2_coregistered/`
3. **Visualizations**: Comprehensive plots and overlays in `outputs/visualization/`
4. **Quality Reports**: Detailed processing reports in `reports/`
5. **Processing Logs**: Complete execution logs in `logs/`

## 🔗 Related Documentation

- Main repository README: `../../README.md`
- UCSF Workflows Guide: `../../docs/examples/ucsf_workflows.md`
- API Documentation: `../../docs/api_reference/`
- User Guides: `../../docs/user_guides/`

## ⚠️ Important Notes

- **Data Privacy**: Ensure UCSF data handling complies with institutional policies
- **Computational Requirements**: Large datasets may require significant memory and processing time
- **Version Compatibility**: Designed for the modernized `iqid_alphas` package structure
- **Path Dependencies**: Verify data paths exist before running workflows

## 🤝 Contributing

To extend or modify the consolidated workflow:

1. Update configuration in `configs/ucsf_data_config.json`
2. Modify processing steps in `ucsf_consolidated_workflow.py`
3. Add corresponding tests in `test_consolidated_workflow.py`  
4. Update documentation in this README

For questions or issues, refer to the main repository documentation or create an issue in the project repository.

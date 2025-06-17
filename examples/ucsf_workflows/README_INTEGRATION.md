# UCSF iQID-H&E Workflow Integration

This directory contains the integrated and modernized UCSF iQID/H&E workflows that can process real UCSF data with robust sample matching and automated batch processing.

## Key Features

✅ **Unified Configuration**: Single `unified_config.json` file with base path and relative paths  
✅ **Real Data Integration**: Automatic discovery and matching of H&E and iQID samples  
✅ **Sample Matching**: Intelligent matching between H&E and iQID data using sample IDs  
✅ **Batch Processing**: Process all available samples automatically  
✅ **Interactive Mode**: Choose specific samples or processing options  
✅ **Read-only Safety**: All UCSF data remains read-only, outputs saved locally  

## Files Overview

### Core Components
- `ucsf_data_loader.py` - Data discovery and sample matching utility
- `configs/unified_config.json` - Unified configuration for both workflows

### Workflow Scripts
- `workflow1_iqid_alignment.py` - iQID raw data alignment workflow (updated for real data)
- `workflow2_he_iqid_coregistration.py` - H&E-iQID co-registration workflow (updated for real data)

### Batch Processing Scripts
- `run_all_samples.py` - Simple batch processor for all samples
- `interactive_workflow_runner.py` - Interactive interface for sample selection and processing

### Testing Scripts
- `test_data_matcher_simple.py` - Test the data matching functionality
- `test_data_matcher.py` - Original validation script

## Quick Start

### 1. Check Available Samples
```bash
python test_data_matcher_simple.py
```

### 2. Process All Samples (Automated)
```bash
python run_all_samples.py
```

### 3. Interactive Processing
```bash
python interactive_workflow_runner.py
```

## Sample Matching

The data loader automatically matches samples between H&E and iQID data:

**H&E Pattern**: `D1M1_L` (Day 1, Mouse 1, Left side)  
**iQID Pattern**: `D1M1(P1)_L` (Day 1, Mouse 1, Position 1, Left side)

The system discovers:
- **H&E Data**: From `DataPush1/HE/{3D|Sequential sections (10um)}/{kidney|tumor}/`
- **iQID Processed**: From `DataPush1/iQID/{3D|Sequential sections}/{kidney|tumor}/`  
- **iQID Raw**: From `ReUpload/iQID_reupload/iQID/{3D|Sequential scans}/{kidney|tumor}/`

## Current Status

✅ **Data Discovery**: Successfully finds and matches real UCSF samples  
✅ **Sample Loading**: Loads file paths and metadata for both H&E and iQID data  
✅ **Batch Processing**: Processes all available samples with progress tracking  
✅ **Results Organization**: Creates organized output structure per sample  

⚠️ **Note**: Current implementation uses simulated processing for demonstration. To enable actual image processing, uncomment the real data loading sections in the workflow files and ensure required image processing libraries are installed.

## Example Output

```
🔬 UCSF Complete Workflow Runner
==================================================
📊 Found 2 samples to process
📋 Sample summary:
   - Total samples: 2
   - Kidney samples: 2  
   - Tumor samples: 0

============================================================
Processing sample 1/2: D1M1_L
============================================================
   📁 Sample data loaded - H&E: 18 files, iQID: reupload
   🔄 Step 1: iQID Alignment
   ✅ iQID alignment completed - Quality score: 8.7
   🔄 Step 2: H&E-iQID Co-registration  
   ✅ Co-registration completed - Registration quality: good
   ✅ Sample D1M1_L processed successfully
```

## Output Structure

```
outputs/
├── batch_processing_results.json     # Comprehensive batch results
├── batch_processing_summary.txt      # Human-readable summary
├── iqid_aligned/                     # iQID alignment results
│   ├── D1M1_L/
│   │   └── D1M1_L_iqid_results.json
│   └── D1M1_R/
│       └── D1M1_R_iqid_results.json
└── he_iqid_analysis/                 # H&E-iQID co-registration results
    ├── D1M1_L/
    │   └── D1M1_L_coregistration_results.json
    └── D1M1_R/
        └── D1M1_R_coregistration_results.json
```

## Configuration

The unified configuration supports:
- **Data Paths**: Base path + relative paths for all data locations
- **File Patterns**: Flexible patterns for different file types
- **Processing Parameters**: Alignment, registration, and analysis settings
- **Output Options**: Customizable output formats and locations

## Next Steps

To enable full real data processing:
1. Install image processing dependencies (`PIL`, `opencv-python`, `scikit-image`, `tifffile`)
2. Uncomment real data loading sections in workflow files
3. Implement actual alignment and registration algorithms
4. Add real visualization generation

## Support

For questions or issues, contact: Wookjin Choi <wookjin.choi@jefferson.edu>

# IQID-Alphas

A Python-based framework for listmode data processing, image processing, and dosimetry using the iQID camera digital autoradiograph.

Repository author: Robin Peter  
iQID camera expert: Brian Miller  

Please contact the authors with any questions or for access to data samples.

## Papers
- (2022, Sci Rep, initial methods): https://doi.org/10.1038/s41598-022-22664-5
- (2024, Sci Rep, 3D sub-organ dosimetry and TCP): https://doi.org/10.1038/s41598-024-70417-3

Permanent DOI of the initial repository release in 2022: [![DOI](https://zenodo.org/badge/540307496.svg)](https://zenodo.org/badge/latestdoi/540307496)

## Repository Organization

This repository has been reorganized for production use with a clean, modular structure supporting two main processing pipelines:

1. **iQID-Only Pipeline** - For ReUpload data processing
2. **Combined H&E-iQID Pipeline** - For DataPush1 data with H&E and iQID co-registration

## Quick Start

### Installation
```bash
git clone <repository-url>
cd iqid-alphas
pip install -r requirements.txt
```

### Running Pipelines
```bash
# iQID-only processing
python pipelines/simplified_iqid_pipeline.py --config configs/iqid_pipeline_config.json

# Combined H&E-iQID processing  
python pipelines/combined_he_iqid_pipeline.py --config configs/combined_pipeline_config.json
```

### Running Tests
```bash
python tests/test_value_ranges.py
python tests/test_value_range_batch.py
```

## Repository Structure

```
iqid-alphas/
├── README.md                    # This file
├── LICENSE.txt                  # License information
├── requirements.txt             # Python dependencies
├── ORGANIZATION_SUMMARY.md      # Detailed organization summary
├── .gitignore                   # Git ignore patterns
│
├── src/                         # Main source code
│   ├── core/                    # Core iQID functionality
│   │   └── iqid/               # Original iQID modules
│   │       ├── align.py        # Alignment and registration
│   │       ├── dpk.py          # Dose kernel processing
│   │       ├── helper.py       # Helper functions
│   │       ├── process_object.py # Data processing classes
│   │       └── spec.py         # Spectroscopy functions
│   │
│   ├── alignment/              # H&E and iQID alignment tools
│   ├── processing/             # Enhanced processing modules
│   ├── segmentation/           # Activity and tissue segmentation
│   ├── utils/                  # Utility functions
│   └── visualization/          # Plotting and visualization
│
├── pipelines/                  # Processing pipelines
│   ├── iqid_only_pipeline.py           # iQID-only processing
│   ├── combined_he_iqid_pipeline.py    # Combined H&E-iQID processing
│   └── simplified_iqid_pipeline.py     # Simplified demo pipeline
│
├── configs/                    # Configuration files
│   ├── config_index.json              # Configuration index
│   ├── iqid_pipeline_config.json      # iQID-only config
│   ├── combined_pipeline_config.json  # Combined pipeline config
│   └── test_config.json               # Test configuration
│
├── tests/                      # Test files
│   ├── test_value_ranges.py           # Value range analysis tests
│   └── test_value_range_batch.py      # Batch processing tests
│
├── docs/                       # Documentation
│   ├── design_document.md             # Design documentation
│   ├── iqid_align_documentation.md    # Alignment documentation
│   ├── iqid_helper_documentation.md   # Helper documentation
│   ├── iqid_process_object_documentation.md # Processing documentation
│   └── ucsf_ac225_iqid_processing_plan.md  # UCSF processing plan
│
├── scripts/                    # Utility scripts
│   ├── organize_repo.py               # Repository organization
│   ├── setup_project.py               # Project setup
│   ├── final_validation.py            # Validation scripts
│   └── run_ucsf_tests.py              # UCSF-specific tests
│
├── archive/                    # Legacy files (preserved)
│   ├── automate_*.py                  # Original automation scripts
│   ├── test_*.py                      # Original test files
│   ├── demo_notebooks/                # Original demo notebooks
│   ├── iqid/                          # Original iqid module
│   └── misc_notebooks/                # Miscellaneous notebooks
│
└── outputs/                    # Output directory (gitignored)
```

## Core Modules

### src/core/iqid/
The core iQID processing modules (updated with proper imports):
- **align.py** - Image alignment and registration functions
- **dpk.py** - Dose point kernel processing and convolution
- **helper.py** - Utility functions for plotting and calculations  
- **process_object.py** - ClusterData class for listmode data processing
- **spec.py** - Gamma spectroscopy functions (requires becquerel package)

### Pipelines
- **iqid_only_pipeline.py** - Process iQID data from ReUpload directory
- **combined_he_iqid_pipeline.py** - Process combined H&E and iQID data from DataPush1
- **simplified_iqid_pipeline.py** - Simplified pipeline for demonstration

### Configuration Management
- Centralized JSON-based configuration system
- Separate configs for different processing modes
- Easy parameter adjustment without code changes

## Dependencies

Core requirements are listed in `requirements.txt`. Key dependencies include:
- numpy (≥1.20.2)
- opencv-python (≥4.0.1) 
- scikit-image (≥0.18.1)
- scipy (≥1.6.2)
- PyStackReg (≥0.2.5)
- matplotlib (for visualization)
- tifffile (for TIFF handling)

Optional dependencies:
- becquerel (for spectroscopy features)
- jupyter (for notebook demos)

## Data Processing Workflows

### iQID-Only Processing (ReUpload Data)
```python
from pipelines.simplified_iqid_pipeline import SimpleiQIDPipeline

pipeline = SimpleiQIDPipeline("configs/iqid_pipeline_config.json")
results = pipeline.process_all_samples(max_samples=5)
```

### Combined H&E-iQID Processing (DataPush1 Data)  
```python
from pipelines.combined_he_iqid_pipeline import CombinedHEiQIDPipeline

pipeline = CombinedHEiQIDPipeline("configs/combined_pipeline_config.json")
results = pipeline.process_all_samples(max_samples=3)
```

## Legacy Information

### Original Demo Notebooks
The original demo notebooks are preserved in `archive/demo_notebooks/` and can be used as reference:
- 2_demo_preprocessing.ipynb
- 3_demo_seq_preprocessing.ipynb  
- 4_demo_alignment_decayCorr.ipynb
- 5_demo_dpk.ipynb
- 6_demo_biod.ipynb

**Note**: These notebooks used the old file structure and may require path adjustments to work with the new organization.

### Archived Automation Scripts
The original automation scripts have been moved to `archive/` but superseded by the new pipeline system:
- `automate_processing.py` → Use `pipelines/iqid_only_pipeline.py`
- `automate_image_alignment.py` → Use `pipelines/combined_he_iqid_pipeline.py`  
- `automate_dose_kernel_processing.py` → Integrated into pipeline workflows

## Development and Testing

### Running Tests
```bash
# Run value range tests
python tests/test_value_ranges.py

# Run batch processing tests  
python tests/test_value_range_batch.py
```

### Extending the Framework
- Add new processing modules to appropriate `src/` subdirectories
- Create new pipeline configurations in `configs/`
- Follow the modular structure for consistency
- Add tests for new functionality in `tests/`

## Data Access
Due to file size constraints, sample data is not included in the repository. Please contact the authors for access to:
- Sample ReUpload data for iQID-only processing
- Sample DataPush1 data for combined H&E-iQID processing
- Test datasets for validation

## Version History

- **v1.0** (2025): Reorganized repository with modular pipeline structure
- **2022-sci-rep**: Original repository release accompanying Scientific Reports publication

## Support

For questions, issues, or data access requests, please contact the repository authors or open an issue on the repository.

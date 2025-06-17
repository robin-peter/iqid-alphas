# Legacy Workflow Code Updates Summary

## Overview
Updated all legacy workflow Python files to use the new hierarchical `data_paths` structure that matches the consolidated workflow and real UCSF data organization.

## Files Updated

### 1. `workflow1_iqid_alignment.py`
**Purpose**: iQID raw data → aligned processing workflow

**Key Changes**:
- **Main function**: Updated to read hierarchical `data_paths` from config instead of hardcoded paths
- **Config access**: Now navigates the UCSF data structure: `config["data_paths"]["iqid"]["DataPush1"]["base_path"]`
- **Readonly compliance**: Checks and displays `storage_policy` settings
- **Default config**: Updated to match hierarchical structure with `file_patterns` sub-structure
- **Data loading**: Updated comments to use `config["data_paths"]["file_patterns"]["iqid"]`

**Before**:
```python
# Hardcoded path
data_path = "data/raw_iqid"

# Flat config structure
"data_paths": {
    "raw_iqid_dir": "data/raw_iqid",
    "file_pattern": "*.tif*"
}
```

**After**:
```python
# Dynamic hierarchical path resolution
iqid_paths = data_paths.get("iqid", {})
if iqid_paths:
    first_dataset = iqid_data_keys[0]  # e.g., "DataPush1"
    data_path = iqid_paths[first_dataset]["base_path"]

# Hierarchical config structure
"data_paths": {
    "base_path": "data/raw_iqid",
    "file_patterns": {
        "iqid": "*.tif*",
        "he": "*.jpg"
    }
}
```

### 2. `workflow2_he_iqid_coregistration.py`
**Purpose**: H&E + aligned iQID co-registration and analysis workflow

**Key Changes**:
- **Main function**: Updated to read both iQID and H&E data paths from hierarchical config
- **Dual data access**: Handles both aligned iQID output and H&E source data paths
- **Config access**: Navigates both `config["data_paths"]["iqid"]` and `config["data_paths"]["he"]`
- **Readonly compliance**: Checks and displays `storage_policy` settings
- **Default config**: Updated to match hierarchical structure
- **Data loading**: Updated H&E image loading to use `config["data_paths"]["file_patterns"]["he"]`

**Before**:
```python
# Hardcoded paths
aligned_iqid_path = "outputs/iqid_aligned/aligned_iqid_stack.npy"
he_dir = "data/he_histology"

# Flat config access
self.config["data_paths"]["he_file_pattern"]
```

**After**:
```python
# Dynamic hierarchical path resolution
he_paths = data_paths.get("he", {})
if he_paths:
    first_dataset = he_data_keys[0]  # e.g., "DataPush1"
    he_dir = he_paths[first_dataset]["base_path"]

# Hierarchical config access
self.config.get("data_paths", {}).get("file_patterns", {}).get("he", "*.jpg")
```

### 3. `run_complete_pipeline.py`
**Purpose**: Master pipeline orchestrating both workflows

**Key Changes**:
- **Main function**: Updated to read data paths from both workflow configs
- **Unified access**: Uses hierarchical structure from workflow1 config as primary, with fallback to workflow2
- **Config coordination**: Ensures both workflows use consistent data paths
- **Readonly compliance**: Displays readonly policy status from configs
- **Data path resolution**: Intelligently selects first available dataset from hierarchical structure

**Before**:
```python
# Hardcoded paths
raw_iqid_path = "data/raw_iqid"
he_images_path = "data/he_histology"
```

**After**:
```python
# Config-driven hierarchical path resolution
config1 = pipeline.workflow1.config
data_paths = config1.get("data_paths", {})

# Intelligent dataset selection
iqid_paths = data_paths.get("iqid", {})
if iqid_paths:
    first_dataset = iqid_data_keys[0]
    raw_iqid_path = iqid_paths[first_dataset]["base_path"]
```

## Key Features of Updated Code

### 1. **Hierarchical Data Path Navigation**
- Automatically detects available datasets (DataPush1, ReUpload, etc.)
- Selects first available dataset for processing
- Falls back gracefully to base paths if specific datasets not found

### 2. **Readonly Policy Compliance**
- Checks `storage_policy.enforce_readonly` from config
- Displays warnings when readonly mode is active
- Shows output directory locations

### 3. **Consistent File Pattern Usage**
- Uses `config["data_paths"]["file_patterns"]["iqid"]` for iQID files
- Uses `config["data_paths"]["file_patterns"]["he"]` for H&E files
- Maintains backward compatibility with default patterns

### 4. **Real UCSF Data Structure Support**
The code now properly handles the real UCSF data structure:
```
/readonly/UCSF-Collab/data/
├── iqid/
│   ├── DataPush1/
│   │   ├── D7M1P1_L/ (and other sample directories)
│   ├── ReUpload/ 
│   └── Visualization/
└── he/
    ├── DataPush1/
    ├── ReUpload/
    └── Visualization/
```

### 5. **Flexible Dataset Selection**
- Can process any dataset in the hierarchy (DataPush1, ReUpload, etc.)
- Automatically adapts to available data organization
- Maintains compatibility with simulated data for testing

## Usage Examples

### Running Individual Workflows
```bash
# Run iQID alignment workflow (uses hierarchical config)
python workflow1_iqid_alignment.py

# Run H&E-iQID co-registration workflow
python workflow2_he_iqid_coregistration.py

# Run complete pipeline
python run_complete_pipeline.py
```

### Expected Output Messages
```
🔬 UCSF iQID Alignment Workflow
==================================================
📂 Using UCSF data from: /readonly/UCSF-Collab/data/
📊 Processing iQID dataset: DataPush1
📍 Dataset location: /readonly/UCSF-Collab/data/iqid/DataPush1/
⚠️  READONLY mode enforced - data will not be modified
📁 All outputs will be saved to: outputs/iqid_aligned/
```

## Validation Status
- ✅ **Syntax validation**: All files compile without errors
- ✅ **Config validation**: Updated configs pass validation script  
- ✅ **Import validation**: All workflows can initialize successfully
- ✅ **Runtime testing**: Workflows run with simulated data (actual UCSF data not available in test environment)

## Migration from Old Code
If using old workflow code:

1. **Update config files**: Use the new hierarchical configs in `configs/`
2. **Update Python imports**: No changes needed - same class names
3. **Update data paths**: Remove hardcoded paths, rely on config-driven paths
4. **Check outputs**: All outputs now go to organized `outputs/` directories

## Safety Features
- **Readonly enforcement**: Will not write to UCSF data directories
- **Output isolation**: All outputs written to local writable directories
- **Graceful fallbacks**: Uses simulated data if real data not available
- **Config validation**: Validates readonly policy before processing

The updated legacy workflows now fully support the real UCSF data organization while maintaining all safety and readonly policies established in the consolidated workflow.

# IQID-Alphas Data Structure and Processing Workflow - Complete Reference

## Executive Summary

This document provides a comprehensive understanding of the UCSF iQID and H&E data structure, processing workflows, and CLI development requirements. This information is critical for proper implementation of the batch processing CLI and pipeline integration.

## 1. Data Hierarchy and Structure

### 1.1 Dataset Overview

The UCSF data is organized into two distinct datasets with different processing stages:

#### **DataPush1**: Production-Ready Dataset
- **iQID**: Already aligned and ready for 3D reconstruction
- **H&E**: Processed histology images paired with iQID
- **Status**: Complete multi-modal dataset (iQID + H&E)
- **Use Case**: Immediate analysis, method validation, production workflows

#### **ReUpload**: Complete Workflow Dataset  
- **iQID**: Contains full workflow (Raw → 1_segmented → 2_aligned)
- **H&E**: Not available in this dataset
- **Status**: Single-modal iQID dataset with all processing stages
- **Use Case**: Workflow development, pipeline testing, method development

### 1.2 Detailed Data Organization

#### **DataPush1 Structure (Production Dataset):**
```
DataPush1/
├── HE/                           # Ready-to-use H&E data
│   ├── 3D/                      # 3D reconstruction ready
│   │   ├── kidney/              # Tissue type: kidney
│   │   │   ├── D1M1_L/         # Sample: Day 1, Mouse 1, Left kidney
│   │   │   │   ├── P1L.tif     # H&E slice 1, Left (aligned)
│   │   │   │   ├── P2L.tif     # H&E slice 2, Left (aligned)
│   │   │   │   └── P3L.tif     # H&E slice 3, Left (aligned)
│   │   │   └── D1M1_R/         # Sample: Day 1, Mouse 1, Right kidney
│   │   └── tumor/              # Tissue type: tumor
│   ├── Sequential/             # Sequential section analysis
│   └── Upper/Lower/            # Specific anatomical sections
└── iQID/                        # Ready-to-use iQID data  
    ├── 3D/                     # 3D reconstruction ready (aligned)
    │   ├── kidney/
    │   │   ├── D1M1(P1)_L/     # Sample: Day 1, Mouse 1, Protocol 1, Left
    │   │   │   ├── mBq_corr_0.tif    # Aligned slice 0
    │   │   │   ├── mBq_corr_1.tif    # Aligned slice 1  
    │   │   │   └── mBq_corr_2.tif    # Aligned slice 2
    │   │   └── D1M1(P1)_R/
    │   └── tumor/
    ├── Sequential/
    └── Upper/Lower/
```

#### **ReUpload Structure (Full Workflow Dataset):**
```
ReUpload/
└── iQID_reupload/               # Complete iQID workflow dataset
    ├── Raw/                     # Original multi-slice images
    │   ├── sample_001/          # Sample directory
    │   │   └── raw_image.tif    # Single image with all slices
    │   └── sample_002/
    ├── 1_segmented/             # Individual cropped slices
    │   ├── sample_001/          
    │   │   ├── slice_0.tif      # Cropped slice 0
    │   │   ├── slice_1.tif      # Cropped slice 1
    │   │   └── slice_2.tif      # Cropped slice 2
    │   └── sample_002/
    └── 2_aligned/               # Sorted and aligned for 3D
        ├── sample_001/
        │   ├── aligned_0.tif    # Aligned slice 0
        │   ├── aligned_1.tif    # Aligned slice 1  
        │   └── aligned_2.tif    # Aligned slice 2
        └── sample_002/
```

### 1.3 Dataset Characteristics Summary

| Dataset | iQID Status | H&E Available | Processing Stages | Primary Use |
|---------|-------------|---------------|-------------------|-------------|
| **DataPush1** | ✅ Aligned (ready) | ✅ Available | Final stage only | Production analysis |
| **ReUpload** | 🔄 Full workflow | ❌ Not available | Raw → Segmented → Aligned | Pipeline development |

### 1.2 Data Types Classification

#### **Image Types:**
- **iQID**: Quantitative imaging data (`mBq_corr_*.tif`, scientific TIFF format)
- **H&E**: Histology images (`P*.tif`, pathology slides)

#### **Tissue Types:**
- **kidney**: Left (`_L`) and Right (`_R`) variants
- **tumor**: Various tumor classifications (T1, T2, etc.)

#### **Preprocessing Types:**
- **3D**: Volumetric reconstruction (multiple slices → 3D volume)
- **Sequential**: Section-by-section analysis
- **Upper/Lower**: Anatomical position-specific processing

### 1.3 Sample Naming Conventions

#### **iQID Samples:**
- Format: `D{day}M{mouse}(P{protocol})_{side}`
- Examples: `D1M1(P1)_L`, `D7M2(P2)_R`
- Note: Protocol notation `(P*)` used in iQID but not H&E

#### **H&E Samples:**
- Format: `D{day}M{mouse}_{side}` or `D{day}M{mouse}-T{tumor}_{side}`
- Examples: `D1M1_L`, `D7M2-T1_R`

#### **Sample Pairing Logic:**
- Remove `(P*)` from iQID names to match H&E: `D1M1(P1)_L` → `D1M1_L`

## 2. iQID Processing Workflow

### 2.1 Three-Stage Processing Pipeline

```
Raw iQID Image → Segmentation → Alignment → 3D Reconstruction
     ↓               ↓            ↓              ↓
Single multi-    Individual    Sorted &      Ready for
slice image      cropped       aligned       volumetric
                 slices        slices        analysis
```

#### **Stage 1: Raw Data**
- **Location**: `Raw/` directory
- **Format**: Single TIFF image containing all tissue slices
- **Characteristics**: Multi-slice data in one file, requires cropping

#### **Stage 2: Segmentation** 
- **Location**: `1_segmented/` directory
- **Process**: Crop individual slices from raw multi-slice image
- **Output**: Separate TIFF files for each tissue slice
- **Files**: `mBq_corr_0.tif`, `mBq_corr_1.tif`, etc.

#### **Stage 3: Alignment**
- **Location**: `2_aligned/` directory  
- **Process**: Sort slices by anatomical position and align for 3D reconstruction
- **Output**: Spatially aligned slice stack ready for volumetric analysis
- **Purpose**: Enables accurate 3D tissue reconstruction

### 2.2 Processing Stage Priorities

1. **Raw Processing**: Convert multi-slice image to individual slices
2. **Segmentation**: Extract clean tissue boundaries from each slice
3. **Alignment**: Prepare for 3D reconstruction through spatial alignment
4. **3D Reconstruction**: Combine aligned slices into volumetric data

### 2.2 Raw iQID Image Structure

#### **Slice Arrangement in Raw Images:**
```
Raw iQID Image Layout (Top to Bottom, Left to Right):
┌─────────┬─────────┬─────────┐
│ Slice 0 │ Slice 1 │ Slice 2 │  ← Top row
├─────────┼─────────┼─────────┤
│ Slice 3 │ Slice 4 │ Slice 5 │  ← Middle row  
├─────────┼─────────┼─────────┤
│ Slice 6 │ Slice 7 │ Slice 8 │  ← Bottom row
└─────────┴─────────┴─────────┘
```

**Key Characteristics:**
- **Spatial Order**: Slices arranged top-to-bottom, left-to-right
- **Natural Sorting**: Reading order corresponds to anatomical sequence
- **Segmentation Process**: Extract individual rectangles from grid layout
- **Anatomical Correspondence**: Grid position relates to tissue depth/position

#### **Segmentation Strategy:**
1. **Grid Detection**: Identify slice boundaries in raw image
2. **Sequential Extraction**: Extract slices in reading order (0, 1, 2, 3...)
3. **Spatial Preservation**: Maintain anatomical sequence through extraction
4. **Quality Validation**: Ensure complete slice extraction from grid

## 3. CLI Implementation Requirements

### 3.1 Data Discovery Logic

The CLI must understand and handle both dataset types:

#### **Dataset Type Detection:**
```python
def detect_dataset_type(data_path):
    if "DataPush1" in str(data_path):
        return "production"  # Aligned data ready for analysis
    elif "ReUpload" in str(data_path) or "iQID_reupload" in str(data_path):
        return "workflow"    # Full workflow data available
    else:
        return "unknown"
```

#### **Processing Stage Detection (Dataset-Specific):**
```python
def detect_processing_stage(sample_dir, dataset_type):
    if dataset_type == "production":
        # DataPush1: Only aligned data available
        return "aligned_ready"
    elif dataset_type == "workflow":
        # ReUpload: Check for workflow stages
        if (sample_dir / "Raw").exists():
            return "raw_available"
        elif (sample_dir / "1_segmented").exists():
            return "segmented_available" 
        elif (sample_dir / "2_aligned").exists():
            return "aligned_available"
    return "unknown_stage"
```

#### **Sample Analysis (Updated for Dataset Types):**
```python
sample_info = {
    'sample_dir': sample_dir,
    'sample_id': sample_dir.name,
    'dataset_type': detect_dataset_type(sample_dir),
    'tissue_type': extract_tissue_type(path),
    'preprocessing_type': extract_preprocessing_type(path),
    'processing_stage': detect_processing_stage(sample_dir, dataset_type),
    'has_he_data': dataset_type == "production",  # Only DataPush1 has H&E
    'available_stages': list_available_stages(sample_dir, dataset_type),
    'slice_count': count_slices_in_latest_stage(sample_dir),
    'can_reconstruct_3d': has_aligned_slices(sample_dir),
    'multi_modal_ready': dataset_type == "production"  # iQID + H&E available
}
```

### 3.2 Processing Strategy by Dataset and Stage

#### **DataPush1 (Production Dataset) Processing:**
```python
if dataset_type == "production":
    # Process aligned iQID data (ready for 3D reconstruction)
    if has_he_data:
        # Multi-modal analysis: iQID + H&E
        pipeline.process_multimodal_aligned(iqid_dir, he_dir, output_dir)
    else:
        # Single-modal iQID analysis
        pipeline.process_aligned_iqid(iqid_dir, output_dir)
```

#### **ReUpload (Workflow Dataset) Processing:**
```python
if dataset_type == "workflow":
    if processing_stage == "raw_available":
        # Process raw multi-slice image → segmented slices
        pipeline.process_raw_iqid(raw_image_path, output_dir)
    elif processing_stage == "segmented_available":
        # Process segmented slices → aligned stack  
        pipeline.process_segmented_slices(segmented_dir, output_dir)
    elif processing_stage == "aligned_available":
        # Process aligned slices → 3D reconstruction
        pipeline.process_aligned_stack(aligned_dir, output_dir)
```

#### **Workflow Progression (ReUpload Only):**
```python
# Auto-progression through workflow stages
def process_full_workflow(sample_dir):
    stages = ["raw", "segmented", "aligned", "3d_reconstruction"]
    for stage in stages:
        if stage_available(sample_dir, stage):
            process_stage(sample_dir, stage)
            validate_stage_output(sample_dir, stage)
        else:
            break  # Stop at first unavailable stage
```

### 3.3 Multi-Modal Integration

#### **iQID + H&E Pairing:**
```python
paired_sample = {
    'iqid_sample': {
        'sample_dir': iqid_sample_dir,
        'processing_stage': 'aligned_available',
        'slice_files': sorted_iqid_slices
    },
    'he_sample': {
        'sample_dir': he_sample_dir,
        'slice_files': sorted_he_slices
    },
    'can_coregister': True if both have aligned slices
}
```

## 4. Pipeline Integration Strategy

### 4.1 Pipeline Method Mapping

#### **SimplePipeline:**
- **Method**: `process_iqid_stack(sample_dir, output_dir)`
- **Input**: Sample directory (any stage)
- **Logic**: Auto-detect stage and process accordingly

#### **AdvancedPipeline:**
- **Method**: `process_image(slice_file, output_dir)` 
- **Input**: Representative slice from aligned stage
- **Strategy**: Use middle slice as most representative

#### **CombinedPipeline:**
- **Method**: `process_image_pair(he_slice, iqid_slice, output_dir)`
- **Input**: Corresponding slices from both modalities
- **Strategy**: Match slice positions for co-registration

### 4.2 Quality Assessment Integration

#### **Stage-Specific Metrics:**
```python
quality_metrics = {
    'raw_stage': ['image_quality', 'slice_separation'],
    'segmented_stage': ['segmentation_accuracy', 'slice_completeness'],
    'aligned_stage': ['alignment_quality', '3d_reconstruction_readiness'],
    'combined_analysis': ['coregistration_accuracy', 'multi_modal_alignment']
}
```

## 5. Development Implementation Plan

### 5.1 Phase 1: Core Data Understanding (COMPLETED)
- ✅ Document data hierarchy
- ✅ Understand processing workflow stages  
- ✅ Define sample naming conventions
- ✅ Clarify tissue slice vs time-series distinction

### 5.2 Phase 2: CLI Data Discovery Enhancement
#### **Required Updates:**
1. **Stage Detection**: Identify Raw/1_segmented/2_aligned directories
2. **Processing Readiness**: Determine what processing can be performed
3. **Multi-Stage Reporting**: Show available processing stages per sample
4. **3D Capability Flags**: Identify samples ready for volumetric analysis

#### **Implementation Priority:**
```python
# High Priority
- detect_processing_stage()
- analyze_sample_workflow_status()
- report_3d_reconstruction_readiness()

# Medium Priority  
- validate_slice_completeness()
- assess_alignment_quality()
- check_multi_modal_compatibility()
```

### 5.3 Phase 3: Pipeline Processing Logic
#### **Required Updates:**
1. **Stage-Aware Processing**: Route to appropriate processing based on available data
2. **Workflow Progression**: Support raw → segmented → aligned pipeline
3. **Quality Gates**: Validate each stage before proceeding to next
4. **3D Integration**: Enable volumetric reconstruction from aligned slices

### 5.4 Phase 4: Multi-Modal Co-Registration
#### **Required Features:**
1. **Slice Correspondence**: Match iQID and H&E slices by position
2. **Spatial Alignment**: Register between imaging modalities
3. **Quality Assessment**: Validate co-registration accuracy
4. **Integrated Visualization**: Combined multi-modal outputs

## 6. Technical Specifications

### 6.1 File Format Support
- **iQID**: Scientific TIFF (float64, quantitative data)
- **H&E**: Standard TIFF (RGB histology images)
- **Output**: PNG/TIFF for visualization, NumPy arrays for analysis

### 6.2 Memory Management
- **Raw iQID**: Large multi-slice images (potential memory issues)
- **Segmented**: Multiple individual slices (moderate memory usage)
- **Aligned**: Optimized for 3D processing (efficient memory usage)

### 6.3 Processing Performance
- **Stage Detection**: Fast directory scanning
- **Slice Processing**: Parallel processing of individual slices
- **3D Reconstruction**: Memory-efficient volumetric operations

## 7. CLI Command Structure (Updated)

### 7.1 Discovery Command Enhancement
```bash
# DataPush1 (Production dataset)
python -m iqid_alphas.cli discover --data data/DataPush1

# Expected Output:
📁 DATA DISCOVERY RESULTS (DataPush1 - Production Dataset)
========================================
iQID samples (aligned, 3D ready): 15
H&E samples (aligned, ready): 12  
Paired samples (multi-modal ready): 12

🔬 Processing Status:
   - Dataset type: Production (aligned data)
   - All samples ready for 3D reconstruction
   - Multi-modal analysis available (iQID + H&E)

🧬 Tissue Distribution:
   - kidney: 10 samples (5 left, 5 right)
   - tumor: 5 samples (various types)

# ReUpload (Workflow dataset)
python -m iqid_alphas.cli discover --data data/ReUpload

# Expected Output:
📁 DATA DISCOVERY RESULTS (ReUpload - Workflow Dataset)
========================================
iQID samples (workflow stages): 25
H&E samples: 0 (not available in this dataset)
Paired samples: 0 (single-modal dataset)

🔬 Processing Stage Analysis:
   - Raw stage available: 25 samples
   - Segmented stage available: 20 samples  
   - Aligned stage available: 15 samples
   - Ready for 3D reconstruction: 15 samples

🧬 Workflow Opportunities:
   - 5 samples need segmentation (raw → segmented)
   - 5 samples need alignment (segmented → aligned)
   - 15 samples ready for 3D reconstruction
```

### 7.2 Processing Command Enhancement
```bash
# DataPush1: Multi-modal analysis (iQID + H&E)
python -m iqid_alphas.cli process \
    --data data/DataPush1 \
    --config configs/multimodal_config.json \
    --pipeline combined \
    --max-samples 5

# DataPush1: Single-modal iQID analysis
python -m iqid_alphas.cli process \
    --data data/DataPush1/iQID \
    --config configs/iqid_analysis_config.json \
    --pipeline advanced \
    --max-samples 10

# ReUpload: Workflow processing (auto-detect stage)
python -m iqid_alphas.cli process \
    --data data/ReUpload \
    --config configs/workflow_config.json \
    --pipeline simple \
    --stage auto \
    --max-samples 5

# ReUpload: Specific stage processing
python -m iqid_alphas.cli process \
    --data data/ReUpload \
    --config configs/alignment_config.json \
    --stage segmented \
    --output results/alignment_test
```

## 8. Validation and Testing Strategy

### 8.1 Dataset-Specific Data Integrity Checks

#### **DataPush1 (Production Dataset):**
- ✅ Verify aligned iQID data completeness
- ✅ Validate H&E data availability and pairing
- ✅ Check multi-modal correspondence (iQID ↔ H&E)
- ✅ Confirm 3D reconstruction readiness

#### **ReUpload (Workflow Dataset):**
- ✅ Verify Raw → Segmented → Aligned progression
- ✅ Validate workflow stage completeness
- ✅ Check slice count consistency across stages
- ✅ Confirm iQID-only structure (no H&E expected)

### 8.2 Processing Workflow Tests

#### **DataPush1 Testing:**
- 🔄 Multi-modal co-registration validation (iQID + H&E)
- 🔄 3D reconstruction from aligned data
- 🔄 Production workflow performance benchmarking
- 🔄 Quality assessment for publication-ready results

#### **ReUpload Testing:**
- 🔄 Raw → Segmented workflow validation
- 🔄 Segmented → Aligned pipeline testing  
- 🔄 Full workflow progression (Raw → 3D)
- 🔄 Pipeline development and method validation

### 8.3 CLI Functionality Tests

#### **Cross-Dataset Testing:**
- 🔄 Dataset type auto-detection (DataPush1 vs ReUpload)
- 🔄 Workflow-appropriate processing selection
- 🔄 Error handling for dataset-inappropriate operations
- 🔄 Performance comparison across dataset types

#### **Use Case Validation:**
- 🔄 Production analysis workflows (DataPush1)
- 🔄 Development and testing workflows (ReUpload)
- 🔄 Multi-modal vs single-modal processing paths
- 🔄 Quality gates and validation checkpoints

## 9. Future Enhancements

### 9.1 Advanced 3D Features
- **Volumetric Visualization**: Interactive 3D tissue rendering
- **Cross-Sectional Analysis**: Arbitrary plane visualization
- **Quantitative 3D Metrics**: Volume-based measurements

### 9.2 Workflow Automation
- **Pipeline Orchestration**: Automated raw → aligned processing
- **Quality-Driven Progression**: Automatic stage advancement based on quality metrics
- **Batch 3D Reconstruction**: Efficient volumetric processing of multiple samples

### 9.3 Multi-Modal Integration
- **Advanced Co-Registration**: Sophisticated alignment algorithms
- **Integrated Analysis**: Combined iQID + H&E quantitative analysis
- **Cross-Modal Validation**: Quality assessment across imaging modalities

## Conclusion

This comprehensive understanding of the UCSF iQID data structure and processing workflow provides the foundation for proper CLI implementation. The key insights are:

### **Critical Dataset Distinctions:**
1. **DataPush1**: Production-ready dataset with aligned iQID + H&E for immediate analysis
2. **ReUpload**: Development dataset with full iQID workflow (Raw → Segmented → Aligned)
3. **Multi-Modal vs Single-Modal**: DataPush1 enables iQID+H&E analysis, ReUpload is iQID-only

### **Processing Strategy:**
1. **DataPush1 Focus**: 3D reconstruction and multi-modal analysis
2. **ReUpload Focus**: Workflow development and pipeline testing
3. **Stage-Aware Processing**: CLI must detect and handle both dataset types appropriately

### **Implementation Priorities:**
1. **Dataset Type Detection**: Auto-identify DataPush1 vs ReUpload
2. **Workflow-Appropriate Processing**: Different strategies per dataset
3. **Multi-Modal Integration**: Leverage DataPush1's iQID+H&E capability
4. **Quality-Driven Pipeline**: Each stage enables higher-level analysis

### **CLI Development Focus:**
1. **Discovery Enhancement**: Dataset-aware reporting and capability detection
2. **Processing Logic**: Route samples to appropriate workflows based on dataset type
3. **Quality Assessment**: Validate readiness for 3D reconstruction and multi-modal analysis
4. **User Experience**: Clear indication of dataset capabilities and processing options

This document serves as the definitive reference for all future development decisions and ensures proper implementation of the IQID-Alphas CLI system with full understanding of both production-ready and development datasets.

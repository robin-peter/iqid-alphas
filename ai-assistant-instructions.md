# AI Assistant Instructions for IQID-Alphas

## 1. Core Principles

1.  **Configuration-Driven Architecture**: All processing parameters, visualization settings, and pipeline behavior are controlled through JSON configuration files in the `configs/` directory.
2.  **Modular Design**: Components are organized into distinct modules (`core`, `pipelines`, `visualization`, `utils`) with clear separation of concerns.
3.  **Quality-First Processing**: Every processing stage includes quality assessment, validation, and comprehensive error handling with detailed logging.
4.  **Pipeline-Centric Workflow**: Use the appropriate pipeline class (`SimplePipeline`, `AdvancedPipeline`, `CombinedPipeline`) rather than calling core modules directly.
5.  **Reproducible Science**: All processing includes metadata capture, versioning, and deterministic results for scientific reproducibility.

## 2. Development Workflow

1.  **Understand Existing Code**: Use file search and AI assistance to check for existing functionality before writing new code, especially in the extensive `evaluation/` and `examples/` directories.
2.  **Follow Architectural Patterns**: Adhere to the established layered architecture:
   - `core/` modules for fundamental processing operations
   - `pipelines/` for workflow orchestration
   - `visualization/` for all plotting and reporting
   - `utils/` for shared utilities
3.  **Implement via Pipeline Layer**: Add new functionality through pipeline classes first, then expose specific components if needed.
4.  **Configuration Integration**: Ensure all new features are configurable through JSON files and follow the established configuration patterns.
5.  **Comprehensive Testing**: Use the evaluation framework in `evaluation/scripts/` to validate new features and ensure they integrate properly with existing workflows.

## 3. Common Pitfalls to Avoid

### ❌ Don't Do This

1.  **Hard-coding Processing Parameters**: Never embed parameters directly in code; use the configuration system in `configs/`.
2.  **Bypassing Pipeline Architecture**: Don't call core modules directly; use the appropriate pipeline class for workflow orchestration.
3.  **Inconsistent Visualization**: Don't create plots without following the visualization guidelines in `docs/technical/visualization_guidelines.md`.
4.  **Ignoring Quality Metrics**: Don't skip quality assessment and validation steps that are integral to the scientific workflow.
5.  **Breaking Configuration Compatibility**: Don't modify existing configuration schemas without providing migration paths.

### ✅ Do This Instead

1.  **Use Configuration Files**: Store all parameters in JSON configuration files and load them through the established configuration management system.
2.  **Leverage Pipeline Classes**: Use `SimplePipeline`, `AdvancedPipeline`, or `CombinedPipeline` for processing workflows, extending them as needed.
3.  **Follow Visualization Standards**: Use the `VisualizationManager` and follow the standardized plotting approach with consistent styling and error handling.
4.  **Implement Quality Controls**: Include quality assessment, validation metrics, and comprehensive error handling in all new processing functions.
5.  **Extend Existing Patterns**: Build upon existing evaluation and validation frameworks rather than creating new ones.

## 4. Essential Files to Reference

- **Architecture Overview**: `docs/design_document.md`
- **API Documentation**: `docs/api_reference/core_modules.md`, `docs/api_reference/pipeline_classes.md`, `docs/api_reference/visualization_system.md`
- **Configuration Guide**: `docs/user_guides/configuration.md`
- **Visualization Guidelines**: `docs/technical/visualization_guidelines.md`
- **Example Workflows**: `examples/basic_usage.py`, `examples/advanced_workflow.py`
- **UCSF Integration**: `examples/ucsf_consolidated/` and `docs/examples/ucsf_workflows.md`
- **Batch Processing**: `examples/ucsf_consolidated/ucsf_batch_processor.py` for processing all samples
- **Visualization Guidelines**: `docs/technical/visualization_guidelines.md`

## 5. Key Package Components

### Core Processing (`iqid_alphas.core`)
- `IQIDProcessor`: Main image processing operations
- `ImageSegmenter`: Tissue and activity segmentation
- `ImageAligner`: Image registration and alignment
- `Visualizer`: Basic visualization components

### Pipeline System (`iqid_alphas.pipelines`)
- `SimplePipeline`: Basic iQID processing workflow
- `AdvancedPipeline`: Comprehensive analysis with quality metrics
- `CombinedPipeline`: H&E + iQID integrated processing

### Visualization System (`iqid_alphas.visualization`)
- `VisualizationManager`: Centralized visualization control
- `PipelineVisualizer`: Pipeline-integrated plotting
- Configuration-driven styling and output management

### Evaluation Framework (`evaluation/`)
- `ComprehensivePipelineEvaluator`: System-wide validation
- `AdvancedEvaluator`: Detailed quality assessment
- Production validation and performance benchmarking

### Batch Processing System (`examples/ucsf_consolidated/`)
- `UCSFBatchProcessor`: Automated processing of all available samples
- Comprehensive visualization suite with individual and summary plots
- Quality assessment and statistical analysis across entire dataset
- Organized output structure with detailed reporting

## 6. Scientific Context

This is a medical imaging analysis system for quantitative iQID (Imaging Quantitative ID) camera data and H&E histology images. The system is used for:
- Quantitative imaging analysis in medical research
- Tissue segmentation and activity quantification
- Image registration between different modalities
- Dose distribution analysis and modeling
- Statistical analysis and visualization for publication

All code must maintain scientific rigor, reproducibility, and quality suitable for peer-reviewed publication.

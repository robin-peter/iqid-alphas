---
mode: 'agent'
description: 'Initiate a multi-step task or goal-oriented action in the IQID-Alphas project.'
---
## Project Agent Task

**My Goal**: {{YOUR_OVERALL_GOAL}} (e.g., "Create a new CLI command for batch processing," "Implement a new feature for 3D volume visualization," "Set up automated quality control pipeline.")

**Initial Steps/Requirements (if known)**: {{INITIAL_STEPS_OR_REQUIREMENTS}} (e.g., "Start by defining the interface in `iqid_alphas/core/`," "The data should be sourced from the existing pipeline results," "Consider integration with the existing evaluation framework.")

**Desired Output**: {{EXPECTED_OUTPUT}} (e.g., "Working CLI command with help documentation," "New visualization module with configuration support," "Automated QC system with reporting capabilities.")

**Please guide me through this task, suggesting steps and generating code to achieve this goal while adhering to our project guidelines.**

---

## Agent Task Categories

### New Feature Development
- "Create a new preprocessing module for advanced noise reduction with multiple algorithm options"
- "Implement batch processing capabilities with progress monitoring and parallel execution"
- "Add support for new image formats and metadata extraction"

### System Integration
- "Integrate a new quality control system that runs automatically after each processing stage"
- "Create a web dashboard for monitoring pipeline execution and results"
- "Implement automated report generation with customizable templates"

### Architecture Enhancement
- "Refactor the visualization system to support plugin-based architecture"
- "Create a new pipeline type for time-series analysis of longitudinal data"
- "Implement caching system for intermediate processing results"

### Scientific Workflow
- "Create a new analysis module for statistical comparison between processing methods"
- "Implement cross-validation framework for segmentation algorithms"
- "Add support for multi-modal image fusion and analysis"

## Project Context

Remember that IQID-Alphas is a scientific image analysis system with:
- **Medical Imaging Focus**: Processing iQID camera data and H&E histology
- **Publication Quality**: Results must be suitable for peer-reviewed research
- **Configuration-Driven**: All behavior controlled through JSON configurations
- **Quality-First**: Comprehensive validation and quality assessment
- **Modular Architecture**: Clean separation between core, pipelines, and visualization
- **Evaluation Framework**: Extensive testing and validation capabilities

The system is actively used in medical research, so all changes must maintain scientific rigor and reproducibility.

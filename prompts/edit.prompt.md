---
mode: 'edit'
description: 'Request a specific, targeted code modification in the IQID-Alphas project.'
---
## Code Edit Request

**File to Modify**: {{PATH_TO_FILE}}

**Goal**: {{YOUR_GOAL}} (e.g., "Add error handling," "Refactor this function," "Fix a bug where...")

**Specific Instructions**: {{CLEAR_INSTRUCTIONS_ON_WHAT_TO_CHANGE}} (e.g., "In the `process_image` function, add quality validation before segmentation step and log a warning if quality metrics fall below threshold.")

**Please provide the modified code, adhering to all project guidelines.**

---

## Guidelines for Edits

### Configuration Changes
- Always use JSON configuration files rather than hard-coded values
- Follow existing configuration patterns in `configs/` directory
- Maintain backward compatibility when modifying configuration schemas

### Pipeline Modifications
- Respect the pipeline architecture and don't bypass established workflows
- Include quality assessment and validation in new processing steps
- Follow the logging patterns established in existing pipeline classes

### Visualization Updates
- Use the VisualizationManager and follow guidelines in `docs/technical/visualization_guidelines.md`
- Ensure consistent styling through configuration-driven approach
- Include proper error handling for visualization failures

### Core Module Changes
- Maintain scientific accuracy and reproducibility
- Include comprehensive error handling and validation
- Follow existing patterns for metadata capture and result organization

## Examples

### Pipeline Enhancement
- "Add a new preprocessing step to the AdvancedPipeline that applies noise reduction with configurable parameters"
- "Modify the quality assessment function to include additional statistical metrics"

### Visualization Improvement
- "Update the comprehensive visualization to include alignment quality metrics in the summary plot"
- "Add support for custom colormaps in the tissue segmentation visualization"

### Configuration Extension
- "Add new configuration options for batch processing parameters in the pipeline config"
- "Extend the visualization configuration to support different export formats"

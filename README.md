# iqid-alphas
A Python-based framework for listmode data processing, image processing, and 
dosimetry using the iQID camera digital autoradiograph.

Repository author: Robin Peter

iQID camera expert: Brian Miller

Please contact the authors with any questions or for access to data samples.

## Papers
- (2022, Sci Rep, initial methods): https://doi.org/10.1038/s41598-022-22664-5.
- (2024, Sci Rep, 3D sub-organ dosimetry and TCP): https://doi.org/10.1038/s41598-024-70417-3

Permanent DOI of the initial repository release in 2022: [![DOI](https://zenodo.org/badge/540307496.svg)](https://zenodo.org/badge/latestdoi/540307496)


## Branches
- **main**: the most up-to-date (but not backwards compatible) stable branch of the repo.
- **2022-sci-rep**: the initial stable repository released alongside the 2022 publication in Scientific Reports.

Any updates will be made to the main branch.

## Python Dependencies
Please run script "check_dependencies.py" to check these against your versions.

Required:
- numpy (1.20.2)
- opencv / cv2 (4.0.1)
- skimage (0.18.1)
- scipy (1.6.2)
- PyStackReg (0.2.5) (https://github.com/glichtner/pystackreg)

Recommended for visualization and Jupyter notebook demos:
- Jupyter (4.7.1) notebook (6.4.0) or lab (3.0.14)
- matplotlib (3.3.4)

## Installation
To install the iQID processing functions:
    1. clone git repo
    2. python check_dependencies.py to check for the required Python packages.
    3. (optional) run the demo notebooks to get a sense of the workflow.
    4. import functions from iqid.process, iqid.dpk, or iqid.align as needed in your scripts.

**Important note** Please mind the file structure if you are running the demo notebooks. See below.

## Repository Structure
Below is a description of the files in the 2022-sci-rep branch of the iqid-alphas repo.

iqid-alphas/ (main project directory)
  - check_dependencies.py   - setup script for checking Python dependencies. 
                              See Installation section above.
  - iqid/ (Python source code)
    - process_object.py (processing iQID listmode data)
    - align.py (alignment and registration)
    - dpk.py (dose kernel and DPK convolution)
    - helper.py (miscellaneous helper functions for plotting, calculations, etc)
    - spec.py (gamma spectroscopy)
  - demo_notebooks/ (demo Jupyter notebook tutorials)
    - Please use these as templates for your own analysis if you like.
    - Please explore the rest of the source code, as not everything is represented by the tutorials.
    - **Important note** each notebook starts with "cd .." to move the working directory up a level to access the source files.
      This MUST be run in order to import the iQID source code from iqid/ unless you move the .py or .ipynb files.
      I opted for this solution to avoid adding the iQID package to your system or Python path.
      However, for more permanent setup, please see discussion on the following thread:
      https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im
  - data_sample/ (sample data used in the demo notebooks)
    - Due to file size, these are not uploaded to the repository.
    - Additionally, select images are visible in the demo notebook previews. Please feel free to
      make one copy of the notebooks for viewing and another to experiment using your own data.
    - Please contact the authors if you would like to request access to the data.

## New Automation Scripts
The repository now includes three new automation scripts to streamline the processing of listmode data, image alignment, and dose kernel processing:

1. `automate_processing.py`:
   - Automates loading and processing of listmode data using functions in `iqid/process_object.py`.
   - Uses `ClusterData` class to load and process data, implements functions to filter, correct, and analyze data.
   - Automates generation of spatial images and temporal information using `image_from_listmode` and `image_from_big_listmode` methods.
   - Implements automated contour detection and ROI extraction using `get_contours` and `get_ROIs` methods.
   - Saves processed data and results in a structured format for further analysis.

2. `automate_image_alignment.py`:
   - Automates alignment and registration of images using functions in `iqid/align.py`.
   - Uses `assemble_stack` and `coarse_stack` functions to create and align image stacks.
   - Implements automated padding and cropping of images using `pad_stack_he` and `crop_down` functions.
   - Automates generation of registered image stacks and saves results in a structured format for further analysis.

3. `automate_dose_kernel_processing.py`:
   - Automates processing of dose kernels and convolution using functions in `iqid/dpk.py`.
   - Uses `load_txt_kernel` and `mev_to_mgy` functions to load and convert dose kernels.
   - Implements automated radial averaging and padding of kernels using `radial_avg_kernel` and `pad_kernel_to_vsize` functions.
   - Automates convolution of dose kernels with activity image stacks and saves results in a structured format for further analysis.

## Centralized Configuration Management
A centralized configuration management system using JSON has been added to manage configuration parameters across the three automation scripts. The `config.json` file stores input/output paths, processing thresholds, and kernel sizes. The automation scripts have been updated to read configuration parameters from `config.json`.

## Robust Error Handling and Logging
Robust error handling and logging have been implemented in each automation script using Python's `logging` module. This provides clear, actionable insights into script execution, warnings, and errors. Error handling has been added for missing input files, corrupted data, and failed computational steps. Example log message for a failed processing step: `logger.error("Failed to process file %s: %s", filename, str(e))`.

## Performance Optimization
Performance bottlenecks in `iqid/process_object.py`, `iqid/align.py`, and `iqid/dpk.py` have been identified. Optimization techniques such as parallel processing, lazy loading, and optimized data structures have been suggested to improve performance.

## Testing and Validation
A testing and validation strategy for the automated workflows has been outlined. Unit tests for individual functions and integration tests for full end-to-end workflows have been created. The output of automated scripts is programmatically validated to ensure correctness and adherence to expected results.

## CI/CD Integration
Steps for integrating automation scripts into a CI/CD pipeline using GitHub Actions have been proposed. Quality gates based on the output of automated scripts have been defined to prevent merging pull requests if certain criteria are not met.

## Unit Tests
The repository includes unit tests for the new automation scripts:

1. `test_automate_processing.py`:
   - Tests loading and processing of listmode data.
   - Tests generation of spatial images and temporal information.
   - Tests contour detection and ROI extraction.
   - Tests saving of processed data and results.

2. `test_automate_image_alignment.py`:
   - Tests image alignment and registration.
   - Tests padding and cropping of images.
   - Tests generation of registered image stacks.
   - Tests saving of registered image stacks.

3. `test_automate_dose_kernel_processing.py`:
   - Tests processing of dose kernels.
   - Tests radial averaging and padding of kernels.
   - Tests convolution of dose kernels with activity image stacks.
   - Tests saving of processed dose kernels and results.

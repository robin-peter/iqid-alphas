# iqid-alphas
A Python-based framework for listmode data processing, image processing, and 
dosimetry using the iQID camera digital autoradiograph, as described in the
paper published here: **(doi to be included)**.

The permanent DOI of this repository is:

[![DOI](https://zenodo.org/badge/540307496.svg)](https://zenodo.org/badge/latestdoi/540307496)

Author: Robin Peter

iQID camera expert: Brian Miller

Please contact the authors with any questions or for access to data samples.

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
Below is a description of the current files in the main (stable) branch of the iqid-alphas repo.

iqid-alphas/ (main project directory)
  - check_dependencies.py   - setup script for checking Python dependencies. 
                              See Installation section above.
  - iqid/ (Python source code)
    - process.py (processing iQID listmode data)
    - align.py (alignment and registration)
    - dpk.py (dose kernel and DPK convolution)
  - demo_notebooks/ (demo Jupyter notebook tutorials)
    - Please use these as templates for your own analysis if you like.
    - Please explore the rest of the source code, as not everything is represented by the tutorials.
    - No demo is currently available for the iqid.process (preprocessing) module, as the data that was published 
      in the Scientific Reports manuscript used a legacy iQID data format. Authors are working to update this for modern iQID devices.
    - **Important note** each notebook starts with "cd .." to move the working directory up a level to access the source files.
      This MUST be run in order to import the iQID source code from iqid/ unless you move the .py or .ipynb files.
      I opted for this solution to avoid adding the iQID package to your system or Python path.
      However, for more permanent setup, please see discussion on the following thread:
      https://stackoverflow.com/questions/34478398/import-local-function-from-a-module-housed-in-another-directory-with-relative-im
  - parameters/ (shareable data files)
    - at211_10E6_151_1um_water.txt  (DPK for At-211 simulated in GEANT4 using $10^7$ alpha particles in water.)
    - scale_data.csv (Spreadsheet containing iQID and H&E scale information for all studies.)
    - activity_correction.csv (Spreadsheet containing small correction values introduced by transformations.)
  - data/ (NOT PUBLIC: canine lymph node iQID data from Fred Hutchinson)
    - For data security reasons, these are not currently uploaded to the repository.
    - However, until they are, you should be able to see the file structure of the folders.
    - Additionally, select images are visible in the demo notebook previews. Please feel free to
      make one copy of the notebooks for viewing and another to experiment using your own data.
    - Finally, please contact the authors if you would like to request access to the data.

import os
import re
import glob
from pathlib import Path

import numpy as np

from tqdm import trange

from pandas import isna



def list_studies(rootdir):
    """Gets list of directory paths for all folders in the current dir."""
    study_list = [f.path for f in os.scandir(rootdir) if f.is_dir()]
    return (study_list)


def list_substudies(rootdir):
    """Gets list of directory paths for all subfolders one level down."""
    study_list = list_studies(rootdir)
    substudy_list = []
    for study in study_list:
        substudies = list_studies(study)
        substudy_list = substudy_list + substudies
    return (substudy_list)


def organize_dirs(rootdir, study_list, copy_files=True, copy_meta=True):
    """Organize a set of subdirectories in the root directory based on list of identified directories.
    Then make a copy of the file in the appropriate subdirectory."""

    import shutil

    Path(os.path.join(rootdir, '..', 'analysis')).mkdir(
        parents=True, exist_ok=True)
    for i in trange(len(study_list)):
        filename = glob.glob(os.path.join(
            study_list[i], 'Listmode', '*.dat'))[0]
        name = os.path.basename(study_list[i])
        seqname = os.path.basename(os.path.dirname(study_list[i]))
        seqname = str.split(seqname)[0]
        dayname = os.path.basename(
            os.path.dirname(os.path.dirname(study_list[i])))
        newname = string_underscores([dayname, seqname, name])
        newdir = os.path.join(rootdir, '..', 'analysis', newname)
        Path(newdir).mkdir(parents=True, exist_ok=True)
        if copy_files:
            shutil.copy2(filename, newdir)
        if copy_meta:
            metaname = glob.glob(os.path.join(
                study_list[i], 'Acquisition_Info' + '*.txt'))[0]
            shutil.copy2(metaname, newdir)

# two helper functions for natural sorting of file names
# from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

def atoi(text):
    """Helper function for natural sort order."""
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Helper function for natural sort order."""
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def natural_sort(l):
    l.sort(key=natural_keys)
    return (l)


def mean_stdev(array, return_vals=False):
    """Simple function to print the mean and stdev of an array to 2 sigfigs."""
    print('{:.2f} +/- {:.2f}'.format(np.nanmean(array), np.nanstd(array)))
    if return_vals:
        return np.nanmean(array), np.nanstd(array)
    else:
        return ''


def pct_err(ref, exp):
    """Simple % err function"""
    return 100 * (ref-exp)/ref


def get_yn(string):
    try:
        return {"yes": True, "no": False, "y": True, "n": False}[string.lower()]
    except KeyError:
        print("Invalid input, please enter yes or no!")


def get_dt(t1, t2, grain='us', verbose=False):
    """Find the time difference between two times.

    Parameters
    ----------
    t1, t2 : datetime.datetime objects
        The two times to be assessed

    grain : 'us' or 's'
        Temporal resolution of the result
        Note that dt.seconds != dt.total_seconds()    

    Returns
    -------
    dt : Time between events

    """

    # handle nan values
    if isna(t1) or isna(t2):
        return np.nan

    # find dt between two datetime.datetime objects
    dt = min(np.abs(t1 - t2), np.abs(-(t1 - t2)))

    if grain == 'us':
        dt = dt.seconds + dt.microseconds * 1e-6
    elif grain == 's':
        dt = dt.total_seconds()
    else:
        print('choose s or us for grain')
    return dt


def check_mono(arr):
    """Check if an array is monotonically increasing.

    Parameters
    ----------
    arr : arrayF

    Returns
    -------
    bool : True (mono increasing) or False

    """
    return np.all(arr[:-1] <= arr[1:])

# ---------------------------- ARRAYS AND LINALG ---------------------------- #


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    From J.F. Sebastian, https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array

    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
        [102 110 118 126 134]
        [182 190 198 206 214]
        [262 270 278 286 294]
        [342 350 358 366 374]]

    """
    operation = operation.lower()
    if operation not in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError(
            "Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [x for p in compression_pairs for x in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def decompose_affine(A):
    """Decompose a 3x3 transformation matrix A into its affine constituents.

    Affine: translation (T), rotation (R), scaling (Z), and shear (S). From
    https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/affines.py
    """

    A = np.asarray(A)
    T = A[:-1, -1]
    RZS = A[:-1, :-1]
    ZS = np.linalg.cholesky(np.dot(RZS.T, RZS)).T
    Z = np.diag(ZS).copy()
    shears = ZS / Z[:, np.newaxis]
    n = len(Z)
    S = shears[np.triu(np.ones((n, n)), 1).astype(bool)]
    R = np.dot(RZS, np.linalg.inv(ZS))
    if np.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = np.dot(RZS, np.linalg.inv(ZS))
    return T, R, Z, S


def com(im):
    # calculate the center of mass of an image
    m = im / np.sum(im)

    # marginal distributions
    dx = np.sum(m, 0)
    dy = np.sum(m, 1)

    (X, Y) = im.shape
    # # expected values
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))

    return cx, cy


def combine_files(glob_path, dim=1):
    # import and combine a set of procedurally named txt files
    # such as AcRa_218s_0.txt, AcRa_218s_1.txt, ...
    filelist = natural_sort(glob_path)
    alist = []
    for i in range(len(filelist)):
        x = np.loadtxt(filelist[i])
        alist.append(x)
        
    if dim==1:
        return np.ravel(alist)
    elif dim > 1:
        return np.vstack(alist)
    else:
        print('select dimension: 1 or greater')

# ---------------------------- DEPRECATED ---------------------------- #


# def plot_all(drstack, nrows=4):
#     f, ax = plt.subplots(nrows=nrows, ncols=int(
#         np.ceil(len(drstack)/nrows)), figsize=(12, 9))
#     ax = ax.ravel()

#     vma = np.max(drstack)

#     for i in range(len(ax)):
#         try:
#             ax[i].imshow(drstack[i], cmap='inferno', vmax=vma)
#             ax[i].axis('off')
#             ax[i].text(1, 0, i, fontsize=18, color='white', transform=ax[i].transAxes,
#                        horizontalalignment='right', verticalalignment='bottom')
#         except IndexError:
#             ax[i].axis('off')

#     # plt.tight_layout()
#     plt.subplots_adjust(wspace=-0.3, hspace=0.05)
#     plt.show()
#     plt.close()


# def subplot_label(text, index):
#     ax[index].text(0.5, 0.5, text, fontsize=16,
#                    color='white', horizontalalignment='center', verticalalignment='center', transform=ax[index].transAxes)
#     return (1)


# def string_underscores(string_list):
#     newstring = ''
#     for i in range(len(string_list)):
#         newstring = newstring + string_list[i] + '_'
#     newstring = newstring[:-1]  # remove last _
#     return (newstring)

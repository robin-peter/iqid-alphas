import os
import re
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange



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
    print('{:.2f} +/- {:.2f}'.format(np.mean(array), np.std(array)))
    if return_vals:
        return np.mean(array), np.std(array)
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


# ---------------------------- PLOTTING FUNCTIONS ---------------------------- #


def set_plot_parms():
    """The standard set of plt parameters that I like.
    It is also useful to be able to reference the cmap."""

    pltmap = plt.get_cmap("tab10")

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 1

    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.alpha'] = 0.5

    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'k'

    return pltmap


def draw_lines(es, label=None, ls='--', **kwargs):
    """Draws dashed lines on the current plot axis at the
    specified x-values (energies).

    Parameters
    ----------
    es : list
        The list of energies at which to draw lines

    kwargs : standard format parameters for plt.plot functions

    """

    if label is not None:
        plt.plot([], [], label=label, ls=ls, **kwargs)

    for i in range(len(es)):
        plt.axvline(es[i], label=None, ls=ls, **kwargs)


def add_scalebar(scalebarlen, ax, px_mm=1/(80/2048), loc='lower right', **kwargs):
    """Draw a formatted and labelled scalebar.

    Parameters
    ----------
    scalebarlen : float
        Physical length of scalebar in mm

    ax : plt.axes object
        Axis on which to draw the scalebar

    px_mm : float
        Pixels per millimeter for the image space

    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    # default 39.0 um effective pixel size for iQID
    scalebar = AnchoredSizeBar(ax.transData,
                               scalebarlen *
                               px_mm, '{} mm'.format(scalebarlen),
                               loc=loc,
                               **kwargs)
    ax.add_artist(scalebar)


def nice_colorbar(mappable, cbar_location="right", orientation="vertical", pad=0.05):
    """Produce a decently formatted colorbar.
    Only really works on single image, don't try with subplots."""
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbar_location, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    plt.sca(last_axes)
    return cbar


def compare_metrics(ref, mov):
    """Wrapper function to compare two images in several ways:
    
    - Mean and max % difference
    - Visual comparison with plot_compare()
    - Pixel intensity histogram comparison with plot_hists()
    """
    mean_diff = (np.mean(mov) - np.mean(ref))/np.mean(ref)
    max_diff = (np.max(mov) - np.max(ref))/np.max(ref)
    print('Mean % err (compared to image 1): {:.2f} %'.format(mean_diff*100))
    print('Max % err (compared to image 1): {:.2f} %'.format(max_diff*100))
    plot_compare(ref, mov)
    plot_hists(ref, mov)


def plot_hists(slice1, slice2):
    """Compare the pixel intensity histograms between two images."""

    f = plt.figure(figsize=(8, 4))
    n1, bins1 = np.histogram(slice1.ravel(), density=True, bins=100)
    n2, bins2 = np.histogram(slice2.ravel(), density=True, bins=100)
    plt.step(bins1[1:-1], n1[1:], alpha=1, label='reference image')
    plt.step(bins2[1:-1], n2[1:], alpha=1, label='comparison image')
    plt.legend()
    plt.xlabel('Dose rate (mGy/h)')
    plt.ylabel('Frequency')
    plt.show()


def plot_compare(slice1, slice2):
    """Nice plotting wrapper to visually compare two images."""

    from mpl_toolkits.axes_grid1 import ImageGrid
    from matplotlib.colorbar import Colorbar

    fig = plt.figure(figsize=(8, 4))
    ax = ImageGrid(fig, 111,
                   nrows_ncols=(1, 2),
                   axes_pad=0.02,
                   share_all=True,
                   cbar_location="right",
                   cbar_mode="single",
                   cbar_size="7%",
                   cbar_pad=0.05,
                   )

    im1 = ax[0].imshow(slice1, cmap='inferno')  # , vmin=immin, vmax=immax)
    vmi, vma = im1.get_clim()
    im2 = ax[1].imshow(slice2, cmap='inferno', vmin=vmi, vmax=vma)

    # Colorbar
    ax[1].cax.cla()
    cbar = Colorbar(ax[1].cax, im2)
    cbar.set_label('Absorbed Dose Rate\n(mGy/h)')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('Reference Image')
    ax[1].set_title('Comparison Image')
    plt.show()
    plt.close()


def plot_sequence(imstack, nrows=2, figsize=(18, 10), balance=False):
    f, ax = plt.subplots(nrows=nrows, ncols=int(
        np.ceil(len(imstack)/nrows)), figsize=figsize)
    ax = ax.ravel()

    vma = np.max(imstack)
    for i in np.arange(0, len(ax)):
        try:
            if balance:
                ax[i].imshow(imstack[i], cmap='gray', vmax=vma)
            else:
                ax[i].imshow(imstack[i], cmap='gray')

            ax[i].axis('off')
            ax[i].text(1, 0, i, fontsize=18, color='white', transform=ax[i].transAxes,
                       horizontalalignment='right', verticalalignment='bottom')
        except IndexError:
            ax[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

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

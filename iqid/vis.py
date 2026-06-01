"""Plotting and visualisation utilities for iQID data."""

import numpy as np
import matplotlib.pyplot as plt


def set_plot_parms():
    """Apply standard matplotlib rcParams and return the tab10 colormap."""
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
    """Draw vertical dashed lines at each value in es on the current axes.

    Parameters
    ----------
    es : list of float
        x-positions at which to draw lines.
    label : str, optional
        Legend label (drawn as a single proxy line).
    ls : str
        Line style.
    """
    if label is not None:
        plt.plot([], [], label=label, ls=ls, **kwargs)
    for e in es:
        plt.axvline(e, label=None, ls=ls, **kwargs)


def add_scalebar(scalebarlen, ax, px_mm=1 / (80 / 2048), loc='lower right', **kwargs):
    """Draw a labelled scalebar on ax.

    Parameters
    ----------
    scalebarlen : float
        Physical length in mm.
    ax : matplotlib Axes
    px_mm : float
        Pixels per millimeter (default: 39 µm effective pixel size for iQID).
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    scalebar = AnchoredSizeBar(
        ax.transData,
        scalebarlen * px_mm,
        '{} mm'.format(scalebarlen),
        loc=loc,
        **kwargs)
    ax.add_artist(scalebar)


def nice_colorbar(mappable, cbar_location="right", orientation="vertical", pad=0.05, **kwargs):
    """Add a neatly sized colorbar to a single-image figure."""
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbar_location, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)
    plt.sca(last_axes)
    return cbar


def plot_compare(slice1, slice2):
    """Side-by-side dose-rate comparison with a shared colorbar."""
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
                   cbar_pad=0.05)

    im1 = ax[0].imshow(slice1, cmap='inferno')
    vmi, vma = im1.get_clim()
    im2 = ax[1].imshow(slice2, cmap='inferno', vmin=vmi, vmax=vma)

    ax[1].cax.cla()
    cbar = Colorbar(ax[1].cax, im2)
    cbar.set_label('Absorbed Dose Rate\n(mGy/h)')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_title('Reference Image')
    ax[1].set_title('Comparison Image')
    plt.show()
    plt.close()


def plot_hists(slice1, slice2):
    """Overlay normalised intensity histograms for two images."""
    plt.figure(figsize=(8, 4))
    n1, bins1 = np.histogram(slice1.ravel(), density=True, bins=100)
    n2, bins2 = np.histogram(slice2.ravel(), density=True, bins=100)
    plt.step(bins1[1:-1], n1[1:], alpha=1, label='reference image')
    plt.step(bins2[1:-1], n2[1:], alpha=1, label='comparison image')
    plt.legend()
    plt.xlabel('Dose rate (mGy/h)')
    plt.ylabel('Frequency')
    plt.show()


def compare_metrics(ref, mov):
    """Print mean/max % difference and display visual + histogram comparison."""
    mean_diff = (np.mean(mov) - np.mean(ref)) / np.mean(ref)
    max_diff = (np.max(mov) - np.max(ref)) / np.max(ref)
    print('Mean % err (compared to image 1): {:.2f} %'.format(mean_diff * 100))
    print('Max % err (compared to image 1): {:.2f} %'.format(max_diff * 100))
    plot_compare_dose(ref, mov)
    plot_hists(ref, mov)


def plot_sequence(imstack, nrows=2, figsize=(18, 10), balance=False):
    """Display a stack of images as a grid with index labels."""
    f, ax = plt.subplots(nrows=nrows,
                         ncols=int(np.ceil(len(imstack) / nrows)),
                         figsize=figsize)
    ax = ax.ravel()
    vma = np.max(imstack)
    for i in np.arange(len(ax)):
        try:
            if balance:
                ax[i].imshow(imstack[i], cmap='gray', vmax=vma)
            else:
                ax[i].imshow(imstack[i], cmap='gray')
            ax[i].axis('off')
            ax[i].text(1, 0, i, fontsize=18, color='white',
                       transform=ax[i].transAxes,
                       horizontalalignment='right', verticalalignment='bottom')
        except IndexError:
            ax[i].axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

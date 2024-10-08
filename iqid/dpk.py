import numpy as np

'''Functions for manipulating a dose kernel and performing DPK convolution with an iQID activity image stack.'''


def load_txt_kernel(filename, dim, num_alpha_decays):
    """Loads .txt dose kernel into MeV/alpha/px np.array.

    File structure has 5 lines of header and cols (x, y, z, E).
    Assumes cubic (NxNxN) kernel with equal dimensions.

    Parameters
    ----------
    filename : str
        The filename of the dose kernel

    dim : int
        The length of one (all) dimension(s) of the kernel

    num_alpha_decays : float
        The number of alpha particles used in the Monte carlo simulation

    Returns
    -------
    dose_kernel : array-like, shape (dim, dim, dim)
        The structured dose kernel array in MeV/alpha/px
    """

    kernelData = np.genfromtxt(filename, delimiter=' ', skip_header=5)
    E = kernelData[:, -1]
    dims = (dim*np.ones(3)).astype(int)
    dose_kernel = E.reshape(dims[0], dims[1], dims[2])
    dose_kernel = dose_kernel/num_alpha_decays
    return (dose_kernel)


def mev_to_mgy(kernel, vox_vol_m, dens_kgm=1e3):
    """Converts an array of MeV values into mGy given voxel size and density.

    Does NOT assume square voxels, since most kernels will be binned in xy and
    in z differently depending on field of view data and slice thickness.

    Parameters
    ----------
    kernel : array_like, shape (N, N, N)
        The input kernel in MeV (or MeV/particle/unit)

    vox_vol_m : float
        The volume in m^3 of one voxel

    dens_kgm : float
        The density in kg/m^3 of the material

    Returns
    -------
    mgy_kernel : array_like, shape (N, N, N)
        The dose kernel in mGy (or mGy/particle/unit)
    """

    mass_kg_px = dens_kgm * vox_vol_m
    gy_kernel = kernel * 1e6 * 1.6021e-19 / mass_kg_px
    mgy_kernel = gy_kernel * 1e3
    return (mgy_kernel)


def radial_avg_kernel(kernel, mode="whole", bin_size=0.5):
    """Radially averages a Monte Carlo dose kernel to reduce variance.

    Can return either the radially averaged whole kernel (3D) for convolution
    or a radial segment (1D) for model analysis.

    Parameters
    ----------
    kernel : array_like, shape (N, N, N)
        The input kernel

    mode : str, "whole" or "segment"
        Choice of mode, 3D avg kernel or 1D avg segment

    bin_size : float
        The thickness (in distance units) of the ring being averaged

    Returns
    -------
    radial_avg : array_like, shape (N, N, N) OR (N//2)
        The radially averaged kernel or segment
    """

    if mode != 'whole' and mode != 'segment':
        print('Unsupported mode. Please select 3D "whole" or 1D "segment".')
        return (None)

    centerpt = len(kernel)//2
    a, b, c = kernel.shape

    # Grid of radial distances from the origin (centerpt)
    [X, Y, Z] = np.meshgrid(np.arange(a)-centerpt,
                            np.arange(b)-centerpt,
                            np.arange(c)-centerpt)
    R = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))

    # Set spatial resolution of computation
    rad = np.arange(0, np.max(R), 1)

    segment_avg_rad = np.zeros(len(rad))
    kernel_avg_rad = np.zeros_like(kernel)
    idx = 0

    for i in rad:
        mask = (np.greater(R, i-bin_size) & np.less(R, i+bin_size))
        values = kernel[mask]
        segment_avg_rad[idx] = np.mean(values)
        kernel_avg_rad[mask] = np.mean(values)
        idx += 1

    if mode == 'whole':
        return (kernel_avg_rad)
    elif mode == 'segment':
        return (segment_avg_rad)
    else:
        return (None)


def pad_kernel_to_vsize(kernel, vox_xy, slice_z=12):
    """Pad a 1-um dose kernel to binning-appropriate voxel dimensions.

    Given a desired voxel size in XY and a desired slice thickness in Z,
    pad the external part of the kernel with zeros to reach integer
    multiples of the voxel size (in µm), allowing for bin_ndarray afterwards.

    Parameters
    ----------
    kernel : array_like, shape (N, N, N)
        The input kernel

    vox_xy : int
        The desired final XY voxel size in µm

    slice_z : int
        The slice thickness and thus the desired Z voxel size in µm

    Returns
    -------
    padded_kernel : array_like, shape (N+z, N+x, N+x)
        The padded kernel with dimensions divisible by desired size
    """

    xy = np.round(vox_xy)
    z = np.round(slice_z)

    rem_xy = np.shape(kernel)[1] % xy
    rem_z = np.shape(kernel)[0] % z
    pad_xy = int(xy - rem_xy)
    pad_z = int(z - rem_z)

    if rem_xy == 0:
        pad_xy = 0
    if rem_z == 0:
        pad_z = 0

    padded_kernel = np.pad(kernel,
                           ((pad_z//2, pad_z//2 + pad_z % 2),
                            (pad_xy//2, pad_xy//2 + pad_xy % 2),
                            (pad_xy//2, pad_xy//2 + pad_xy % 2)),
                           'constant')

    return padded_kernel


def get_pd(thalf, framerate):
    """Get scalar correction factor to events missed in framerate
    due to the fastest-decaying progeny radionuclide.

    Parameters
    ----------
    thalf : float
        Half-life of the fast-decaying progeny, in s

    framerate : float
        Frame-rate of device in frames per second (fps)


    Returns
    -------
    pd : float in interval [0,1]
        Probability of double-decay, given observed parent decay.

    Derivation in accompanying manuscript.

    """
    lam = np.log(2)/thalf
    t1 = 1 / framerate
    p = 1 - 1/(lam*t1) * (1 - np.exp(-lam * t1))
    return p


def fr_corr_ac(framerate):
    """Wrapper for the Ac-225 case.

    Parameters
    ----------
    framerate : float
        Frame-rate of device in frames per second (fps)

    Returns
    -------
    out : float in interval [1,2]
        Scalar correction to recorded counts.

    """
    thalf_at217 = 32.3 * 1e-3
    pd = get_pd(thalf_at217, framerate)
    pa = 0.25 * (1-pd)
    pf = 0.25 / (0.25 + 0.25 + 0.25 + pa)
    out = 1 + pd * pf
    return out, pd, pf

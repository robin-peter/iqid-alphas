import os
import numpy as np
import cv2
import glob
from datetime import datetime
from functools import cached_property
from scipy.optimize import curve_fit
import imageio
from skimage import io, filters
import matplotlib.pyplot as plt
import matplotlib.cm
from tqdm import tqdm, trange

from iqid import helper


# ── Module-level functional API ────────────────────────────────────────────────

def exponential(x, a, thalf):
    return a * np.exp(-np.log(2) / thalf * x)


def eval_thalf(t_hours, cdat, time_threshold_hours=0, t_binsize=0.1, tol=0.1,
               time_duration=8, thalf=110/60, show_fulldat=True, return_fit=False):
    """Fit an exponential half-life to a time-series of events (hours).

    Convenience wrapper around iQData.fitHist. Default half-life is F-18.
    """
    if show_fulldat:
        og_nbins = np.round((t_hours[-1] - t_hours[0]) / t_binsize)
        ogn, og_bins = np.histogram(t_hours, bins=np.arange(0, og_nbins) * t_binsize)
        ogbe = 0.5 * (og_bins[1:] + og_bins[:-1])

    tdata = t_hours[(t_hours > time_threshold_hours) *
                    (t_hours < (time_threshold_hours + time_duration))]
    nbins = np.round((tdata[-1] - tdata[0]) / t_binsize)
    n, bins = np.histogram(tdata, bins=np.arange(0, nbins) * t_binsize + time_threshold_hours)
    binedges = 0.5 * (bins[1:] + bins[:-1])
    popt, pcov, param_std, res, chisq, chisqn = cdat.fitHist(
        binedges, n, func=exponential, p0=[1, thalf], tol=tol)

    tdummy = np.linspace(time_threshold_hours + 0.01, time_duration, 5000)
    plt.scatter(binedges, n)
    if show_fulldat:
        plt.scatter(ogbe, ogn)
    plt.plot(tdummy, exponential(tdummy, *popt))
    plt.xlabel('hours')
    plt.ylabel('counts/bin')
    plt.title('After {} h: Half-life of fit: {:.3f} h'.format(time_threshold_hours, popt[1]))
    plt.show()
    plt.close()

    if return_fit:
        return binedges, n, popt
    return binedges, n


# ── iQID class ────────────────────────────────────────────────────────────────

class iQData:

    # Maps ftype → glob pattern relative to datdir/Listmode/
    _LM_PATTERNS = {
        'processed_lm': '*Compressed_Processed_Listmode.dat',
        'offset_lm':    '*Offsets_Full_Raw_Listmode.dat',
        'clusters':     '*Cropped_Raw_Listmode.dat',
    }

    def __init__(self, source, ftype='processed_lm', c_area_thresh=15, makedir=False):
        """
        Parameters
        ----------
        source : str
            Acquisition directory (preferred) or direct path to a .dat file.
            When a directory is given the correct listmode file is located
            automatically inside source/Listmode/. Pass a file path directly
            to bypass discovery — useful when files are named non-standardly.
        ftype : {'processed_lm', 'offset_lm', 'clusters'}
        """
        if ftype not in self._LM_PATTERNS:
            raise ValueError(f"Unknown ftype '{ftype}'. Choose: {', '.join(self._LM_PATTERNS)}")

        self.ftype = ftype
        self.c_area_thresh = c_area_thresh

        if os.path.isdir(source):
            self.datdir = source
            self.file_name = self._find_lm_file(ftype)
        else:
            self.file_name = source
            # standard layout: file lives at datdir/Listmode/<name>
            self.datdir = os.path.dirname(os.path.dirname(source))

        self.savedir = os.path.join(self.datdir, 'Analysis')

        if makedir:
            os.makedirs(self.savedir, exist_ok=True)

    def _find_lm_file(self, ftype):
        """Return the path to the listmode file for ftype inside self.datdir/Listmode/."""
        lm_dir = os.path.join(self.datdir, 'Listmode')
        matches = helper.natural_sort(glob.glob(os.path.join(lm_dir, self._LM_PATTERNS[ftype])))
        if not matches:
            raise FileNotFoundError(
                f"No '{self._LM_PATTERNS[ftype]}' found in {lm_dir}")
        return matches[-1]

    @cached_property
    def acq_time(self):
        """Acquisition start datetime, parsed lazily on first access."""
        info_files = [f for f in helper.natural_sort(
            glob.glob(os.path.join(self.datdir, '*Acquisition_Info*.txt')))
            if 'Header' not in os.path.basename(f)]
        if not info_files:
            raise FileNotFoundError(f"No Acquisition_Info file found in {self.datdir}")
        with open(info_files[-1]) as f:
            lines = f.readlines()
        date_str = lines[0].rstrip('\n').split(': ', 1)[1]
        time_str = lines[1].rstrip('\n').split(': ', 1)[1]
        return datetime.strptime(date_str + ',' + time_str, "%A, %B %d, %Y,%I:%M:%S %p")

    def init_header(self):
        HEADER = np.fromfile(self.file_name, dtype=np.int32, count=100)
        self.HEADER_SIZE = HEADER[0]
        self.XDIM = HEADER[1]
        self.YDIM = HEADER[2]

        if self.ftype == 'processed_lm':
            self.NUM_DATA_ELEMENTS = 14
        elif self.ftype == 'offset_lm':
            self.NUM_DATA_ELEMENTS = 6
        elif self.ftype == 'clusters':
            self.cluster_radius = HEADER[20]
            self.cluster_imsize = 2 * self.cluster_radius + 1
            self.NUM_DATA_ELEMENTS = 8 + 2 * (self.cluster_imsize ** 2)
        else:
            raise (TypeError, "File format not specified: processed_lm, offset_lm, clusters")

        return self.HEADER_SIZE, self.XDIM, self.YDIM

    def set_process_params(self, binfac, ROI_area_thresh, t_binsize, t_half):
        self.binfac = binfac
        self.ROI_area_thresh = ROI_area_thresh
        self.t_binsize = t_binsize
        self.t_half = t_half

    def parse_acquisition_info(self):
        """Read the Acquisition_Info file and parse fps, shutter, and acquisition datetime.

        Sets self.fps, self.shutter, self.acq_time and returns them as a tuple.
        Replaces the former parse_framerate / parse_acqt / parse_shutter methods.
        """
        info_files = helper.natural_sort(
            glob.glob(os.path.join(self.datdir, '*Acquisition_Info*.txt')))
        info_files = [f for f in info_files if 'Header' not in os.path.basename(f)]
        if not info_files:
            raise FileNotFoundError(f"No Acquisition_Info file found in {self.datdir}")
        self.info_file = info_files[-1]

        with open(self.info_file, "r") as f:
            lines = f.readlines()

        date_str = lines[0].rstrip('\n').split(': ', 1)[1]
        time_str = lines[1].rstrip('\n').split(': ', 1)[1]
        self.acq_time = datetime.strptime(
            date_str + ',' + time_str, "%A, %B %d, %Y,%I:%M:%S %p")
        self.fps = float(lines[4].rstrip('\n').split(': ', 1)[1])
        self.shutter = float(lines[5].rstrip('\n').split(': ', 1)[1])

        return self.fps, self.shutter, self.acq_time

    def load_cluster_data(self, event_fx=1, dtype=np.float64):
        self.init_header()
        file_size_bytes = os.path.getsize(self.file_name)

        if dtype == np.float32 or dtype == np.int32:
            byteSize = 4
            byteFac = 1
        else:
            byteSize = 8
            byteFac = 2
        NUM_CLUSTERS = np.floor(
            (np.int64(file_size_bytes) - 4 * self.HEADER_SIZE) / (byteSize * self.NUM_DATA_ELEMENTS))
        NUM_LOAD = int(event_fx * NUM_CLUSTERS)

        unshaped_data = np.fromfile(
            self.file_name, dtype=dtype,
            count=self.HEADER_SIZE // byteFac + int(NUM_LOAD * self.NUM_DATA_ELEMENTS))
        data = unshaped_data[self.HEADER_SIZE // byteFac:].reshape(
            int(self.NUM_DATA_ELEMENTS), int(NUM_LOAD), order='F')
        return data

    def load_raws(self, cluster_size=10):
        rlistmode = glob.glob(os.path.join(
            os.path.dirname(self.file_name), '*Cropped_Raw_Listmode.dat'))
        if not rlistmode:
            raise Exception(
                "No cropped raw listmode file found in directory. "
                "Check that it is located in the same location: \n{}".format(self.file_name))
        rlistmode = rlistmode[0]

        byteSize = 4
        file_size_bytes = os.path.getsize(rlistmode)
        NUM_CLUSTERS = np.floor(
            (file_size_bytes - 4 * self.HEADER_SIZE) / (byteSize * self.NUM_DATA_ELEMENTS))
        unshaped_data = np.fromfile(
            rlistmode, dtype=np.int32,
            count=self.HEADER_SIZE + int(NUM_CLUSTERS * self.NUM_DATA_ELEMENTS))
        data = unshaped_data[self.HEADER_SIZE:].reshape(
            int(self.NUM_DATA_ELEMENTS), int(NUM_CLUSTERS), order='F')
        return data

    def init_metadata(self, data):
        """Parse a loaded data array into named attributes based on file type."""
        if self.ftype == 'processed_lm':
            frame_num          = data[0, :]
            time_ms            = data[1, :]
            sum_cluster_signal = data[2, :]
            cluster_area       = data[3, :]
            yC_global          = data[4, :]
            xC_global          = data[5, :]
            var_y              = data[6, :]
            var_x              = data[7, :]
            covar_xy           = data[8, :]
            eccentricity       = data[9, :]
            skew_y             = data[10, :]
            skew_x             = data[11, :]
            kurt_y             = data[12, :]
            kurt_x             = data[13, :]

            self.cluster_area = cluster_area
            self.cluster_sum  = sum_cluster_signal
            self.xC     = xC_global
            self.yC     = yC_global
            self.f      = frame_num
            self.time_ms = time_ms
            self.miss = None
            self.nevents_per_frame  = None
            self.offset_frame_time  = None

            return time_ms, cluster_area, xC_global, yC_global, frame_num

        elif self.ftype == 'offset_lm':
            frame_num    = data[0, :]
            time_ms      = data[1, :]
            n_miss       = data[2, :]
            n_cluster    = data[3, :]
            pix          = data[4, :]
            cam_temp_10K = data[5, :]

            self.f    = frame_num
            self.time_ms = time_ms
            self.miss = n_miss
            self.nevents_per_frame = n_cluster
            self.offset_frame_time = time_ms
            return frame_num, time_ms, n_miss, n_cluster, pix, cam_temp_10K

        elif self.ftype == 'clusters':
            a = self.cluster_imsize
            raw_imgs = data[:a**2, :].T
            # fil_imgs = data[a**2: 2*a**2, :]  # filtered/binarized cluster images

            frame_num    = data[2*a**2, :]
            yC           = data[2*a**2 + 1, :]
            xC           = data[2*a**2 + 2, :]
            # Element 4: Sum of the filtered cluster signal, all pixels above the threshold (size INT)
            cluster_sum  = data[2*a**2 + 3, :]
            # Element 6: Sum of filtered signal within the cropped sub-image
            cim_sum      = data[2*a**2 + 5, :]
            # Element 7: Time elapsed since start of acquisition (size Unsigned INT)
            time_ms      = data[2*a**2 + 6, :]
            # Element 8: Number of pixels (area) in the cluster (size INT)
            cluster_area = data[2*a**2 + 7, :]

            self.xC          = xC
            self.yC          = yC
            self.f           = frame_num
            self.cim_sum     = cim_sum
            self.cluster_area = cluster_area
            self.cluster_sum = cluster_sum
            self.raws        = raw_imgs
            self.time_ms     = time_ms

            return frame_num, time_ms, xC, yC, raw_imgs, cim_sum, cluster_area

        else:
            print('Accepted types: processed_lm, offset_lm, clusters')

    def image(self):
        cim = np.zeros((int(self.YDIM), int(self.XDIM)))
        for i in trange(len(self.xC), desc='Building image...'):
            cim[self.yC[i].astype(int), self.xC[i].astype(int)] += 1
        return cim

    def image_from_xy(self, x, y):
        x = np.round(x)
        y = np.round(y)
        cim = np.zeros((int(self.YDIM), int(self.XDIM)))
        for i in trange(len(x), desc='Building image...'):
            cim[y[i].astype(int), x[i].astype(int)] += 1
        return cim

    def image_from_distribution(self, ab_LUT):
        """For cluster images only. Requires a calibrated A/B lookup table."""
        if self.ftype != 'clusters':
            raise TypeError("Data set is not of type 'cluster': {}".format(self.ftype))
        Aim = np.zeros((int(self.YDIM), int(self.XDIM)))
        Bim = np.zeros((int(self.YDIM), int(self.XDIM)))
        for i in trange(len(self.xC), desc='Building 2 images...'):
            p = ab_LUT[self.cluster_area[i]]
            Aim[self.yC[i].astype(int), self.xC[i].astype(int)] += p[0]
            Bim[self.yC[i].astype(int), self.xC[i].astype(int)] += p[1]
        return Aim, Bim

    def image_from_listmode(self, subpx=1):
        self.init_header()
        data = self.load_cluster_data()
        time_ms, _, xC, yC, _ = self.init_metadata(data)

        xC_filtered = xC[np.logical_and(xC > 0, self.cluster_area > self.c_area_thresh)]
        yC_filtered = yC[np.logical_and(yC > 0, self.cluster_area > self.c_area_thresh)]

        if subpx == 1:
            xC_px = np.round(xC_filtered, 0)
            yC_px = np.round(yC_filtered, 0)
        else:
            xC_px = np.floor(xC_filtered * subpx)
            yC_px = np.floor(yC_filtered * subpx)

        cluster_bool = ((xC_px > 0) * (yC_px > 0)
                        * np.isfinite(xC_px) * np.isfinite(yC_px))
        xC_good = xC_px[cluster_bool].astype(int)
        yC_good = yC_px[cluster_bool].astype(int)

        cluster_image = np.zeros((int(subpx * self.YDIM), int(subpx * self.XDIM)))
        for i in range(len(yC_good)):
            cluster_image[yC_good[i], xC_good[i]] += 1

        time_s = time_ms[self.cluster_area > self.c_area_thresh] / 1e3
        time_s = time_s[cluster_bool]

        self.xC  = xC_good
        self.yC  = yC_good
        self.t_s = time_s

        return cluster_image

    def image_from_big_listmode(self, event_fx=0.1, xlim=(0, None), ylim=(0, None)):
        self.init_header()

        xlim = np.array(xlim)
        ylim = np.array(ylim)
        if xlim[1] is None:
            xlim[1] = self.XDIM
        if ylim[1] is None:
            ylim[1] = self.YDIM

        if xlim[1] > self.XDIM:
            raise ValueError('X-limit ({}, {}) exceeds acquisition width ({})'.format(
                xlim[0], xlim[1], self.XDIM))
        if ylim[1] > self.YDIM:
            raise ValueError('Y-limit ({}, {}) exceeds acquisition height ({})'.format(
                ylim[0], ylim[1], self.YDIM))

        data = self.load_cluster_data(event_fx=event_fx)
        time_ms, _, xC, yC, _ = self.init_metadata(data)

        cluster_bool = ((xC > xlim[0]) * (yC > ylim[0])
                        * (xC < xlim[1]) * (yC < ylim[1])
                        * np.isfinite(xC) * np.isfinite(yC)
                        * (self.cluster_area > self.c_area_thresh))
        xC   = np.round(xC[cluster_bool]).astype(int)
        yC   = np.round(yC[cluster_bool]).astype(int)
        t_s  = time_ms[cluster_bool] / 1e3

        cluster_image = np.zeros((self.YDIM, self.XDIM))
        for i in range(len(yC)):
            cluster_image[yC[i], xC[i]] += 1

        self.xC  = xC
        self.yC  = yC
        self.t_s = t_s

        return cluster_image

    def apply_selection(self, selection_bool):
        try:
            self.xC = self.xC[selection_bool]
            self.yC = self.yC[selection_bool]
            self.f  = self.f[selection_bool]
        except Exception as e:
            print(e)
        return self.xC, self.yC, self.f

    def get_subset(self, selection_bool):
        """Return a new Subset containing only the selected events (non-destructive)."""
        subset_data = Subset(self.file_name,
                             c_area_thresh=self.c_area_thresh,
                             makedir=False,
                             ftype=self.ftype)
        subset_data.HEADER_SIZE = self.HEADER_SIZE
        subset_data.XDIM = self.XDIM
        subset_data.YDIM = self.YDIM

        subset_data.xC           = self.xC[selection_bool]
        subset_data.yC           = self.yC[selection_bool]
        subset_data.f            = self.f[selection_bool]
        subset_data.time_ms      = self.time_ms[selection_bool]
        subset_data.cluster_area = self.cluster_area[selection_bool]
        subset_data.cluster_sum  = self.cluster_sum[selection_bool]

        if self.ftype == 'clusters':
            subset_data.raws         = self.raws[selection_bool]
            subset_data.cim_sum      = self.cim_sum[selection_bool]
            subset_data.cluster_area = self.cluster_area[selection_bool]
            subset_data.cluster_imsize = self.cluster_imsize
            subset_data.cluster_sum  = self.cluster_sum[selection_bool]

        return subset_data

    def get_mean_n(self, vis=False):
        """Find and set mean number of events per frame (requires offset file loaded)."""
        if self.nevents_per_frame is not None:
            n = self.nevents_per_frame
            self.mean_n = np.mean(n)
            if vis:
                plt.hist(n, bins=np.arange(12), edgecolor='white')
                plt.xlabel('number of events per frame')
                plt.ylabel('number of frames')
                plt.axvline(np.mean(n), color='gray', ls='--')
                plt.title('mean events/frame = {:.2f}'.format(np.mean(n)))
                plt.show()
            return self.mean_n
        else:
            print('Offset file not loaded.')

    def estimate_missed_timestamps(self, verbose=True):
        """Estimate timestamps of missed frames from the offset file."""
        if verbose:
            def iter_fn(n): return trange(n, desc="Estimating missed timestamps")
        else:
            iter_fn = range

        if self.miss is not None:
            m = self.miss
            t = self.offset_frame_time
            num_new_missed = np.diff(m)
            missed_events_time = np.array([])
            for i in iter_fn(len(num_new_missed)):
                if num_new_missed[i] > 0:
                    missed_events_time = np.append(
                        missed_events_time, np.repeat(t[i], num_new_missed[i]))
            return missed_events_time
        else:
            print('Offset file not loaded.')

    def filter_singles(self, fmax, vis=False, save=False, filter_time=False):
        vals, N = np.unique(self.f, return_counts=True)
        single_fnums = vals[N == 1]

        focc   = len(vals)
        fempty = fmax - focc

        if vis:
            pct_singles = len(single_fnums) / fmax * 100
            plt.hist(N, bins=np.arange(12), edgecolor='white')
            plt.xlabel('number of events per frame')
            plt.ylabel('number of frames')
            plt.title('{:.1f} % single-event, {} ({:.1f}%) empty'.format(
                pct_singles, fempty, fempty / fmax * 100))
            if save:
                plt.savefig(os.path.join(save), dpi=500, bbox_inches='tight')
            plt.show()

        cluster_bool = ((self.xC > 0) * (self.yC > 0)
                        * np.isfinite(self.xC) * np.isfinite(self.yC)
                        * np.isin(self.f, single_fnums))

        self.xC = self.xC[cluster_bool].astype(int)
        self.yC = self.yC[cluster_bool].astype(int)
        self.f  = self.f[cluster_bool]

        if filter_time:
            self.time_ms      = self.time_ms[cluster_bool]
            self.cluster_area = self.cluster_area[cluster_bool]

        return self.xC, self.yC, self.f, self.time_ms

    def set_coin_params(self, fps, t0_dt, TS_ROI, binfac=1, verbose=True):
        self.fps = fps
        self.exp = 1 / fps

        a = self.f * self.exp
        b = TS_ROI + t0_dt
        s_bins = np.arange(np.max(self.f) + 2, step=binfac) * self.exp

        assert np.all(a[:-1] <= a[1:]), "iQID array not monotonically increasing"
        assert np.all(b[:-1] <= b[1:]), "IDM array not monotonically increasing"

        if verbose:
            print('{:.2f} ms per frame'.format(self.exp * 1e3))
            print('iQID first 10:', a[:10])
            print('IDM first 10:', b[:10])
            print('Bins', s_bins[:10])

        self.a      = a
        self.b      = b
        self.s_bins = s_bins

        return a, b, s_bins

    def find_coin(self, singles=False, return_hist=False, verbose=True):
        if verbose:
            print('Generating iQID hist...')
        iq_n, _  = np.histogram(self.a, bins=self.s_bins)
        if verbose:
            print('Generating IDM hist...')
        idm_n, _ = np.histogram(self.b, bins=self.s_bins)

        coin  = np.logical_and(iq_n, idm_n)
        multi = np.sum(coin)
        if verbose:
            print('{} multi events found'.format(multi))

        if singles == 'iqid':
            iq_n = (iq_n == 1)
        elif singles == 'idm':
            idm_n = (idm_n == 1)
        elif singles == 'both':
            iq_n  = (iq_n == 1)
            idm_n = (idm_n == 1)

        coin = np.logical_and(iq_n, idm_n)
        if verbose:
            print('Selected {} coincident bins'.format(np.sum(coin)))

        if return_hist:
            return coin, iq_n, idm_n
        return coin

    def image_from_coin(self, coin=None, verbose=True, binfac=1, **kwargs):
        if coin is None:
            coin = self.find_coin(**kwargs)

        f_bins     = np.arange(np.max(self.f) + 2, step=binfac)
        good_f     = f_bins[:-1][coin]
        good_events = np.isin(self.f, good_f)

        if verbose:
            print('Found {:.2e} "good" events.'.format(np.sum(good_events)))

        x_good = self.xC[good_events]
        y_good = self.yC[good_events]

        nim = np.zeros((int(self.YDIM), int(self.XDIM)))
        if verbose:
            for i in trange(len(y_good), desc='Building selected image...'):
                nim[y_good[i], x_good[i]] += 1
        else:
            for i in range(len(y_good)):
                nim[y_good[i], x_good[i]] += 1

        return x_good, y_good, nim

    def check_elements(self, a, idx, x):
        if a[idx] == x:
            return idx
        else:
            idx += 1
            return self.check_elements(a, idx, x)

    def correct_frames(self, a1, a2, m1):
        """Map missed frame counts from offset file onto the full listmode frame array."""
        running_idx = 0
        m2 = np.zeros_like(a2)

        for i in trange(len(a2), desc='Assigning missed frames'):
            try:
                running_idx = self.check_elements(a1, running_idx, a2[i])
                m2[i] = m1[running_idx]
            except IndexError:
                print('aborting: listmode contains higher frames than offset. '
                      'Remainder filled with previous value.')
                m2[i:] = m2[i - 1]
                break

        return m2

    def correct_listmode(self, offset_frames, missed_frames, vis=True):
        a1 = offset_frames
        a2 = self.f
        m1 = missed_frames
        m2 = self.correct_frames(a1, a2, m1)

        u2, idx2 = np.unique(a2, return_index=True)
        u1, idx1, _ = np.intersect1d(a1, u2, return_indices=True)

        if vis:
            plt.plot(m1[idx1])
            plt.plot(m2[idx2])
            plt.xlabel('Index')
            plt.ylabel('Number of missed frames')
            plt.show()
            plt.close()

        corr_lm = m2 + self.f
        self.f = corr_lm
        return corr_lm

    def get_coms(self):
        """Get center-of-mass of each cluster image (clusters file type only)."""
        if self.ftype != 'clusters':
            raise TypeError("Data set is not of type 'cluster': {}".format(self.ftype))

        coms = np.zeros((len(self.raws), 2))
        for i in trange(len(self.raws), desc='Finding CoMs...'):
            im    = self.raws[i].reshape(self.cluster_imsize, self.cluster_imsize)
            thresh = filters.threshold_otsu(im)
            bin_im = (im > thresh).astype(int)
            cx, cy = helper.com(bin_im)
            coms[i, :] = cx, cy

        self.coms = coms
        return coms

    def filter_coms(self, size=7):
        """Discard cluster events whose center-of-mass falls outside a centered square."""
        if self.ftype != 'clusters':
            raise TypeError("Data set is not of type 'cluster': {}".format(self.ftype))

        coms = self.get_coms()
        x0   = self.cluster_imsize // 2 - size // 2
        good_coms = ((coms[:, 0] > x0) * (coms[:, 0] < x0 + size) *
                     (coms[:, 1] > x0) * (coms[:, 1] < x0 + size))
        self.com_bool = good_coms
        return self.get_subset(good_coms)

    def set_contour_params(self, gauss=15, thresh=0):
        self.gauss  = gauss
        self.thresh = thresh

    def prep_contour(self, im, gauss=15, thresh=0, binfac=1):
        binned_image = helper.bin_ndarray(
            im, (np.array(np.shape(im)) / binfac).astype(int), operation='sum')
        mask    = binned_image > thresh
        bin_im  = mask.astype('uint8')
        prep_im = cv2.GaussianBlur(bin_im, (gauss, gauss), 0)
        return prep_im

    def get_contours(self, im):
        prep_im = self.prep_contour(im=im, gauss=self.gauss, thresh=self.thresh)
        contours, _ = cv2.findContours(prep_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        good_contours = [
            self.binfac * c for c in contours if cv2.contourArea(c) > self.ROI_area_thresh]
        return good_contours

    def get_maskstack(self, im):
        xdim, ydim = np.shape(im)
        maskstack  = np.zeros((len(self.contours), xdim, ydim))
        for i in range(len(self.contours)):
            mask = np.zeros_like(im)
            cv2.drawContours(mask, self.contours, i, (255, 255, 255), -1)
            maskstack[i, :, :] = (mask / 255).astype(np.uint8)
        return maskstack

    def _load_manual_masks(self, mask_dir, fformat='png', pad=10):
        """Load pre-drawn mask PNGs directly into maskstack and ROIlists."""
        file_names = helper.natural_sort(
            glob.glob(os.path.join(mask_dir, f'*.{fformat}')))
        n = len(file_names)
        maskstack = []
        ROI_array = np.zeros((n, 4), dtype=int)
        for i, fname in enumerate(file_names):
            raw = io.imread(fname)
            if raw.ndim == 3:
                raw = raw[..., 0]
            binary = (raw > 0).astype(np.uint8)
            maskstack.append(binary)
            pts = cv2.findNonZero(binary)
            x, y, w, h = cv2.boundingRect(pts)
            ROI_array[i] = x - pad // 2, y - pad // 2, w + pad, h + pad
        return np.stack(maskstack), ROI_array

    def events_in_ROI(self, maskstack):
        ROI_array_bool = np.zeros((len(maskstack), len(self.xC)))
        for i in range(len(maskstack)):
            mask = maskstack[i, :, :]
            inROI_bool = (mask[self.yC.astype(int), self.xC.astype(int)]
                          * (self.t_s > 0) * np.isfinite(self.xC) * np.isfinite(self.yC))
            ROI_array_bool[i, :] = inROI_bool
        return ROI_array_bool.astype(bool)

    def get_ROIs(self, pad=10):
        ROI_array = np.zeros((len(self.contours), 4))
        for i in range(len(self.contours)):
            x, y, w, h = cv2.boundingRect(self.contours[i])
            ROI_array[i, :] = x - pad // 2, y - pad // 2, w + pad, h + pad
        return ROI_array.astype(int)

    def setup_ROIs(self, im, mode='auto', **kwargs):
        self._roi_mode = mode
        if mode == 'auto':
            self.contours  = self.get_contours(im)
            self.maskstack = self.get_maskstack(im)
            self.ROIlists  = self.get_ROIs()
        elif mode == 'manual':
            self.maskstack, self.ROIlists = self._load_manual_masks(**kwargs)
        elif mode == 'single':
            pass
        else:
            raise TypeError('Mode must be "auto", "manual", or "single".')
        self.ROIbool = self.events_in_ROI(self.maskstack)

    def fitHist(self, t, n, func=exponential, p0=None, tol=0.05):
        if p0 is None:
            p0 = [1, self.t_half]
        thalf = p0[1]
        popt, pcov = curve_fit(
            f=func, xdata=t, ydata=n,
            p0=p0,
            sigma=np.maximum(np.ones_like(n), np.sqrt(n)),
            bounds=([0, thalf * (1 - tol)], [np.inf, thalf * (1 + tol)]))
        param_std = np.sqrt(np.diag(pcov))
        res   = n - func(t, *popt)
        chisq = np.sum(res ** 2 / func(t, *popt))
        chisqn = chisq / len(n)
        return popt, pcov, param_std, res, chisq, chisqn

    def fitROI(self, temporal_array, func=exponential, p0=None, binsize=1000, tol=0.05):
        if p0 is None:
            p0 = [1, self.t_half]
        nbins = np.round(temporal_array[-1] / binsize)
        count, bins = np.histogram(temporal_array, np.arange(0, nbins) * binsize)
        timepoints  = bins[:-1] + binsize / 2
        popt, pcov, param_std, res, chisq, chisqn = self.fitHist(
            timepoints, count, func=func, p0=p0, tol=tol)
        return count, timepoints, popt, pcov, param_std, res, chisq, chisqn

    def get_imslice(self, im, idx):
        x, y, w, h = self.ROIlists[idx, :]
        return im[y:y + h, x:x + w] * self.maskstack[idx, y:y + h, x:x + w]

    def get_manual_maskstack(self):
        maskdir   = os.path.join(self.savedir, 'manual_masks')
        fileList  = glob.glob(os.path.join(maskdir, '*.tif'))
        man_maskstack = np.array(io.ImageCollection(fileList))

        good_contours = []
        for i in range(len(man_maskstack)):
            contours, _ = cv2.findContours(
                man_maskstack[i, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            gcontours = [c for c in contours if cv2.contourArea(c) > self.ROI_area_thresh]
            good_contours.append(gcontours)

        maskstack = np.zeros((len(good_contours), self.YDIM, self.XDIM))
        for i in range(len(good_contours)):
            mask = np.zeros((self.YDIM, self.XDIM))
            cv2.drawContours(mask, good_contours[i], -1, (255, 255, 255), -1)
            maskstack[i, :, :] = (mask / 255).astype(np.uint8)

        ROI_array_bool = np.zeros((len(maskstack), len(self.xC)))
        for i in range(len(maskstack)):
            mask = maskstack[i, :, :]
            inROI_bool = (mask[self.yC.astype(int), self.xC.astype(int)]
                          * (self.t_s > 0) * np.isfinite(self.xC) * np.isfinite(self.yC))
            ROI_array_bool[i, :] = inROI_bool

        pad = 10
        ROI_array = np.zeros((len(good_contours), 4))
        for i in range(len(good_contours)):
            x, y, w, h = cv2.boundingRect(good_contours[i][0])
            ROI_array[i, :] = x - pad // 2, y - pad // 2, w + pad, h + pad

        self.ROIlists  = ROI_array.astype(int)
        self.ROIbool   = ROI_array_bool.astype(bool)
        self.contours  = good_contours
        self.maskstack = maskstack
        return maskstack

    def save_cim(self, cim, fname='cim.tif'):
        os.makedirs(self.savedir, exist_ok=True)
        imageio.imwrite(os.path.join(self.savedir, fname), cim.astype(np.float32))
        print('cim saved to:', self.savedir)

    def save_manual_mask(self, mask):
        os.makedirs(self.savedir, exist_ok=True)
        plt.imsave(os.path.join(self.savedir, 'manual_mask_preview.png'), mask)
        imageio.imwrite(os.path.join(self.savedir, 'manual_mask.tif'), mask)
        print('Manual mask saved to:', self.savedir)

    def fitROIs(self, im, vis=True, tol=0.05, tstart=0, tcond=None, savemasks=False):

        iterlist = list(range(len(self.ROIlists)))
        self._fit_iterlist = iterlist

        if vis:
            f, ax = plt.subplots(nrows=len(iterlist), ncols=2,
                                 figsize=(8, 4 * len(iterlist)), squeeze=False)

        all_Ai  = np.zeros(len(iterlist))
        all_dAi = np.zeros(len(iterlist))

        for i in tqdm(range(len(iterlist)), desc='Getting ROIs'):
            t_data = self.t_s[self.ROIbool[iterlist[i]]]

            count, ts, popt, _, pstd, _, _, chisqn = self.fitROI(
                t_data, binsize=self.t_binsize, tol=tol)

            if tstart > 0:
                if vis:
                    ax[i, 1].axvspan(tstart, ts[-1], color='gray', alpha=0.3)
                gidx = ts > tstart
                popt, _, pstd, _, _, chisqn = self.fitHist(ts[gidx], count[gidx])
            elif tcond is not None:
                if len(tcond) not in (2, 4):
                    raise ValueError(
                        'tcond input should be (start1, stop1) OR (start1, stop1, start2, stop2)')
                if len(tcond) == 2:
                    if vis:
                        ax[i, 1].axvspan(tcond[0], tcond[1], color='gray', alpha=0.3)
                    gidx = (ts > tcond[0]) * (ts < tcond[1])
                    popt, _, pstd, _, _, chisqn = self.fitHist(ts[gidx], count[gidx])
                else:
                    if vis:
                        ax[i, 1].axvspan(tcond[0], tcond[1], color='gray', alpha=0.3)
                        ax[i, 1].axvspan(tcond[2], tcond[3], color='gray', alpha=0.3)
                    gidx = ((ts > tcond[0]) * (ts < tcond[1]) +
                            (ts > tcond[2]) * (ts < tcond[3]))
                    popt, _, pstd, _, _, chisqn = self.fitHist(ts[gidx], count[gidx])

            Ai  = popt[0] * 1000 / self.t_binsize
            dAi = pstd[0] * 1000 / self.t_binsize
            all_Ai[i]  = Ai
            all_dAi[i] = dAi

            if vis:
                imslice = self.get_imslice(im, iterlist[i])
                aslice  = Ai * imslice / np.sum(imslice)
                ax[i, 0].imshow(aslice, cmap='gray', vmax=0.15 * np.max(aslice))
                ax[i, 0].axis('off')
                xdummy = np.linspace(0, max(ts))
                ax[i, 1].scatter(ts, count)
                ax[i, 1].errorbar(ts, count, np.sqrt(count), ls='none', capsize=3)
                ax[i, 1].plot(xdummy, exponential(xdummy, *popt))
                ax[i, 1].set_xlabel('time (s)')
                ax[i, 1].set_ylabel('Counts in {} s bins'.format(self.t_binsize))
                ax[i, 1].set_title(
                    'Ai={:.0f} ± {:.0f} mBq  χ²/N={:.3f}'.format(
                        Ai, dAi, chisqn),fontsize=12)

        self.all_Ai  = all_Ai
        self.all_dAi = all_dAi

        if savemasks:
            if getattr(self, '_roi_mode', 'auto') == 'manual':
                print('savemasks skipped: manual masks already on disk.')
            else:
                os.makedirs(self.savedir, exist_ok=True)
                maskdir = os.path.join(self.savedir, 'masks')
                os.makedirs(maskdir, exist_ok=True)
                for i in range(len(iterlist)):
                    imageio.imwrite(os.path.join(maskdir, 'mask_{}.png'.format(i)),
                                   (255 * self.maskstack[iterlist[i], :, :]).astype(np.uint8))
                print('masks saved to:', maskdir)

    def save_rois(self, inc, order, im, new_name, decay_correct_to):
        """Save decay-corrected (or uncorrected) mBq images for selected ROIs.

        Parameters
        ----------
        inc : array-like
            Inclusion flags (1=include, 0=exclude), one per ROI in maskstack.
        order : array-like
            0-based output filename index for each included ROI, in display order.
        im : 2D array
            Full image passed to fitROIs.
        new_name : str
            Output subdirectory name under self.savedir.
        decay_correct_to : datetime or False
            Datetime to correct activity back to (typically sacrifice time).
            Pass False explicitly to save uncorrected Ai values.
        """
        if not hasattr(self, 'all_Ai'):
            raise RuntimeError("Call fitROIs before save_rois.")
        if decay_correct_to is not False:
            corr  = (self.acq_time - decay_correct_to).total_seconds()
            scale = np.exp(np.log(2) * corr / self.t_half)
        else:
            scale = 1.0
        outdir = os.path.join(self.savedir, new_name)
        os.makedirs(outdir, exist_ok=True)
        included_idxs = [i for i, v in enumerate(inc) if v]
        for j, roi_idx in enumerate(included_idxs):
            ai_pos = self._fit_iterlist.index(roi_idx)
            roi_im = self.get_imslice(im, roi_idx)
            aslice = self.all_Ai[ai_pos] * scale * roi_im / roi_im.sum()
            fname  = os.path.join(outdir, 'mBq_{}.tif'.format(order[j]))
            imageio.imwrite(fname, aslice.astype(np.float32))
        print(f'Saved {len(included_idxs)} ROI images to {outdir}')

    def plot_vis_masks(self, im):
        for i in range(len(self.maskstack)):
            plt.figure(figsize=(8, 6))
            helper_im = np.ma.array(im, mask=self.maskstack[i, :, :])
            _, helper_im = cv2.imencode('.png', helper_im.astype(np.uint8))
            helper_im = helper_im.tobytes()
            cmap = matplotlib.cm.inferno
            plt.imshow(helper_im, interpolation='nearest', cmap=cmap)
            plt.axis('off')
            plt.title(i)
            plt.show()
            plt.close()


class Subset(iQData):
    def __init__(self, file_name, c_area_thresh, makedir, ftype):
        super().__init__(file_name, c_area_thresh, makedir, ftype)

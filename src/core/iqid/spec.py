import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import becquerel as bq

from tqdm import trange

from iqid import helper as iq

from scipy.stats import linregress
pltmap = plt.get_cmap("tab10")


####################################################################################
#### Functions made for Hidex / BioD analysis in general (Ac225/Ce134)##############
####################################################################################

def sumROI_arr(arr, x, ROI):  # for use with hidex, takes in spec and kev as arr and x
    inROI = (x > ROI[0]) * (x < ROI[1])
    Nvals = arr[inROI]
    N = np.sum(Nvals[Nvals > 0])  # exclude negatives
    # todo: dN
    return N


def sumROI_spec(spec, ROI):
    inROI = (spec.bin_centers_kev > ROI[0]) * (spec.bin_centers_kev < ROI[1])
    Nvals = spec.counts_vals[inROI]
    N = np.sum(Nvals[Nvals > 0])  # exclude negatives
    # todo: dN
    return N


def th227_fit_hidex(s, ROI, model, vis=False):

    fitter = bq.Fitter(
        model,
        x=s.bin_centers_kev,
        y=s.counts_vals,
        y_unc=s.counts_uncs,
        roi=ROI,
    )

    fitter.fit()

    if vis:
        # this flag doesnt work for some reason >:(
        fitter.custom_plot(enable_fit_panel=False)
        plt.show()

    xdata = fitter.x_roi
    ydata = fitter.y_roi

    bg = fitter.x_roi * \
        fitter.result.params['linear_m'] + fitter.result.params['linear_b']
    gaussnet = fitter.eval(xdata, **fitter.best_values) - bg
    counts = np.sum(gaussnet)
    # chisq = fitter.result.chisqr

    # plt.plot(xdata, ydata)
    # plt.plot(xdata, bg)
    # plt.plot(xdata, gaussnet)
    return counts, fitter, bg


def load_hidex_xls(fpath, headstring="Results", headskip=19, colskip=2, other_idx=None, peak_flag=False):
    '''Gets the time stamp associated with the measurement as well as the dataframe of spectra.'''

    try:
        xls = pd.ExcelFile(fpath)
    except ValueError as e:
        print(
            'Pandas ValueError: make sure to close the data files before trying to import.')
        raise ValueError(e)

    # Spectrum sheet has consistent 19-row header. Rack/vial are not saved.
    spec_df = pd.read_excel(
        xls, sheet_name=1, skiprows=headskip, header=None).to_numpy()[:, colskip:]

    # Metadata sheet has variable rack-description header
    # Look for "Results" row header to separate metadata vs measured data
    meta_df = pd.read_excel(xls, sheet_name=0, header=None)
    results_row = meta_df.loc[meta_df.iloc[:, 0] == headstring].index[0]

    time_df = meta_df.iloc[(results_row + 2):, 2]
    ts = time_df[~time_df.isna()].to_numpy()

    if type(other_idx) is int:
        other_data = meta_df.iloc[(results_row + 2):, other_idx]

        if peak_flag:  # special flag to do analysis with Ac and Ce
            iso = other_data.to_numpy()
            e_peaks = []

            for i in range(len(iso)):
                peaks = []
                if pd.isna(iso[i]):
                    e_peaks.append(peaks)
                    continue
                if 'Ce' in iso[i]:
                    peaks.append(511)
                    peaks.append(1022)
                if 'Ac' in iso[i]:
                    peaks.append(218)
                    peaks.append(440)
                e_peaks.append(peaks)
            return spec_df, ts, other_data, e_peaks

        else:
            return spec_df, ts, other_data.to_numpy()
    else:
        return spec_df, ts


def pick_peaks(found_peak_array, es, de=50):
    good_centroid_array = []
    good_es = []
    for i in range(len(found_peak_array)):
        centroid = found_peak_array[i]
        differ = np.abs(es - centroid)
        if min(differ) < de:
            best_match = es[np.argmin(differ)]
            good_centroid_array.append(centroid)
            good_es.append(best_match)
    return good_centroid_array, good_es


def calibrate_spec(spectrum, es, kernel, min_snr=2, xmin=200, livetime=60., de=50, import_new=True, verbose=False):
    if import_new:
        spec = bq.Spectrum(counts=spectrum, livetime=livetime)
    else:
        spec = spectrum

    finder = bq.PeakFinder(spec, kernel)
    finder.find_peaks(min_snr=min_snr, xmin=xmin)
    good_peaks, good_es = pick_peaks(finder.centroids, es, de=de)

    if verbose:
        print('Centroids:', finder.centroids)
        print('Peaks:', good_peaks, good_es)

    cal = bq.Calibration.from_points(
        "p[0] + p[1] * x", good_peaks, good_es, params0=[5.0, 0.15])
    spec.apply_calibration(cal)
    return cal, spec


def get_cali_fn(a, n, nsamples=3, return_res=True):
    '''Given xdata (e.g. series of nCi activites) and ydata (e.g. Fr-221 ROI counts),
    return a function that calculates new X (A) from a new Y (cpm) measurement.

    Assumes nsamples = 3 samples at each concentration A.'''
    n = n.reshape(len(n)//nsamples, nsamples)
    x = np.mean(n, axis=1)
    dx = np.std(n, axis=1)  # not currently used
    res = linregress(x, a)

    def cali_fn(new_count_measurement):
        return res.intercept + res.slope * new_count_measurement

    if return_res:
        return cali_fn, res
    else:
        return cali_fn


def get_ac225_counts(spec, e_peaks, ROIFr=[180, 260], ROIBi=[400, 480], kernel=bq.GaussianPeakFilter(500, 50, fwhm_at_0=10), vis=False, xlim=(-10, 700)):
    frs = np.zeros(len(spec))
    bis = np.zeros(len(spec))
    totals = np.zeros(len(spec))

    for i in trange(len(spec)):
        try:
            cal, s = calibrate_spec(
                spec[i], e_peaks[i], kernel, import_new=True)
        except:  # CalibrationError: # if can't find peaks, use previous calibration
            s = bq.Spectrum(counts=spec[i], livetime=60.)
            s.apply_calibration(cal)

        frs[i] = sumROI_spec(s, ROIFr)
        bis[i] = sumROI_spec(s, ROIBi)
        totals[i] = np.sum(s.counts_vals)

        if vis:
            f, ax = plt.subplots(1, 1, figsize=(8, 4))

            s.plot(xlim=xlim, ax=ax)
            ax.set_ylabel('counts/bin (binw = {} keV)'.format(1))

            iq.draw_lines(e_peaks[i])
            plt.axvspan(ROIBi[0], ROIBi[1], color='gray', alpha=0.3)
            plt.axvspan(ROIFr[0], ROIFr[1], color='gray', alpha=0.3)

            plt.suptitle('Measurement {:.0f}'.format(i))
            # plt.yscale('log')
            plt.tight_layout()
            plt.show()
            plt.close()

    return frs, bis, totals


#######################################################################################

def gaussian(x, a, mu, width, b, c):
    return (a*np.exp(-(x-mu)**2 / (2*width**2)) + b*x + c)


def gaussian_nobg(x, a, mu, width):
    return (a*np.exp(-(x-mu)**2 / (2*width**2)))


def gaussian_cbg(x, a, mu, width, c):
    return (a*np.exp(-(x-mu)**2 / (2*width**2)) + c)


def load_masses_old(xls, sheet_name="Meta", headskip=23):
    meta_df = pd.read_excel(xls, sheet_name=sheet_name,
                            skiprows=headskip, header=None).to_numpy()[:, 1:]
    tumor_g = meta_df[0, :]
    kidney_g = meta_df[1, :]
    return (tumor_g, kidney_g)


def load_time_pi_old(xls, unit='h', sheet_name="Meta", headskip=17):
    if not unit in ['h', 'm', 's', 'ms']:
        raise ValueError(
            "Invalid unit '{}'. Accepted units: 'h', 'm', 's', 'ms'".format(unit))

    meta_df = pd.read_excel(xls, sheet_name=sheet_name,
                            skiprows=headskip, nrows=3, header=None)
    dt = pd.to_datetime(meta_df[1])
    spi = pd.Timedelta(dt[1]-dt[0]).total_seconds()
    sps = pd.Timedelta(dt[2]-dt[1]).total_seconds()

    base_unit = ['h', 'm', 's', 'ms'].index(unit)
    fac = [3600, 60, 1, 1e-6]

    return spi/fac[base_unit], sps/fac[base_unit]


def load_macpeg_xls_old(fpath):
    xls = pd.ExcelFile(fpath)
    time_pi, time_ps = load_time_pi_old(xls)
    tumor_g, kidney_g = load_masses_old(xls)
    tumor_spec = pd.read_excel(
        xls, sheet_name='PEG4-Tumor', header=0).to_numpy()[:, 1:].T
    kidney_spec = pd.read_excel(
        xls, sheet_name='PEG4-kidney', header=0).to_numpy()[:, 1:].T
    return (time_pi, time_ps, tumor_g, kidney_g, tumor_spec, kidney_spec)


def load_t0s(xls, sheet_name="Results", t0_rowskip=17, hpi_unit='h'):
    if not hpi_unit in ['h', 'm', 's', 'ms']:
        raise ValueError(
            "Invalid unit '{}'. Accepted units: 'h', 'm', 's', 'ms'".format(hpi_unit))
    base_unit = ['h', 'm', 's', 'ms'].index(hpi_unit)
    fac = [3600, 60, 1, 1e-6]

    meta_df = pd.read_excel(xls, sheet_name=sheet_name,
                            skiprows=t0_rowskip, nrows=3, header=None)
    dt = pd.to_datetime(meta_df[meta_df.columns[1]])
    # injection time to sac time
    spi = pd.Timedelta(dt[1]-dt[0]).total_seconds()
    # dt[2] is the counting start time, but each rack/vial has a more precise measurement time
    t0_sac_s = dt[1]
    return spi/fac[base_unit], t0_sac_s


def load_meta(xls, sheet_name="Results", header_rowskip=41, res_rowskip=102):
    df = pd.read_excel(xls, sheet_name=sheet_name,
                       skiprows=header_rowskip, header=0, nrows=0)
    df = pd.read_excel(xls, sheet_name=sheet_name,
                       skiprows=res_rowskip, header=None, names=list(df))
    dt = pd.to_datetime(df['Time'])
    mass_g = df['Sample mass (g)'].to_numpy()
    return dt, mass_g


def load_macpeg_xls(fpath, skiprows=79):
    xls = pd.ExcelFile(fpath)
    time_pi, t0_sac = load_t0s(xls)
    dt, mass_g = load_meta(xls)

    time_ps = np.zeros(len(dt))
    for i in range(len(dt)):
        time_ps[i] = pd.Timedelta(dt[i]-t0_sac).total_seconds()

    spectra = pd.read_excel(xls, sheet_name="Spectra",
                            skiprows=skiprows, header=None).to_numpy()[:, 2:]
    return (time_pi, time_ps, mass_g, spectra)


def ac225_spectra_stats(kev, spectra, roi1, roi2, min_peak_signal, vis=False, unpack=False, fp0=None, bp0=None):
    # hacky way to handle single array vs multiple array
    # there's probably a better way to do this?
    if len(np.shape(spectra)) == 1:
        spectra = spectra[np.newaxis, :]

    data_matrix = np.zeros((len(spectra), 6))
    # fr_nets, dfr_nets, fr_gross, bi_nets, dbi_nets, bi_gross

    for i in range(len(spectra)):
        espec = energy_spectrum(
            kev, spectra[i], min_peak_signal=min_peak_signal)

        if vis:
            espec.init_plot()

        # Fr
        if fp0 is not None:
            espec.fit_gaussian(roi1, p0=fp0, func=gaussian)
        else:
            espec.fit_gaussian(roi1, p0='fr', func=gaussian)
        net, dnet, gross = espec.get_stats(roi1)
        data_matrix[i, 0] = net
        data_matrix[i, 1] = dnet
        data_matrix[i, 2] = gross

        if vis:
            espec.plot_fit(roi1, 1)

        # Bi
        if bp0 is not None:
            espec.fit_gaussian(roi1, p0=bp0, func=gaussian)
        else:
            espec.fit_gaussian(roi2, p0='bi', func=gaussian)
        net, dnet, gross = espec.get_stats(roi2)
        data_matrix[i, 3] = net
        data_matrix[i, 4] = dnet
        data_matrix[i, 5] = gross

        if vis:
            espec.plot_fit(roi2, 2)
            plt.tight_layout()
            plt.show()

    # this is gross (ha, not net) but I'm too lazy to fix it right now

    if unpack:
        fr, dfr, frg, bi, dbi, big = np.split(data_matrix, 6, axis=1)
        return np.squeeze(fr), np.squeeze(dfr), np.squeeze(frg), np.squeeze(bi), np.squeeze(dbi), np.squeeze(big)
    else:
        return data_matrix


def calibrate_ac225(kev, spectra, roi1, roi2, uCi, min_peak_signal=5, t=60, vis=False):
    data_matrix = ac225_spectra_stats(
        kev, spectra, roi1, roi2, min_peak_signal=min_peak_signal, vis=vis)
    cts_theory = uCi * 1e-6 * 3.7e10 * t
    fr_theory = 0.116 * cts_theory
    bi_theory = 0.261 * cts_theory

    fr_eff = data_matrix[:, 0] / fr_theory
    bi_eff = data_matrix[:, 3] / bi_theory

    return np.mean(fr_eff), np.mean(bi_eff)


def correct_counts(n, br, eff):
    return n / br / eff


def get_activity_ratio(fr, dfr, bi, dbi, fr_eff, bi_eff):
    fr_br = 0.116  # intensity (br) of fr gamma ray
    bi_br = 0.261  # " " of bi gamma ray

    fr = correct_counts(fr, fr_br, fr_eff)
    # uncertainty propagated by scalar mult.
    dfr = correct_counts(dfr, fr_br, fr_eff)

    bi = correct_counts(bi, bi_br, bi_eff)
    dbi = correct_counts(dbi, bi_br, bi_eff)

    ratio = bi/fr
    dratio = ratio * np.sqrt((dfr/fr)**2 + (dbi/bi)**2)
    return ratio, dratio


def est_dr_ac225(fr, dfr, bi, dbi, mass_g, fr_eff, bi_eff, t):
    '''TODO:DEPRECATED'''
    fr_br = 0.116  # intensity (br) of fr gamma ray
    bi_br = 0.261  # " " of bi gamma ray

    # assume that each Fr decay implies 3 alphas with corresponding mean energies: (keV)
    # calculated in 0_1_alpha brs notebook
    en_first3 = 19160.841648908892
    en_last1 = 8323.036167

    fr_bq = correct_counts(fr, fr_br, fr_eff) / t
    # uncertainty propagated by scalar mult.
    dfr_bq = correct_counts(dfr, fr_br, fr_eff) / t

    bi_bq = correct_counts(bi, bi_br, bi_eff) / t
    dbi_bq = correct_counts(dbi, bi_br, bi_eff) / t

    fr_gy_s = fr_bq * en_first3 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dfr_gys = dfr_bq * en_first3 * 1e3 * 1.602e-19 / (mass_g * 1e-3)

    bi_gy_s = bi_bq * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dbi_gys = dbi_bq * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)

    gy_s = fr_gy_s + bi_gy_s
    dgys = np.sqrt(dfr_gys**2 + dbi_gys**2)

    return gy_s, dgys


def correct_activities(fr, dfr, bi, dbi, fr_eff, bi_eff, t):
    '''TODO:DEPRECATED'''
    fr_br = 0.116  # intensity (br) of fr gamma ray
    bi_br = 0.261  # " " of bi gamma ray
    fr_bq = correct_counts(fr, fr_br, fr_eff) / t
    # uncertainty propagated by scalar mult.
    dfr_bq = correct_counts(dfr, fr_br, fr_eff) / t
    bi_bq = correct_counts(bi, bi_br, bi_eff) / t
    dbi_bq = correct_counts(dbi, bi_br, bi_eff) / t

    return fr_bq, dfr_bq, bi_bq, dbi_bq


def corr_bi(fr, dfr, bi, dbi, t):
    '''TODO:DEPRECATED'''
    # using seconds
    # s; updated value (2019) from NNDC for Ac-225
    thalf_ac225 = 9.92 * 24 * 3600
    thalf_bi213 = 45.6 * 60  # s
    lamA = np.log(2)/thalf_ac225
    lamB = np.log(2)/thalf_bi213
    f = lamB / (lamB-lamA)
    b0 = bi * np.exp(lamB * t) - f * fr * (np.exp(lamB * t) - np.exp(lamA * t))

    d1 = dbi * np.exp(lamB * t)
    d2 = dfr * f * (np.exp(lamB * t) - np.exp(lamA * t))
    db0 = np.sqrt(d1**2 + d2**2)
    return (b0, db0)


def corr_ac(fr, dfr, t):
    '''TODO:DEPRECATED'''
    # s; updated value (2019) from NNDC for Ac-225
    thalf_ac225 = 9.92 * 24 * 3600
    lamA = np.log(2)/thalf_ac225
    fr = fr * np.exp(lamA * t)
    dfr = dfr * np.exp(lamA * t)
    return fr, dfr


def dr_ac225(fr, dfr, bi, dbi, mass_g, unit='gys'):
    '''TODO:DEPRECATED'''
    # inputs are corrected a0 and b0 at time of sacrifice [bq]
    # assume that each Fr decay implies 3 alphas with corresponding mean energies: (keV)
    # calculated in 0_1_alpdha brs notebook
    en_last1 = 8323.036167
    en_all = 19160.841648908892 + en_last1

    fr_gys = fr * en_all * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dfr_gys = dfr * en_all * 1e3 * 1.602e-19 / (mass_g * 1e-3)

    bi_gys = bi * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    dbi_gys = dbi * en_last1 * 1e3 * 1.602e-19 / (mass_g * 1e-3)
    if unit == 'mgyh':
        fr_out = fr_gys * 1e3 * 3600
        dfr_out = dfr_gys * 1e3 * 3600
        bi_out = bi_gys * 1e3 * 3600
        dbi_out = dbi_gys * 1e3 * 3600
    elif unit == 'gys':
        fr_out = fr_gys
        dfr_out = dfr_gys
        bi_out = bi_gys
        dbi_out = dbi_gys
    else:
        raise ValueError("Unaccepted unit type, use 'gys' or 'mgyh'.")
    return fr_out, dfr_out, bi_out, dbi_out


class energy_spectrum():

    def __init__(self, kev, n, xmin=0, xmax=2048, binsize=1, min_peak_signal=0):
        self.xmin = xmin
        self.xmax = xmax
        self.binsize = binsize
        self.kev = kev
        self.n = n
        self.thresh = min_peak_signal

    def init_plot(self):
        # defines set of axes and plots energy spectrum with two cutouts for ROIs
        f, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True,
                             gridspec_kw={'width_ratios': [3, 1, 1]})

        ax[0].plot(self.kev, self.n)
        ax[1].plot(self.kev, self.n)
        ax[2].plot(self.kev, self.n)

        # in future these can be variables, maybe param dictionary
        ax[0].set_xlim([0, 600])
        ax[1].set_xlim([150, 300])
        ax[2].set_xlim([370, 530])

        ax[0].set_xlabel('keV')
        ax[1].set_xlabel('Fr-221 ROI (218 keV)')
        ax[2].set_xlabel('Bi-213 ROI (440 keV)')

        ax[0].set_ylabel('counts/bin (binw = {} keV)'.format(self.binsize))

        self.f = f
        self.ax = ax

        return f, ax

    def fit_gaussian(self, ROIarray, p0='fr', func=gaussian, ret=False):
        # ROIarray = np.array([kev_min, kev_max]).astype(int) of ROI surrounding peak
        # can also do gaussian_nobg or gaussian_cbg
        xdata = self.kev[ROIarray[0]:ROIarray[1]]
        ydata = self.n[ROIarray[0]:ROIarray[1]]

        if np.max(ydata) < self.thresh:
            return np.nan, np.nan, np.nan

        if p0 == 'fr':
            p0 = [1000, 218, 30, -0.5, 10]
        elif p0 == 'bi':
            p0 = [1000, 440, 25, -0.1, 10]

        # uncertainty = sqrt(N) or 1
        dy = np.maximum(np.sqrt(ydata), np.ones_like(ydata))
        popt, pcov = curve_fit(
            func, xdata, ydata, sigma=dy, p0=p0, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))

        self.popt = popt
        self.perr = perr

        if ret:
            return popt, pcov, perr  # otherwise no return

    def get_stats(self, ROIarray):

        xdata = self.kev[ROIarray[0]:ROIarray[1]]
        ydata = self.n[ROIarray[0]:ROIarray[1]]

        if np.max(ydata) < self.thresh:
            return np.nan, np.nan, np.nan

        # uncertainty = sqrt(N) or 1
        dy = np.maximum(np.sqrt(ydata), np.ones_like(ydata))

        try:
            bg = self.popt[3] * xdata + self.popt[4]
            dbg = np.sqrt((self.perr[3])**2 + (self.perr[4])**2)
            darray = np.sqrt(dy**2 + dbg**2)
        except IndexError:  # no background (i.e. gaussian_nobg)
            bg = np.zeros_like(ydata)
            dbg = 0
        net_data = ydata - bg
        net_data = np.maximum(net_data, np.zeros_like(net_data))
        net_cts = np.sum(net_data)
        dnet = np.sqrt(np.sum(darray**2))
        gross_cts = np.sum(ydata)

        return net_cts, dnet, gross_cts

    def plot_fit(self, ROIarray, idx, func=gaussian):
        xspace = np.linspace(ROIarray[0], ROIarray[1], 100)
        xdata = self.kev[ROIarray[0]:ROIarray[1]]

        try:
            bg = self.popt[3] * xdata + self.popt[4]
            # dbg = np.sqrt((self.perr[3])**2 + (self.perr[4])**2)

            self.ax[0].plot(xspace, func(xspace, *self.popt),
                            color='black', label='Fit')
            self.ax[idx].plot(xspace, func(xspace, *self.popt),
                              color='black', label='Fit')
            self.ax[idx].plot(xdata, bg, color='gray')
            self.ax[0].legend()
        except AttributeError:
            print('No fit parameters found to plot.')
        except IndexError:
            self.ax[0].plot(xspace, func(xspace, *self.popt),
                            color='black', label='Fit')
            self.ax[idx].plot(xspace, func(xspace, *self.popt),
                              color='black', label='Fit')
            self.ax[0].legend()

    def test_p0(self, p0, ROIarray, idx, func=gaussian):
        self.init_plot()
        xspace = np.linspace(ROIarray[0], ROIarray[1], 100)
        self.ax[idx].plot(xspace, func(xspace, *p0),
                          color=pltmap(1), label='p0 function')
        self.ax[0].plot(xspace, func(xspace, *p0),
                        color=pltmap(1), label='p0 function')
        self.ax[0].legend()
        plt.tight_layout()
        plt.show()


def check_ROI(ROI, valid_dtypes):
    if not isinstance(ROI, valid_dtypes):
        print(f'ROI must be {valid_dtypes}. Input: {type(ROI)}')
    elif ROI[1] < ROI[0]:
        print(f'ROI must be in (Emin, Emax) format: {ROI}')
    elif ROI[1] > 2047 or ROI[0] < 0:
        print(f'ROI must fall in (0, 2047) CHA: {ROI}')
    else:
        return True


class BioD():

    def __init__(self, spec_matrix=None, mass_g=None, scorr=None, kev=np.arange(2048), live=60):
        self._spectra = spec_matrix
        self._mass = mass_g
        self._timeCorrection = scorr
        self._kev = kev
        self._livetime = live
        self._ROIFr = None
        self._ROIBi = None
        self._effFr = None
        self._effBi = None
        self._counts = None
        self._activity = None
        self._doserate = None
        self._ratio = None
        self._dratio = None

        self._p0Fr = None
        self._p0Bi = None

        # flags of stuff... is this ok?
        self._corrected = False

        # constants. is there a better practice for this?
        self._brFr = 0.116
        self._brBi = 0.261
        self._thalfAc225 = 9.92 * 24 * 3600  # s
        self._thalfBi213 = 45.6 * 60  # s
        self._kevBi = 8323.036167
        self._kevAc = 19160.84165 + self._kevBi

    @property
    def ROIFr(self):
        return self._ROIFr

    @ROIFr.setter
    def ROIFr(self, ROIFr):
        valid_dtypes = (tuple, list, np.ndarray)
        if check_ROI(ROIFr, valid_dtypes):
            self._ROIFr = ROIFr

    @property
    def ROIBi(self):
        return self._ROIBi

    @ROIBi.setter
    def ROIBi(self, ROIBi):
        valid_dtypes = (tuple, list, np.ndarray)
        if check_ROI(ROIBi, valid_dtypes):
            self._ROIBi = ROIBi

    @property
    def effFr(self):
        return self._effFr

    @effFr.setter
    def effFr(self, effFr):
        valid_dtypes = (int, float)
        if isinstance(effFr, valid_dtypes):
            self._effFr = effFr

    @property
    def effBi(self):
        return self._effBi

    @effBi.setter
    def effBi(self, effBi):
        valid_dtypes = (int, float)
        if isinstance(effBi, valid_dtypes):
            self._effBi = effBi

    def set_properties(self, ROIFr, ROIBi, effFr, effBi):
        self.ROIFr = ROIFr
        self.ROIBi = ROIBi
        self.effFr = effFr
        self.effBi = effBi

    @property
    def p0Fr(self):
        return self._p0Fr

    @ROIFr.setter
    def p0Fr(self, p0Fr):
        self._p0Fr = p0Fr

    @property
    def p0Bi(self):
        return self._p0Bi

    @ROIFr.setter
    def p0Bi(self, p0Bi):
        self._p0Bi = p0Bi

    '''TODO: properties for spectra, mass, time correction, etc'''
    '''TODO: catches for when requisite properties are not defined'''

    def get_data_matrix(self, min_peak_signal=5, vis=False):
        m = ac225_spectra_stats(self._kev,
                                self._spectra,
                                self._ROIFr,
                                self._ROIBi,
                                min_peak_signal=min_peak_signal,
                                vis=vis,
                                unpack=False,
                                fp0=self._p0Fr,
                                bp0=self._p0Bi)
        # fr, dfr, bi, dbi
        m = np.concatenate((m[:, :2], m[:, 3:5]), axis=1)

        # # fr, dfr, bi, dbi, mass (g)
        # m = np.concatenate((m, self._mass[:, np.newaxis]), axis=1) # TODO add getter/setter to mass to check it's 1d numpy array
        self._counts = m
        return m

    def cts_to_activity(self):
        # convert to activity, correcting counts for intensity and det eff
        m = self._counts
        n = np.zeros_like(m)
        n[:, 0] = correct_counts(
            m[:, 0], self._brFr, self._effFr) / self._livetime
        n[:, 1] = correct_counts(
            m[:, 1], self._brFr, self._effFr) / self._livetime
        n[:, 2] = correct_counts(
            m[:, 2], self._brBi, self._effBi) / self._livetime
        n[:, 3] = correct_counts(
            m[:, 3], self._brBi, self._effBi) / self._livetime
        self._activity = n
        return n

    def decay_corr(self, thresh=5*3600):
        if self._corrected:
            return self._activity

        t = self._timeCorrection
        m = self._activity
        fr, dfr, bi, dbi = m[:, 0], m[:, 1], m[:, 2], m[:, 3]

        lamA = np.log(2)/self._thalfAc225
        lamB = np.log(2)/self._thalfBi213

        # Fr
        f0 = fr * np.exp(lamA * t)
        df0 = dfr * np.exp(lamA * t)

        # Bi
        f = lamB / (lamB-lamA)
        b0 = bi * np.exp(lamB * t) - f * fr * \
            (np.exp(lamB * t) - np.exp(lamA * t))

        d1 = dbi * np.exp(lamB * t)
        d2 = dfr * f * (np.exp(lamB * t) - np.exp(lamA * t))
        db0 = np.sqrt(d1**2 + d2**2)

        # Free Bi
        fb0 = b0 - f0
        dfb0 = np.sqrt(df0**2 + db0**2)

        try:
            for i in range(len(t)):
                # if the time decay between sac and counting was too long
                # no (free) bi would be observed; true B(0) not knowable
                if t[i] > thresh:
                    fb0[i] = np.nan
                    dfb0[i] = np.nan
        except TypeError:  # TypeError: object of type 'int' has no len(), i.e. only one spectrum being analyzed
            if t > thresh:
                fb0 = np.nan
                dfb0 = np.nan

        n = np.array([f0, df0, fb0, dfb0]).T
        self._activity = n
        self._corrected = True
        return n

    def doserate(self, unit="mgyh"):
        if not self._corrected:
            print("WARNING: no recorded activity decay correction.")

        valid_units = ["mgyh", "gys"]
        scale_factor = [1e3 * 3600, 1]
        if unit not in valid_units:
            print(
                f"({unit}) must be from the following: {valid_units}. Defaulting to mGy/h.")

        m = self._activity
        n = np.zeros_like(m)

        n[:, :2] = m[:, :2] * self._kevAc * 1e3 * \
            1.602e-19 / (self._mass[:, np.newaxis] * 1e-3)
        n[:, 2:] = m[:, 2:] * self._kevBi * 1e3 * \
            1.602e-19 / (self._mass[:, np.newaxis] * 1e-3)
        self._doserate = n * scale_factor[valid_units.index(unit)]
        return self._doserate  # fr, dfr, bi, dbi

    def spec2dr(self, unit="mgyh"):
        # wrapper function for the different steps
        try:
            self.get_data_matrix()
            self.cts_to_activity()
            self.decay_corr()
            dr = self.doserate(unit=unit)
            return dr
        except TypeError:
            print("Aborted: Define parameters first with set_properties()")

    def activity_ratio(self):
        if self._corrected:
            print("WARNING: stored activity has already been time-corrected.")
        m = self._activity
        ratio = m[:, 2] / m[:, 0]  # Bi/Fr
        dratio = ratio * np.sqrt((m[:, 1]/m[:, 0])**2 + (m[:, 3]/m[:, 2])**2)
        # dratio = ratio * np.sqrt((dfr/fr)**2 + (dbi/bi)**2)
        self._ratio = ratio
        self._dratio = dratio
        return ratio, dratio


class MultiBioD():

    def __init__(self, dr_list=None, t_list=None):
        self._drlist = dr_list
        self._fr = None
        self._dfr = None
        self._bi = None
        self._dbi = None
        self._t = t_list  # e.g. "time post-injection, with no duplicates"
        self._tRavel = None

        self._frmean = None
        self._frstd = None
        self._bimean = None
        self._bistd = None

    @property
    def fr(self):
        return self._fr

    @property
    def dfr(self):
        return self._dfr

    @property
    def bi(self):
        return self._bi

    @property
    def dbi(self):
        return self._dbi

    @property
    def t(self):
        return self._t

    @property
    def tRavel(self):
        return self._tRavel

    @property
    def frmean(self):
        return self._frmean

    @property
    def frstd(self):
        return self._frstd

    @property
    def bimean(self):
        return self._bimean

    @property
    def bistd(self):
        return self._bistd

    def extract_drlist(self):
        fr, dfr, bi, dbi = np.array([]), np.array(
            []), np.array([]), np.array([])
        tRavel = np.array([])
        for i in range(len(self._drlist)):
            m = self._drlist[i]
            fr = np.append(fr, m[:, 0])
            dfr = np.append(dfr, m[:, 1])
            bi = np.append(bi, m[:, 2])
            dbi = np.append(dbi, m[:, 3])
            tRavel = np.append(tRavel, np.repeat(self._t[i], len(m)))
        # fr, dfr, bi, dbi = np.array(fr), np.array(dfr), np.array(bi), np.array(dbi)

        self._fr = fr
        self._dfr = dfr
        self._bi = bi
        self._dbi = dbi
        self._tRavel = tRavel

    def aggregate_drlist(self):
        fr = np.zeros(len(self._drlist))
        dfr = np.zeros(len(self._drlist))
        bi = np.zeros(len(self._drlist))
        dbi = np.zeros(len(self._drlist))

        for i in range(len(self._drlist)):
            m = self._drlist[i]
            means = np.mean(m, axis=0)
            stds = np.std(m, axis=0)
            fr[i], dfr[i], bi[i], dbi[i] = means[0], stds[0], means[2], stds[2]

        self._frmean = fr
        self._frstd = dfr
        self._bimean = bi
        self._bistd = dbi

    def dr2data(self):
        self.extract_drlist()
        self.aggregate_drlist()

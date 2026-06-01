#!/usr/bin/env python3
"""iQID acquisition data review GUI.

Usage
-----
    python data_review.py /path/to/acquisition_subfolder
    python data_review.py          # opens a folder-chooser dialog
"""

import sys
import os
import glob
import re
import threading
import argparse
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np
import tifffile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from iqid import process as pc
from iqid import helper, vis


PIXEL_UM = 39.0   # iQID effective pixel size (µm)


# ── Acquisition info parsing ───────────────────────────────────────────────────

def find_acq_info(datdir):
    """Return path to the main Acquisition_Info .txt (not the Header variant)."""
    hits = [f for f in glob.glob(os.path.join(datdir, 'Acquisition_Info*.txt'))
            if 'Header' not in f]
    return sorted(hits)[-1] if hits else None


def parse_acq_info(path):
    """Parse Acquisition_Info .txt → dict.

    Sets key '_crashed' True when End Acquisition Time or
    Acquisition Message Output are absent (indicates a device crash).
    """
    info = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, _, val = line.rstrip('\n').partition(':')
                info[key.strip()] = val.strip()
    info['_crashed'] = (
        'End Acquisition Time' not in info or
        'Acquisition Message Output' not in info
    )
    return info


def missed_count(info):
    """Extract missed-image count from Acquisition Message Output, or None."""
    m = re.search(r'Missed\s+(\d+)\s+images',
                  info.get('Acquisition Message Output', ''), re.I)
    return int(m.group(1)) if m else None


def aborted_info(info):
    """Return (n_captured, n_total) if acquisition was aborted early, else None."""
    m = re.search(r'(\d+)/(\d+)\s+Images\s+Captured',
                  info.get('Acquisition Message Output', ''), re.I)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_duration_h(val):
    """Parse '24 Hours', '30 Minutes', etc. → hours as float, or None."""
    val = val.strip()
    m = re.match(r'(\d+(?:\.\d+)?)\s*hour', val, re.I)
    if m:
        return float(m.group(1))
    m = re.match(r'(\d+(?:\.\d+)?)\s*min', val, re.I)
    if m:
        return float(m.group(1)) / 60
    return None


# ── Main application ───────────────────────────────────────────────────────────

class DataReviewApp:

    TABS = ['Acquisition Info', 'Image', 'Statistics']

    def __init__(self, root, datdir):
        self.root = root
        self.datdir = datdir
        self.root.title(f'iQID Data Review — {os.path.basename(datdir)}')

        # Populated by background thread
        self.cim = None
        self.t_s = None
        self.temp_c = None
        self.t_frame_h = None
        self.N_per_frame = None
        self.planned_duration_h = None

        self._build_shell()
        self._load_acq_info()
        threading.Thread(target=self._load_data, daemon=True).start()

    # ── Shell ──────────────────────────────────────────────────────────────────

    def _build_shell(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.tabs = {}
        for name in self.TABS:
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=name)
            self.tabs[name] = f
        for name in ['Image', 'Statistics']:
            ttk.Label(self.tabs[name], text='Loading data…',
                      font=('', 13)).pack(expand=True)

    def _clear(self, name):
        for w in self.tabs[name].winfo_children():
            w.destroy()

    def _error(self, name, msg):
        self._clear(name)
        ttk.Label(self.tabs[name], text=f'Error: {msg}',
                  foreground='red', font=('', 11)).pack(expand=True)

    # ── Tab 1: Acquisition Info ────────────────────────────────────────────────

    def _load_acq_info(self):
        path = find_acq_info(self.datdir)
        if path is None:
            self._error('Acquisition Info', 'No Acquisition_Info .txt found in folder.')
            return
        info = parse_acq_info(path)
        self.planned_duration_h = parse_duration_h(info.get('Acquisition Time', ''))
        self._populate_info_tab(info)

    def _populate_info_tab(self, info):
        self._clear('Acquisition Info')
        frame = self.tabs['Acquisition Info']

        crashed = info['_crashed']
        missed  = missed_count(info)
        aborted = aborted_info(info)

        # ── Status banner ──────────────────────────────────────────────────────
        if crashed:
            banner_txt = '⚠   DEVICE CRASH DETECTED — acquisition incomplete — use "Image" tab to export image '
            banner_bg  = '#c0392b'
        elif aborted:
            n_cap, n_tot = aborted
            parts = [f'Acquisition aborted early — {n_cap:,} / {n_tot:,} frames captured']
            if missed:
                parts.append(f'{missed} missed image{"s" if missed != 1 else ""}')
            banner_txt = '⚠   ' + ' — '.join(parts)
            banner_bg  = '#e67e22'
        elif missed:
            banner_txt = f'⚠   Acquisition complete — {missed} missed image{"s" if missed != 1 else ""}'
            banner_bg  = '#e67e22'
        else:
            banner_txt = '✓   Acquisition complete — no missed images'
            banner_bg  = '#27ae60'

        tk.Label(frame, text=banner_txt, bg=banner_bg, fg='white',
                 font=('', 13, 'bold'), pady=7,
                 anchor='w', padx=12).pack(fill=tk.X)

        # ── Parameter groups ───────────────────────────────────────────────────
        GROUPS = [
            ('Acquisition', [
                ('Date',                    'Date'),
                ('Start time',              'Time'),
                ('Duration',                'Acquisition Time'),
                ('End time',               'End Acquisition Time'),
                ('FPS',                     'Frame Rate (fps)'),
                ('Shutter (ms)',            'Shutter (ms)'),
                ('Number of frames',        'Number of Images'),
            ]),
            ('Camera', [
                ('Serial number',           'Camera Serial Number'),
                ('Gain (dB)',               'Camera Gain (dB)'),
                ('Image width',             'Image Width'),
                ('Image height',            'Image Height'),
                ('Effective pixel (µm)',    'Effective Pixel Size (microns)'),
                ('Acquisition mode',        'Acquisition Mode'),
            ]),
            ('Thresholds', [
                ('Min pixel',               'Pixel Minimum Threshold Setting'),
                ('Max pixel',               'Pixel Maximum Threshold Setting'),
                ('Min cluster area',        'Cluster Minimum Area Threshold Setting'),
                ('Max cluster area',        'Cluster Maximum Area Threshold Setting'),
                ('Filtering',               'Filtering Kernel and Size'),
            ]),
            ('Message output', [
                ('Message',                 'Acquisition Message Output'),
                ('End acquisition time',    'End Acquisition Time'),
            ]),
        ]

        grid = ttk.Frame(frame, padding=10)
        grid.pack(fill=tk.BOTH, expand=True)

        CRASH_KEYS = {'End Acquisition Time', 'Acquisition Message Output'}
        val_labels = []

        for i, (group_name, fields) in enumerate(GROUPS):
            r, c = divmod(i, 2)
            grp = ttk.LabelFrame(grid, text=group_name, padding=8)
            grp.grid(row=r, column=c, sticky='nsew', padx=5, pady=4)
            grid.columnconfigure(c, weight=1)
            grid.rowconfigure(r, weight=1)

            for row_i, (label, key) in enumerate(fields):
                val = info.get(key, '')
                missing = (val == '') and (key in CRASH_KEYS)
                display = val if val else ('⚠  MISSING' if missing else '—')
                color = '#c0392b' if missing else 'black'

                ttk.Label(grp, text=label + ':', foreground='gray40',
                          anchor='e').grid(row=row_i, column=0, sticky='e',
                                           padx=(0, 6), pady=1)
                lbl = ttk.Label(grp, text=display, foreground=color,
                                justify='left', anchor='w')
                lbl.grid(row=row_i, column=1, sticky='ew', pady=1)
                grp.columnconfigure(1, weight=1)
                val_labels.append(lbl)

        def _update_wraplength(event=None):
            wrap = max(120, frame.winfo_width() // 2 - 180)
            for lbl in val_labels:
                lbl.configure(wraplength=wrap)

        frame.bind('<Configure>', lambda e: _update_wraplength())
        frame.after(100, _update_wraplength)

    # ── Background loader ──────────────────────────────────────────────────────

    def _load_data(self):
        try:
            lm_files = helper.natural_sort(
                glob.glob(os.path.join(self.datdir, 'Listmode',
                                       '*Compressed_Processed_Listmode.dat')))
            if not lm_files:
                raise FileNotFoundError('No Compressed_Processed_Listmode.dat found.')

            cdat = pc.iQData(lm_files[-1], ftype='processed_lm', c_area_thresh=5)
            self.cim = cdat.image_from_big_listmode(event_fx=1)
            self.t_s = cdat.t_s
            self.root.after(0, self._populate_image_tab)

            off_files = helper.natural_sort(
                glob.glob(os.path.join(self.datdir, 'Listmode',
                                       '*Offsets_Full_Raw_Listmode.dat')))
            if off_files:
                off = pc.iQData(off_files[-1], ftype='offset_lm')
                data = off.load_cluster_data(dtype=np.int32)
                _, t_ms, _, N, _, kelv = off.init_metadata(data)
                self.t_frame_h  = t_ms / 1e3 / 3600
                self.temp_c     = kelv / 10 - 273.15
                self.N_per_frame = N
                self.root.after(0, self._populate_stats_tab)
            else:
                self.root.after(0, self._error, 'Statistics',
                                'Offsets_Full_Raw_Listmode.dat not found.')

        except Exception as exc:
            for name in ('Image', 'Statistics'):
                self.root.after(0, self._error, name, str(exc))

    # ── Tab 2: Image ──────────────────────────────────────────────────────────

    def _populate_image_tab(self):
        self._clear('Image')
        frame = self.tabs['Image']

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ttk.Frame(frame)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        ttk.Label(ctrl, text='Contrast (% of max):').pack(side=tk.LEFT)
        self._vmax_pct = tk.DoubleVar(value=10.0)
        ttk.Scale(ctrl, from_=1, to=100, orient=tk.HORIZONTAL,
                  variable=self._vmax_pct, length=200,
                  command=lambda _: self._redraw_image()).pack(side=tk.LEFT, padx=6)
        self._vmax_lbl = ttk.Label(ctrl, text='10%', width=5)
        self._vmax_lbl.pack(side=tk.LEFT)

        self._log_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text='Log scale', variable=self._log_var,
                        command=self._redraw_image).pack(side=tk.LEFT, padx=14)

        self._sb_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text='Scalebar', variable=self._sb_var,
                        command=self._redraw_image).pack(side=tk.LEFT, padx=(14, 2))
        self._sb_mm = tk.StringVar(value='10')
        sb_entry = ttk.Entry(ctrl, textvariable=self._sb_mm, width=5)
        sb_entry.pack(side=tk.LEFT)
        sb_entry.bind('<Return>',   lambda e: self._redraw_image())
        sb_entry.bind('<FocusOut>', lambda e: self._redraw_image())
        ttk.Label(ctrl, text='mm').pack(side=tk.LEFT, padx=(2, 0))

        ttk.Button(ctrl, text='Export raw TIF',
                   command=self._export_tif).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Label(ctrl, text='Adjust in ImageJ-->',
                  foreground='gray40').pack(side=tk.RIGHT, padx=(8, 0))

        # ── Paned layout: image left, histogram right ──────────────────────
        pw = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        left_pane  = ttk.Frame(pw)
        right_pane = ttk.Frame(pw)
        pw.add(left_pane,  weight=3)
        pw.add(right_pane, weight=2)

        self._img_fig, self._ax_im = plt.subplots(figsize=(7, 5))
        self._img_fig.tight_layout(pad=2)
        img_canvas = FigureCanvasTkAgg(self._img_fig, master=left_pane)
        img_toolbar = NavigationToolbar2Tk(img_canvas, left_pane, pack_toolbar=False)
        img_toolbar.update()
        img_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        img_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._img_canvas = img_canvas

        self._hist_fig, self._ax_hist = plt.subplots(figsize=(5, 5))
        self._hist_fig.tight_layout(pad=2)
        hist_canvas = FigureCanvasTkAgg(self._hist_fig, master=right_pane)
        hist_toolbar = NavigationToolbar2Tk(hist_canvas, right_pane, pack_toolbar=False)
        hist_toolbar.update()
        hist_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._hist_canvas = hist_canvas

        self._draw_temporal_hist()
        self._redraw_image()

    def _export_tif(self):
        if self.cim is None:
            return
        default = os.path.join(self.datdir,
                               os.path.basename(self.datdir) + '_image.tif')
        path = filedialog.asksaveasfilename(
            title='Export raw TIF',
            initialfile=os.path.basename(default),
            initialdir=self.datdir,
            defaultextension='.tif',
            filetypes=[('TIFF', '*.tif *.tiff'), ('All files', '*')])
        if path:
            tifffile.imwrite(path, self.cim)

    def _redraw_image(self):
        if self.cim is None:
            return
        pct = self._vmax_pct.get()
        self._vmax_lbl.config(text=f'{pct:.0f}%')

        ax = self._ax_im
        ax.cla()

        if self._log_var.get():
            pos = self.cim[self.cim > 0]
            floor = pos.min() if len(pos) else 1
            display = np.where(self.cim > 0, self.cim, floor)
            ax.imshow(display / display.max(), cmap='inferno', norm=LogNorm())
        else:
            ax.imshow(self.cim, cmap='inferno', vmax=(pct / 100) * self.cim.max())

        if self._sb_var.get():
            try:
                sb_mm = float(self._sb_mm.get())
            except (ValueError, tk.TclError):
                sb_mm = 10.0
            px_mm = 1e3 / PIXEL_UM
            ax.add_artist(AnchoredSizeBar(
                ax.transData, sb_mm * px_mm, f'{sb_mm:g} mm', 'lower left',
                color='white', frameon=False, size_vertical=3, borderpad=3))

        ax.axis('off')
        self._img_canvas.draw_idle()

    def _draw_temporal_hist(self):
        if self.t_s is None:
            return
        ax = self._ax_hist
        ax.cla()
        t_h = self.t_s / 3600
        n, bins, _ = ax.hist(t_h, bins=60, color='steelblue',
                             edgecolor='white', linewidth=0.3)
        be = 0.5 * (bins[1:] + bins[:-1])
        ax.errorbar(be, n, np.sqrt(np.maximum(n, 1)), fmt='none',
                    color='steelblue', capsize=2, linewidth=0.8)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Counts / bin')
        ax.set_title('Event time distribution')
        xmi, xma = ax.get_xlim()
        if self.planned_duration_h is not None:
            ax.set_xlim(xmi, max(xma, self.planned_duration_h))
        self._hist_fig.tight_layout(pad=2)
        self._hist_canvas.draw_idle()

    # ── Tab 3: Statistics ─────────────────────────────────────────────────────

    def _populate_stats_tab(self):
        self._clear('Statistics')
        frame = self.tabs['Statistics']

        cmap = vis.set_plot_parms()

        N = self.N_per_frame

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: temperature timeline
        ax1.step(self.t_frame_h, self.temp_c, color=cmap(0), linewidth=1)
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('iQID temperature (°C)')
        ax1.set_title('Temperature')

        # Right: distribution of events per frame
        ax2.hist(N, bins=np.arange(N.max()), color=cmap(0), edgecolor='white', linewidth=0.3,
                 label = f'min {N.min()}, max {N.max()},\nN={len(N):}')
        ax2.set_xlabel('Events per frame')
        ax2.set_ylabel('Frame count')
        ax2.set_title(f'Events per frame')
        ax2.legend()

        fig.tight_layout(pad=3)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        NavigationToolbar2Tk(canvas, frame, pack_toolbar=False).pack(
            side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='iQID acquisition data review GUI')
    parser.add_argument('datdir', nargs='?',
                        help='Path to acquisition subfolder')
    args = parser.parse_args()

    datdir = args.datdir
    if not datdir:
        tmp = tk.Tk()
        tmp.withdraw()
        datdir = filedialog.askdirectory(title='Select acquisition subfolder')
        tmp.destroy()
        if not datdir:
            sys.exit(0)

    if not os.path.isdir(datdir):
        sys.exit(f'Error: {datdir} is not a directory.')

    root = tk.Tk()
    root.geometry('1150x780')
    DataReviewApp(root, datdir)
    root.mainloop()


if __name__ == '__main__':
    main()

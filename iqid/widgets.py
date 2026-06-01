"""Interactive tkinter GUI for manual ROI labelling."""

import multiprocessing as mp

import numpy as np
from PIL import Image, ImageTk
from skimage import transform as skt


def _render_thumb(im, mask, thumb_size, vmax, deg):
    overlay = np.copy(im).astype(float)
    overlay[mask == 1] = np.max(overlay)
    if deg:
        overlay = skt.rotate(overlay, deg, preserve_range=True)
    mx = overlay.max()
    if mx > 0:
        overlay = overlay * 255.0 / (vmax * mx)
    arr = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).resize((thumb_size, thumb_size), Image.LANCZOS)


def _dim(pil_im, factor=0.3):
    arr = np.array(pil_im).astype(float)
    arr = arr * factor + 160 * (1 - factor)
    return Image.fromarray(arr.astype(np.uint8))


def _gui_worker(conn, thumbs_on_arr, thumbs_off_arr, cols, default_name, thumb_size):
    """Runs the tkinter GUI in a subprocess — avoids macOS/Jupyter event loop conflicts."""
    import tkinter as tk

    n = len(thumbs_on_arr)
    thumbs_on  = [Image.fromarray(a) for a in thumbs_on_arr]
    thumbs_off = [Image.fromarray(a) for a in thumbs_off_arr]

    included    = [True] * n
    photo_refs  = []
    thumb_labels = []

    root = tk.Tk()
    root.title("ROI Labelling")

    # ── Status bar ────────────────────────────────────────────────────────────
    top = tk.Frame(root, pady=4, padx=8)
    top.pack(side=tk.TOP, fill=tk.X)
    tk.Label(top, text="Click thumbnail to include / exclude",
             anchor='w').pack(side=tk.LEFT)
    status_var = tk.StringVar(value=f"Included: {n} / {n}")
    tk.Label(top, textvariable=status_var, anchor='e').pack(side=tk.RIGHT)

    def _update_status():
        status_var.set(f"Included: {sum(included)} / {n}")

    # ── Scrollable canvas ─────────────────────────────────────────────────────
    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True, padx=8)

    canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
    vsb = tk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    grid_frame = tk.Frame(canvas)
    cw = canvas.create_window((0, 0), window=grid_frame, anchor='nw')
    grid_frame.bind('<Configure>',
                    lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>',
                lambda e: canvas.itemconfig(cw, width=e.width))

    def _scroll(event):
        if event.num == 4:
            canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            canvas.yview_scroll(1, 'units')
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    canvas.bind_all('<MouseWheel>', _scroll)
    canvas.bind_all('<Button-4>', _scroll)
    canvas.bind_all('<Button-5>', _scroll)

    # ── ROI tiles ─────────────────────────────────────────────────────────────
    CELL_PAD = 6
    order_vars    = []
    order_entries = []

    def _make_toggle(idx):
        def _toggle(event=None):
            included[idx] = not included[idx]
            if included[idx]:
                photo = ImageTk.PhotoImage(thumbs_on[idx])
                thumb_labels[idx].config(image=photo, relief='flat')
                order_vars[idx].set(str(idx + 1))
                order_entries[idx].config(state='normal')
            else:
                photo = ImageTk.PhotoImage(thumbs_off[idx])
                thumb_labels[idx].config(image=photo, relief='sunken')
                order_vars[idx].set('—')
                order_entries[idx].config(state='disabled')
            thumb_labels[idx].image = photo  # prevent GC
            photo_refs[idx] = photo
            _update_status()
        return _toggle

    for i in range(n):
        r, c = divmod(i, cols)
        cell = tk.Frame(grid_frame, padx=CELL_PAD, pady=CELL_PAD)
        cell.grid(row=r, column=c, padx=2, pady=2)

        photo = ImageTk.PhotoImage(thumbs_on[i])
        photo_refs.append(photo)

        lbl = tk.Label(cell, image=photo)
        lbl.image = photo  # prevent GC
        lbl.pack()
        lbl.bind('<Button-1>', _make_toggle(i))
        thumb_labels.append(lbl)

        ctrl = tk.Frame(cell)
        ctrl.pack(fill=tk.X)
        tk.Label(ctrl, text=f'#{i+1}', width=4, anchor='e').pack(side=tk.LEFT)
        ovar = tk.StringVar(value=str(i + 1))
        order_vars.append(ovar)
        ent = tk.Entry(ctrl, textvariable=ovar, width=4, justify='center')
        ent.pack(side=tk.LEFT, padx=2)
        order_entries.append(ent)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    bottom = tk.Frame(root, pady=6, padx=8, relief='groove', borderwidth=1)
    bottom.pack(side=tk.BOTTOM, fill=tk.X)

    tk.Label(bottom, text='Save as:').pack(side=tk.LEFT)
    name_var = tk.StringVar(value=default_name or '')
    tk.Entry(bottom, textvariable=name_var, width=24).pack(side=tk.LEFT, padx=4)

    def _on_apply():
        inc = [1 if included[i] else 0 for i in range(n)]

        # Collect (display_idx, user_value) for included ROIs
        order_vals = []
        for i in range(n):
            if not included[i]:
                continue
            try:
                v = int(order_vars[i].get().strip())
            except ValueError:
                v = i + 1
            order_vals.append((i, v))

        vals = [v for _, v in order_vals]
        if len(set(vals)) != len(vals):
            import tkinter.messagebox as mb
            mb.showerror("Duplicate order values",
                         "Each included ROI must have a unique order number.")
            return

        # Sort by user value; map display position → output filename index
        sorted_by_order = sorted(order_vals, key=lambda x: x[1])
        display_to_out  = {disp: out for out, (disp, _) in enumerate(sorted_by_order)}
        order = [display_to_out[i] for i, _ in order_vals]

        new_name = name_var.get().strip()
        conn.send(('apply', inc, order, new_name))
        root.destroy()

    def _on_cancel():
        conn.send(('cancel', None, None, None))
        root.destroy()

    tk.Button(bottom, text='Apply',  width=10, command=_on_apply).pack(side=tk.RIGHT, padx=4)
    tk.Button(bottom, text='Cancel', width=10, command=_on_cancel).pack(side=tk.RIGHT)

    root.bind('<Return>', lambda e: _on_apply())
    root.bind('<Escape>', lambda e: _on_cancel())
    root.protocol('WM_DELETE_WINDOW', _on_cancel)

    n_rows = int(np.ceil(n / cols))
    cell_w = thumb_size + 2 * CELL_PAD + 20
    cell_h = thumb_size + 2 * CELL_PAD + 36
    win_w  = min(cols * cell_w + 40,  root.winfo_screenwidth()  - 80)
    win_h  = min(n_rows * cell_h + 120, root.winfo_screenheight() - 100)
    root.geometry(f'{win_w}x{win_h}')

    root.mainloop()
    conn.close()


def label_rois(maskstack, im, vmax=1, deg=0, cols=4, thumb_size=150,
               default_name='slice_images'):
    """Open an interactive GUI for ROI labelling.

    Parameters
    ----------
    maskstack : (N, H, W) array
    im : 2D array
    vmax, deg, cols, thumb_size : display options
    default_name : str
        Pre-fills the 'Save as' folder name field.

    Returns
    -------
    inc : ndarray of int (0/1), shape (N,)
    order : ndarray of int, shape (n_included,)
    new_name : str
    """
    n = len(maskstack)
    thumbs_on_arr  = [np.array(_render_thumb(im, maskstack[i], thumb_size, vmax, deg))
                      for i in range(n)]
    thumbs_off_arr = [np.array(_dim(Image.fromarray(t))) for t in thumbs_on_arr]

    ctx = mp.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_gui_worker,
                    args=(child_conn, thumbs_on_arr, thumbs_off_arr,
                          cols, default_name, thumb_size))
    p.start()
    child_conn.close()

    try:
        action, inc_list, order_list, new_name = parent_conn.recv()
    except EOFError:
        action = 'cancel'
    finally:
        p.join()

    if action == 'cancel':
        raise RuntimeError("ROI labelling cancelled.")
    return np.array(inc_list, dtype=int), np.array(order_list, dtype=int), new_name

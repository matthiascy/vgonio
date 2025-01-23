import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import multiprocessing
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings to avoid clogging the console
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

def errormap(data, rowlabels, collabels, ax=None, enable_cbar=True, cbarkw=None, cbarlabel="", vmin=None, vmax=None, **kwargs):
    """Plot a 2D error map with square pixels.

    Parameters
    ----------
    data : ndarray
        2D numpy array of the error map
    rowlabels : list
        Labels for the rows
    collabels : list
        Labels for the columns 
    ax : matplotlib.axes.Axes, optional
        Matplotlib axis to plot on. If None, uses current axis
    cbarkw : dict, optional
        Keyword arguments for the colorbar
    cbarlabel : str, optional
        Label for the colorbar
    vmin : float, optional
        Minimum value for the colorbar
    vmax : float, optional
        Maximum value for the colorbar
    **kwargs
        Additional keyword arguments passed to imshow

    Returns
    -------
    tuple
        (matplotlib.image.AxesImage, matplotlib.colorbar.Colorbar)
        The displayed image and colorbar
    """
    if ax is None:
        ax = plt.gca()

    if cbarkw is None:
        cbarkw = {}
    
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    # Plot the error map
    im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)

    cbar = None
    if enable_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbarkw)
        cbar.set_label(cbarlabel, rotation=-90, va='bottom')

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=collabels,
                  rotation=45, ha='left', rotation_mode='anchor', fontfamily='monospace')
    ax.set_yticks(range(data.shape[0]), labels=rowlabels, fontfamily='monospace')

    # Make the horizontal axes labels line up appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_errormap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """Annotate a heatmap/error map with text values.

    Parameters
    ----------
    im : matplotlib.image.AxesImage
        The image to annotate, as returned by imshow()
    data : array-like, optional
        The data to annotate. Defaults to the array from im
    valfmt : str or matplotlib.ticker.Formatter, optional
        Format of the annotations. If str, should use Python string formatting, 
        e.g. "{x:.2f}". Defaults to "{x:.2f}"
    textcolors : list of str, optional
        List of two color specifications - the first is used for values below a threshold,
        the second for those above. Defaults to ["black", "white"]
    threshold : float, optional
        Value where the text colors switch. If None, defaults to the middle of the 
        colormap range. Defaults to None
    **textkw
        Additional keyword arguments passed to matplotlib.text.Text

    Returns
    -------
    list of matplotlib.text.Text
        The text objects created for each "pixel"
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2

    # Set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt) 

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

MAX_DIM_PER_SUBPLOT = 64

def plot_brdf_error_map_single(i, name, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax, cbarlabel='residual'):
    """
    Plot a single brdf error map.

    Parameters
    ----------
    i : int
        Index of the wavelength to plot
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    model_name : str
        Name of the model
    metric : str
        Metric to plot
    x_labels : list
        Labels for the x-axis
    y_labels : list
        Labels for the y-axis
    maps : list
        List of maps to plot; each map is a 2D numpy array
    vmin : float, optional
        Minimum value for the colorbar
    vmax : float, optional
        Maximum value for the colorbar
    """
    (_, imgw, imgh) = maps.shape
    if imgw > MAX_DIM_PER_SUBPLOT or imgh > MAX_DIM_PER_SUBPLOT:
        n_rows = math.ceil(imgw / MAX_DIM_PER_SUBPLOT)
        n_cols = math.ceil(imgh / MAX_DIM_PER_SUBPLOT)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 * n_cols, 16 * n_rows)) 
        axes = axes.reshape(n_rows, n_cols)

        for i in range(n_rows):
            for j in range(n_cols):
                x_start = i * MAX_DIM_PER_SUBPLOT
                x_end = min(x_start + MAX_DIM_PER_SUBPLOT, imgw)
                y_start = j * MAX_DIM_PER_SUBPLOT
                y_end = min(y_start + MAX_DIM_PER_SUBPLOT, imgh)
                data = maps[i, x_start:x_end, y_start:y_end]
                x_labels_sub = x_labels[x_start:x_end]  
                y_labels_sub = y_labels[y_start:y_end]
                im, cbar, _ = plot_brdf_error_map_subplot(data, x_labels_sub, y_labels_sub, axes[i, j], i * n_cols + j + 1, n_rows * n_cols, enable_annotate=False, enable_cbar=False, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=axes.ravel()[-1], location='right')
        cbar.set_label(cbarlabel, rotation=-90, va='bottom')
    else:
        fig, axes = plt.subplots(figsize=(16, 16)) 
        plot_brdf_error_map_subplot(maps[i], x_labels, y_labels, axes, 1, 1, enable_annotate=True, enable_cbar=True, cbarlabel=cbarlabel, vmin=vmin, vmax=vmax)
    
    fig.suptitle(fr'{name} $\lambda = {spectrum[i]:.2f}$ nm') 
    fig.savefig(f"{name}_{model_name}_{metric}_{spectrum[i]:.2f}nm.png", bbox_inches='tight')
    plt.close()

def plot_brdf_error_map_subplot(data, x_labels, y_labels, ax, part_idx=1, total_parts=1, enable_annotate=True, enable_cbar=True, cbarlabel='residual', vmin=None, vmax=None, **kwargs):
    """
    Plot a 2D error map with square pixels.
    """
    im, cbar = errormap(data, x_labels, y_labels, ax=ax, enable_cbar=enable_cbar, cmap='YlGn', cbarlabel=cbarlabel, vmin=vmin, vmax=vmax)  

    texts = None
    if enable_annotate:
        texts = annotate_errormap(im, valfmt='{x:.2f}')

    ax.set_ylabel(r'Incident Direction $\omega_i = (\theta_i, \phi_i)$')
    ax.set_xlabel(r'Outgoing Direction $\omega_o = (\theta_o, \phi_o)$')

    if total_parts > 1:
        ax.set_title(fr'{part_idx} / {total_parts}')
    
    return im, cbar, texts

def task(progress, tid, base_idx, count, name, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax):
    """
    Task to plot the brdf error map in parallel.

    Parameters
    ----------
    progress : dict
        Progress dictionary
    tid : int
        Task ID
    base_idx : int
        Base index
    count : int
        Number of maps to plot
    model_name : str
        Name of the model
    metric : str
        Metric to plot
    x_labels : list
        Labels for the x-axis
    y_labels : list
        Labels for the y-axis
    maps : list
        List of maps to plot; each map is a 2D numpy array
    """
    for i in range(base_idx, base_idx + count):
        time.sleep(0.05)
        plot_brdf_error_map_single(i, name, model_name + str(i), metric, x_labels, y_labels, maps, spectrum, vmin, vmax)
        progress[tid] = { "progress": i - base_idx + 1, "total": count }

def plot_brdf_error_map(name, residuals, maps, metric, model_name, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel):
    """
    Plot the brdf error map. Entry point for the brdf error map plot in Rust. 

    Parameters
    ----------
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    residuals : ndarray
        Residuals between the measured data and the model. The shape of residuals could be either 
            - ϑi, ϕi, ϑo, ϕo, λ - grid  
            - ϑi, ϕi, ωo, λ - list
    maps : ndarray
        Residuals arranged in a 2D array with dimension: [Nλ, Nωo, Nωi], where Nλ is the number of wavelengths, 
        Nωo is the number of outgoing directions, and Nωi is the number of incident directions.
        In most cases, Nωi is computed by Nωi = Nθi * Nφi, where Nθi and Nφi are the number of incident directions 
        in the theta (inclination) and phi (azimuth) directions. Depends on the measured BRDF data, the outgoing 
        directions could be either a grid or a list; if it is a grid, Nωo = Nθo * Nφo, where Nθo and Nφo are the 
        number of outgoing directions in the theta (inclination) and phi (azimuth) directions. 
        If it is a list, Nωo is the length of the `o_phis` list.
    metric : str
        Metric used to compute the residuals
    model_name : str
        Name of the brdf model
    i_thetas : ndarray
        Incident directions in the theta (inclination) direction
    i_phis : ndarray
        Incident directions in the phi (azimuth) direction
    o_thetas : ndarray
        Outgoing directions in the theta (inclination) direction
    o_phis : ndarray
        Outgoing directions in the phi (azimuth) direction
    offsets : ndarray
        Offsets of the outgoing directions. Used only when the outgoing directions are a list. 
        This is a list of length Nφo + 1, the element at index `i` is the start index of phi (azimuth) 
        direction for the outgoing theta at index `i`; the element at index `i+1` is the end index of 
        phi (azimuth) direction for the outgoing theta at index `i`. To get all the combinations of the 
        outgoing directions, iterate over the `o_thetas` and use `np.arange(offsets[i], offsets[i + 1])` 
        to get all the phi (azimuth) directions for the theta (inclination) direction at index `i`.
    spectrum : ndarray
        Wavelengths in nanometers
    parallel : bool
        Whether to plot in parallel or not
    """
    # Check if outgoing directions offsets exists
    is_grid = not isinstance(offsets, np.ndarray)

    # 1. plot residual maps per wavelength
    #    each map contains all the incident directions and all the outgoing directions
    if is_grid:
        (n_theta_i, n_phi_i, n_theta_o, n_phi_o, n_lambda) = residuals.shape

        x_labels = np.empty(n_theta_i * n_phi_i, dtype=object)
        for ti in range(n_theta_i):
            for pi in range(n_phi_i):
                theta_deg = np.degrees(i_thetas[ti])
                phi_deg = np.degrees(i_phis[pi])
                if pi == 0 or pi % MAX_DIM_PER_SUBPLOT == 0:
                    x_labels[ti * n_phi_i + pi] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    x_labels[ti * n_phi_i + pi] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'

        y_labels = np.empty(n_theta_o * n_phi_o, dtype=object)
        for to in range(n_theta_o):
            for po in range(n_phi_o):
                theta_deg = np.degrees(o_thetas[to])
                phi_deg = np.degrees(o_phis[po])
                if po == 0 or po % MAX_DIM_PER_SUBPLOT == 0:
                    y_labels[to * n_phi_o + po] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    y_labels[to * n_phi_o + po] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'
    else:
        n_theta_o = len(o_thetas)
        (n_theta_i, n_phi_i, n_omega_o, n_lambda) = residuals.shape

        x_labels = np.empty(n_theta_i * n_phi_i, dtype=object)
        for ti in range(n_theta_i):
            for pi in range(n_phi_i):
                theta_deg = np.degrees(i_thetas[ti])
                phi_deg = np.degrees(i_phis[pi])
                if pi == 0 or pi % MAX_DIM_PER_SUBPLOT == 0:
                    x_labels[ti * n_phi_i + pi] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    x_labels[ti * n_phi_i + pi] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'
        
        y_labels = np.empty(n_omega_o, dtype=object)
        for i, theta_o in enumerate(o_thetas):
            theta_deg = np.degrees(theta_o)
            for j in range(offsets[i], offsets[i + 1]):
                phi_deg = np.degrees(o_phis[j])
                if j == offsets[i] or j % MAX_DIM_PER_SUBPLOT == 0:
                    y_labels[j] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    y_labels[j] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'

        if len(y_labels) != n_omega_o:
            raise ValueError(f"Number of y_labels ({len(y_labels)}) does not match number of omega_o ({n_omega_o})")

    vmin = np.nanmin(maps)
    vmax = np.nanmax(maps)

    print(f"vmin: {vmin}, vmax: {vmax}")
    
    if parallel:
        n_workers = os.cpu_count() * 3 // 4

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(style="bold"),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            futures = [] # keep track of the futures
            with multiprocessing.Manager() as manager:
                status = manager.dict()
                overall = progress.add_task(f"Generating brdf error maps [{n_lambda}] in total")

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # create tasks with chunk size of 16
                    for chunk_idx in range(0, n_lambda, 16):
                        chunk_size = min(16, n_lambda - chunk_idx)
                        tid = progress.add_task(f"  Processing {chunk_idx+1} ~ {chunk_idx+chunk_size}", visible=False)
                        futures.append(executor.submit(task, status, tid, chunk_idx, chunk_size, name, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax))

                    n_to_be_finished = len(futures)

                    # monitor the progress:
                    while (n_finished := sum([future.done() for future in futures])) <= n_to_be_finished:
                        progress.update(overall, completed=n_finished, total=n_to_be_finished)
                        for tid, update_data in status.items():
                            latest = update_data["progress"]
                            total = update_data["total"]
                            progress.update(
                                tid,
                                completed=latest,
                                total=total,
                                visible=latest < total,
                            )
                        if n_finished == n_to_be_finished:
                            break

                    for future in futures:
                        future.result()
    else:
        with Progress(
            TextColumn("[green]Generating error maps...[/green] [progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(style="bold"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            for i in progress.track(range(n_lambda)):
                plot_brdf_error_map_single(i, name, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax) 
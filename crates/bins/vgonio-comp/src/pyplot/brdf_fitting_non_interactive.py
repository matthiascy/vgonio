import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import multiprocessing
import rich
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress warnings to avoid clogging the console
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)

SIZE_TO_FIGSIZE_RATIO = 36 / 64

def errormap(data, xlabels, ylabels, ax=None, enable_cbar=True, cbarkw=None, cbarlabel="", vmin=None, vmax=None, **kwargs):
    """Plot a 2D error map with square pixels.

    Parameters
    ----------
    data : ndarray
        2D numpy array of the error map
    ylabels : list
        Labels for the y-axis
    xlabels : list
        Labels for the x-axis
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
    # imshow expects the data to be in the shape of [y, x]
    im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)

    cbar = None
    if enable_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbarkw)
        cbar.set_label(cbarlabel, rotation=-90, va='bottom')

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=xlabels,
                  rotation=45, ha='left', rotation_mode='anchor', fontfamily='monospace')
    ax.set_yticks(range(data.shape[0]), labels=ylabels, fontfamily='monospace')

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

def plot_brdf_error_map_single(idx, name, figtype, model_name, metric, x_label, y_label, xticks_labels, yticks_labels, maps, spectrum, vmin, vmax, cbarlabel='residual', maxw=MAX_DIM_PER_SUBPLOT, maxh=MAX_DIM_PER_SUBPLOT):
    """
    Plot a single brdf error map.

    Parameters
    ----------
    idx : int
        Index of the wavelength to plot
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    figtype : str
        Type of the figure to plot
    model_name : str
        Name of the model
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    xticks_labels : list
        Labels for the x-axis ticks
    yticks_labels : list
        Labels for the y-axis ticks
    maps : list
        List of maps to plot; each map is a 2D numpy array
    vmin : float, optional
        Minimum value for the colorbar
    vmax : float, optional
        Maximum value for the colorbar
    """
    is_residuals = figtype == 'residuals'
    if is_residuals:
        (_, h, w) = maps.shape # nlambda, nwi, nwo
    else:
        (_, w, h) = maps.shape # 1, nlambda, nomega_i

    model_name = model_name.lower()
 
    if h > maxh or w > maxw:
        n_rows = math.ceil(h / maxh)
        n_cols = math.ceil(w / maxw)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * SIZE_TO_FIGSIZE_RATIO, h * SIZE_TO_FIGSIZE_RATIO)) 
        axes = axes.reshape(n_rows, n_cols)

        for i in range(n_rows):
            for j in range(n_cols):
                x_start = j * maxw
                x_end = min(x_start + maxw, w)
                y_start = i * maxh
                y_end = min(y_start + maxh, h)
                x_labels_sub = xticks_labels[x_start:x_end] 
                y_labels_sub = yticks_labels[y_start:y_end]
                if is_residuals:
                    data = maps[idx, y_start:y_end, x_start:x_end]
                else:
                    data = maps[0, x_start:x_end, y_start:y_end].T
                im, cbar, _ = plot_brdf_error_map_subplot(data, x_label, y_label, x_labels_sub, y_labels_sub, axes[i, j], i * n_cols + j + 1, n_rows * n_cols, enable_annotate=False, enable_cbar=False, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=axes.ravel()[-1], location='right')
        cbar.set_label(cbarlabel, rotation=-90, va='bottom')
    else:
        fig, axes = plt.subplots(figsize=(w * SIZE_TO_FIGSIZE_RATIO, h * SIZE_TO_FIGSIZE_RATIO)) 
        if is_residuals:
            data = maps[idx]
        else:
            data = maps.T

        enable_annotate = True if figtype == 'residuals' else False
        im, cbar, _ = plot_brdf_error_map_subplot(data, x_label, y_label, xticks_labels, yticks_labels, axes, 1, 1, enable_annotate=enable_annotate, enable_cbar=True, cbarlabel=cbarlabel, vmin=vmin, vmax=vmax)
    
    if is_residuals:
        fig.suptitle(fr'{name} $\lambda = {spectrum[idx]:.2f}$ nm') 
        fig.savefig(f"{name}_{figtype}_{model_name}_{spectrum[idx]:.2f}nm.png", bbox_inches='tight')
    else:
        fig.suptitle(fr'{name}')
        fig.savefig(f"{name}_{figtype}_{model_name}.png", bbox_inches='tight')

    plt.close()

def plot_brdf_error_map_subplot(data, x_label, y_label, xticks_labels, yticks_labels, ax, part_idx=1, total_parts=1, enable_annotate=True, enable_cbar=True, cbarlabel='residual', vmin=None, vmax=None, **kwargs):
    """
    Plot a 2D error map with square pixels.
    """
    im, cbar = errormap(data, xticks_labels, yticks_labels, ax=ax, enable_cbar=enable_cbar, cmap='YlGn', cbarlabel=cbarlabel, vmin=vmin, vmax=vmax)  

    texts = None
    if enable_annotate:
        texts = annotate_errormap(im, valfmt='{x:.2f}')

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if total_parts > 1:
        ax.set_title(fr'{part_idx} / {total_parts}')
    
    return im, cbar, texts

def task_plot_rs(progress, tid, base_idx, count, name, model_name, metric, x_label, y_label, xticks_labels, yticks_labels, maps, spectrum, vmin, vmax):
    """
    Task to plot the BRDF residuals in parallel.

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
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    model_name : str
        Name of the model
    metric : str
        Metric to plot
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    xticks_labels : list
        Labels for the x-axis ticks
    yticks_labels : list
        Labels for the y-axis ticks
    maps : list
        List of maps to plot; each map is a 2D numpy array
    """
    for i in range(base_idx, base_idx + count):
        time.sleep(0.05)
        plot_brdf_error_map_single(i, name, 'residuals', model_name, metric, x_label, y_label, xticks_labels, yticks_labels, maps, spectrum, vmin, vmax)
        progress[tid] = { "progress": i - base_idx + 1, "total": count }

def plot_brdf_fitting_residuals(name, residuals, rmaps, is_grid, metric, model_name, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel):
    """
    Plot the brdf fitting residuals.

    Parameters
    ----------
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    residuals : ndarray
        Residuals between the measured data and the model. The shape of residuals could be either 
            - ϑi, ϕi, ϑo, ϕo, λ - grid  
            - ϑi, ϕi, ωo, λ - list
    rmaps : ndarray
        Residuals arranged in a 2D array with dimension: [Nλ, Nωo, Nωi], where Nλ is the number of wavelengths, 
        Nωo is the number of outgoing directions, and Nωi is the number of incident directions.
        In most cases, Nωi is computed by Nωi = Nθi * Nφi, where Nθi and Nφi are the number of incident directions 
        in the theta (inclination) and phi (azimuth) directions. Depends on the measured BRDF data, the outgoing 
        directions could be either a grid or a list; if it is a grid, Nωo = Nθo * Nφo, where Nθo and Nφo are the 
        number of outgoing directions in the theta (inclination) and phi (azimuth) directions. 
        If it is a list, Nωo is the length of the `o_phis` list.
    is_grid : bool
        Whether the outgoing directions are a grid or a list
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
    # 1. plot residual maps per wavelength
    #    each map contains all the incident directions and all the outgoing directions
    if is_grid:
        (n_theta_i, n_phi_i, n_theta_o, n_phi_o, n_lambda) = residuals.shape

        yticks_labels = np.empty(n_theta_i * n_phi_i, dtype=object)
        for ti in range(n_theta_i):
            for pi in range(n_phi_i):
                theta_deg = np.degrees(i_thetas[ti])
                phi_deg = np.degrees(i_phis[pi])
                if pi == 0 or pi % MAX_DIM_PER_SUBPLOT == 0:
                    yticks_labels[ti * n_phi_i + pi] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    yticks_labels[ti * n_phi_i + pi] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'

        xticks_labels = np.empty(n_theta_o * n_phi_o, dtype=object)
        for to in range(n_theta_o):
            for po in range(n_phi_o):
                theta_deg = np.degrees(o_thetas[to])
                phi_deg = np.degrees(o_phis[po])
                if po == 0 or po % MAX_DIM_PER_SUBPLOT == 0:
                    xticks_labels[to * n_phi_o + po] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    xticks_labels[to * n_phi_o + po] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'
    else:
        n_theta_o = len(o_thetas)
        (n_theta_i, n_phi_i, n_omega_o, n_lambda) = residuals.shape

        yticks_labels = np.empty(n_theta_i * n_phi_i, dtype=object)
        for ti in range(n_theta_i):
            for pi in range(n_phi_i):
                theta_deg = np.degrees(i_thetas[ti])
                phi_deg = np.degrees(i_phis[pi])
                if pi == 0 or pi % MAX_DIM_PER_SUBPLOT == 0:
                    yticks_labels[ti * n_phi_i + pi] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    yticks_labels[ti * n_phi_i + pi] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'
        
        xticks_labels = np.empty(n_omega_o, dtype=object)
        for i, theta_o in enumerate(o_thetas):
            theta_deg = np.degrees(theta_o)
            for j in range(offsets[i], offsets[i + 1]):
                phi_deg = np.degrees(o_phis[j])
                if j == offsets[i] or j % MAX_DIM_PER_SUBPLOT == 0:
                    xticks_labels[j] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
                else:
                    xticks_labels[j] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'

        if len(xticks_labels) != n_omega_o:
            raise ValueError(f"Number of y_labels ({len(xticks_labels)}) does not match number of omega_o ({n_omega_o})")

    vmin = np.nanmin(rmaps)
    vmax = np.nanmax(rmaps)
    rich.print(f"vmin: {vmin}, vmax: {vmax}")

    x_label = r'Outgoing Direction $\omega_o = (\theta_o, \phi_o)$'
    y_label = r'Incident Direction $\omega_i = (\theta_i, \phi_i)$'
 
    if parallel:
        parallel_execute(n_lambda, task_plot_rs, name, model_name, metric, x_label, y_label, xticks_labels, yticks_labels, rmaps, spectrum, vmin, vmax)
    else:
        sequential_plot(n_lambda, name, 'residuals', model_name, metric, x_label, y_label, xticks_labels, yticks_labels, rmaps, spectrum, vmin, vmax)

def task_plot_mse(progress, tid, base_idx, count, name, model_name, metric, x_label, y_label, xticks_labels, yticks_labels, maps, spectrum, vmin, vmax, cbarlabel):
    """
    Task to plot the MSE of per incident angle fitting results.

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
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    model_name : str
        Name of the model
    metric : str
        Metric used to compute the residuals
    x_label : str
        Label for the x-axis
    y_label : str
        Label for the y-axis
    xticks_labels : list
        Labels for the x-axis ticks
    yticks_labels : list
        Labels for the y-axis ticks
    maps : list
        List of maps to plot; each map is a 2D numpy array
    spectrum : ndarray
        Wavelengths in nanometers
    vmin : float, optional
        Minimum value for the colorbar
    vmax : float, optional
        Maximum value for the colorbar 
    cbarlabel : str
        Label for the colorbar
    """
    for i in range(base_idx, base_idx + count):
        time.sleep(0.05)
        plot_brdf_error_map_single(i, name, 'mse', model_name, metric, x_label, y_label, xticks_labels, yticks_labels, maps, spectrum, vmin, vmax, cbarlabel)
        progress[tid] = { "progress": i - base_idx + 1, "total": count }

def plot_brdf_fitting_mse_per_incident_angle(name, residuals, mmaps, metric, model_name, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel):
    """
    Plot the MSE of per incident angle fitting results.
    """
    (n_lambda, n_omega_i) = mmaps.shape
    n_phi_i = len(i_phis)
    xticks_labels = np.fromiter((f'{l:.2f} nm' for l in spectrum), dtype=object)

    yticks_labels = np.empty(n_omega_i, dtype=object)
    for i, theta_i in enumerate(i_thetas):
        theta_deg = np.degrees(theta_i)
        for j, phi_i in enumerate(i_phis):
            phi_deg = np.degrees(phi_i)
            if j == 0 or j % MAX_DIM_PER_SUBPLOT == 0:
                yticks_labels[i * n_phi_i + j] = f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°'
            else:
                yticks_labels[i * n_phi_i + j] = f'{" ":>3s}~{" ":>3s} {phi_deg:>6.1f}°'

    vmin = np.nanmin(mmaps)
    vmax = np.nanmax(mmaps)
    rich.print(f"vmin: {vmin}, vmax: {vmax}")

    x_label = r'Wavelength $\lambda$ (nm)'
    y_label = r'Incident Direction $\omega_i = (\theta_i, \phi_i)$'


    mmaps = mmaps.reshape(1, n_lambda, n_omega_i)

    if parallel:
        parallel_execute(1, task_plot_mse, name, model_name, metric, x_label, y_label, xticks_labels, yticks_labels, mmaps, spectrum, vmin, vmax, "MSE")
    else:
        sequential_plot(1, name, 'mse', model_name, metric, x_label, y_label, xticks_labels, yticks_labels, mmaps, spectrum, vmin, vmax, "MSE")

def plot_brdf_fitting_errors(name, data, model_name, metric, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel):
    """
    Plot the BRDF error map. Entry point for the non-interactive BRDF fitting plot in Rust. This function will generate
    two types of plots:

    1. Residuals maps for each wavelength: x-axis is ωo (ϑo, ϕo), y-axis is ωi (ϑi, ϕi), pixel value: residuals
    2. MSE of per incident angle fitting results: x-axis is the wavelength, y-axis is the incident angle, pixel value: MSE

    The computation is done in Rust.

    Parameters
    ----------
    name : str
        Name of the measured BRDF; used as the title of the plot and the filename of the plot
    data : (ndarray, ndarray, ndarray)
        Include three arrays:
        1. residuals: Residuals between the measured data and the model. The shape of residuals could be either 
            - ϑi, ϕi, ϑo, ϕo, λ - grid  
            - ϑi, ϕi, ωo, λ - list
        2. rmaps: Residuals arranged in a 2D array with dimension: [Nλ, Nωo, Nωi], where Nλ is the number of wavelengths, 
            Nωo is the number of outgoing directions, and Nωi is the number of incident directions.
            In most cases, Nωi is computed by Nωi = Nθi * Nφi, where Nθi and Nφi are the number of incident directions 
            in the theta (inclination) and phi (azimuth) directions. Depends on the measured BRDF data, the outgoing 
            directions could be either a grid or a list; if it is a grid, Nωo = Nθo * Nφo, where Nθo and Nφo are the 
            number of outgoing directions in the theta (inclination) and phi (azimuth) directions. 
            If it is a list, Nωo is the length of the `o_phis` list.
        3. mmaps: MSE of per incident angle fitting results arranged in a 2D array with dimension: [Nλ, Nωi], 
            where Nλ is the number of wavelengths, and Nωi is the number of incident directions. 
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
    residuals, rmaps, mmaps = data
    is_grid = not isinstance(offsets, np.ndarray)

    # 1. plot residual maps per wavelength
    #    each map contains all the incident directions and all the outgoing directions
    print('\nPlotting residuals...')
    plot_brdf_fitting_residuals(name, residuals, rmaps, is_grid, metric, model_name, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel)

    # 2. plot MSE of per incident angle fitting results
    #    each map contains the MSE of per incident angle fitting results
    print('\nPlotting MSE...')
    plot_brdf_fitting_mse_per_incident_angle(name, residuals, mmaps, metric, model_name, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel)


def parallel_execute(ntotal, task, *task_args): 
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
            overall = progress.add_task(f"Generating BRDF fitting maps, {ntotal} in total: ")

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # create tasks with chunk size of 16
                for chunk_idx in range(0, ntotal, 16):
                    chunk_size = min(16, ntotal - chunk_idx)
                    tid = progress.add_task(f"  Processing {chunk_idx+1} ~ {chunk_idx+chunk_size}", visible=False)
                    futures.append(executor.submit(task, status, tid, chunk_idx, chunk_size, *task_args))

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

def sequential_plot(ntotal, name, figtype, *args):
    with Progress(
        TextColumn("[green]Generating error maps...[/green] [progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(style="bold"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        for i in progress.track(range(ntotal)):
            plot_brdf_error_map_single(i, name, figtype, *args) 
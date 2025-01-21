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

def errormap(data, rowlabels, collabels, ax=None, cbarkw=None, cbarlabel="", vmin=None, vmax=None, **kwargs):
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

    # Add colorbar
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

def plot_brdf_error_map_single(i, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax):
    """
    Plot a single brdf error map.

    Parameters
    ----------
    i : int
        Index of the wavelength to plot
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
    fig, ax = plt.subplots(figsize=(12, 12))    
 
    im, cbar = errormap(maps[i], x_labels, y_labels, ax=ax, cmap='YlGn', cbarlabel='residual', vmin=vmin, vmax=vmax)  
    texts = annotate_errormap(im, valfmt='{x:.2f}')

    ax.set_xlabel(r'Incident Direction $\omega_i = (\theta_i, \phi_i)$')
    ax.set_ylabel(r'Outgoing Direction $\omega_o = (\theta_o, \phi_o)$')
    ax.set_title(fr'{model_name} {metric} $\lambda = {spectrum[i]:.2f}$ nm')
    
    # # Calculate and display error statistics
    # valid_errors = maps[i][~np.isnan(maps[i])]
    # mean_err = np.mean(valid_errors)
    # std_err = np.std(valid_errors)
    # max_err = np.max(valid_errors)
    # min_err = np.min(valid_errors)
    
    # stats_text = f'Mean: {mean_err:.3f}\nStd: {std_err:.3f}\nMax: {max_err:.3f}\nMin: {min_err:.3f}'
    # plt.figtext(1.15, 0.7, stats_text, fontsize=8)
    
    # Adjust layout to prevent text overlap
    fig.tight_layout() 
    fig.savefig(f"{model_name}_{metric}_{spectrum[i]:.2f}nm.png", 
                bbox_inches='tight')
    plt.close()

def task(progress, tid, base_idx, count, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax):
    for i in range(base_idx, base_idx + count):
        time.sleep(0.05)
        plot_brdf_error_map_single(i, model_name + str(i), metric, x_labels, y_labels, maps, spectrum, vmin, vmax)
        progress[tid] = { "progress": i - base_idx + 1, "total": count }

def plot_brdf_error_map(metric, model_name, residuals, maps, i_thetas, i_phis, o_thetas, o_phis, offsets, spectrum, parallel):
    # Check if outgoing directions offsets exists
    is_grid = not isinstance(offsets, np.ndarray)

    # the shape of residuals could be either 
    # ϑi, ϕi, ϑo, ϕo, λ - grid  
    # or ϑi, ϕi, ωo, λ - list

    # 1. plot residual maps per wavelength
    #    each map contains all the incident directions and all the outgoing directions
    if is_grid:
        (n_theta_i, n_phi_i, n_theta_o, n_phi_o, n_lambda) = residuals.shape

        x_labels = []
        for ti in range(n_theta_i):
            for pi in range(n_phi_i):
                theta_deg = np.degrees(i_thetas[ti])
                phi_deg = np.degrees(i_phis[pi])
                x_labels.append(f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°')

        y_labels = []
        for to in range(n_theta_o):
            for po in range(n_phi_o):
                theta_deg = np.degrees(o_thetas[to])
                phi_deg = np.degrees(o_phis[po])
                y_labels.append(f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°')
    else:
        n_theta_o = len(o_thetas)
        (n_theta_i, n_phi_i, n_omega_o, n_lambda) = residuals.shape

        x_labels = []
        for ti in range(n_theta_i):
            for pi in range(n_phi_i):
                theta_deg = np.degrees(i_thetas[ti])
                phi_deg = np.degrees(i_phis[pi])
                x_labels.append(f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°')
        
        y_labels = []
        for i, theta_o in enumerate(o_thetas):
            theta_deg = np.degrees(theta_o)
            for j in range(offsets[i], offsets[i + 1]):
                phi_deg = np.degrees(o_phis[j])
                y_labels.append(f'{theta_deg:>6.1f}° {phi_deg:>6.1f}°')

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
                        futures.append(executor.submit(task, status, tid, chunk_idx, chunk_size, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax))

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
                plot_brdf_error_map_single(i, model_name, metric, x_labels, y_labels, maps, spectrum, vmin, vmax) 
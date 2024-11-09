from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.tri import Triangulation
from matplotlib.widgets import Button, TextBox
import numpy as np
import seaborn as sns

# Use this to avoid GUI
# mpl.use('Agg')

# mpl.use('qtagg')

# Paired/tab10

sns.set_theme(style="whitegrid", color_codes=True)


def plot_err(xstart, xend, xstep, errs):
    # Plot the error, with x-axis from xstart to xend (inclusive) with xstep.
    xs = np.arange(xstart, xend + xstep, xstep)
    x_min = np.argmin(errs)
    plt.plot(xs, errs)
    plt.annotate(f"Min: {xs[x_min]:.4f}", (xs[x_min], errs[x_min]))
    plt.show()


def format_angle_pair(t):
    return f"θ: {np.degrees(t[0]):>4.2f}, φ: {np.degrees(t[1]):>6.2f}"


def plot_brdf_comparison(
        n_wi,
        dense,
        n_wo_itrp,
        wi_wo_pairs_itrp,
        n_wo_olaf,
        wi_wo_pairs_olaf,
        brdf_itrp,
        wavelengths_itrp,
        brdf_max_itrp,
        brdf_olaf,
        wavelengths_olaf,
        brdf_max_olaf,
):
    n_wavelengths_itrp = len(wavelengths_itrp)
    n_wavelengths_olaf = len(wavelengths_olaf)
    # print(
    #     f"n_wi: {n_wi}, n_wo_itrp: {n_wo_itrp}, n_wo_olaf: {n_wo_olaf}, n_wavelengths_itrp: {n_wavelengths_itrp}, n_wavelengths_olaf: {n_wavelengths_olaf}"
    # )
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("BRDF")
    fig.set_figwidth(13)
    fig.set_figheight(6)

    # for i, ((wi_theta, wi_phi), wos) in enumerate(wi_wo_pairs):
    #     print(f"{i} -- θi: {np.degrees(wi_theta):>6.2f}, φi: {np.degrees(wi_phi):>6.2f}")
    #     for j, (wo_theta, wo_phi) in enumerate(wos):
    #         print(f"    {j}  θo: {np.degrees(wo_theta):>6.2f}, φo: {np.degrees(wo_phi):>6.2f}")

    ax_olaf = ax[0]
    ax_comp = ax[1]
    ax_itrp = ax[2]

    # print(f"n_wo_olaf: {n_wo_olaf}, n_wo_itrp: {n_wo_itrp}")

    # from -80 to 80, with step 10
    xs_olaf = np.arange(-n_wo_olaf / 2 * 10, (1 + n_wo_olaf / 2) * 10, 10)
    xs_itrp = (
        np.arange(-n_wo_itrp / 2 * 2.5, (1 + n_wo_itrp / 2) * 2.5, 2.5)
        if dense
        else xs_olaf
    )
    # print(f"len: {len(xs_itrp)}, xs_itrp: {xs_itrp}")
    # print(f"len: {len(xs_olaf)}, xs_olaf: {xs_olaf}")

    # rearrange the data to match the xs, NaN for data that is not measured
    # input data is in the form of [λ, ωi, ωo]
    # we want to merge ωo into a single dimension (in-plane brdf)
    olaf_arranged = np.full((n_wavelengths_olaf, n_wi, (n_wo_olaf + 1)), np.nan)
    itrp_arranged = np.full((n_wavelengths_itrp, n_wi, (n_wo_itrp + 1)), np.nan)
    for wi_idx, ((_wi_theta, _wi_phi), wos) in enumerate(wi_wo_pairs_olaf):
        # print(f"wi idx: {wi_idx}, θi: {np.degrees(wi_theta):>6.2f}, φi: {np.degrees(wi_phi):>6.2f}")
        for wo_idx_org, (wo_theta, wo_phi) in enumerate(wos):
            diff = (
                np.abs(xs_olaf - np.degrees(wo_theta))
                if wo_phi < np.pi * 0.5
                else np.abs(xs_olaf + np.degrees(wo_theta))
            )
            wo_idx = np.argmin(diff)
            for k in np.arange(0, n_wavelengths_olaf):
                olaf_arranged[k, wi_idx, wo_idx] = brdf_olaf[
                    k * n_wi * n_wo_olaf + wi_idx * n_wo_olaf + wo_idx_org
                    ]

    for wi_idx, ((wi_theta, wi_phi), wos) in enumerate(wi_wo_pairs_itrp):
        # print(f"wi idx: {wi_idx}, θi: {np.degrees(wi_theta):>6.2f}, φi: {np.degrees(wi_phi):>6.2f}")
        for wo_idx_org, (wo_theta, wo_phi) in enumerate(wos):
            diff = (
                np.abs(xs_itrp - np.degrees(wo_theta))
                if wo_phi < np.pi * 0.5
                else np.abs(xs_itrp + np.degrees(wo_theta))
            )
            wo_idx = np.argmin(diff)
            for k in range(n_wavelengths_itrp):
                itrp_arranged[k, wi_idx, wo_idx] = brdf_itrp[
                    k * n_wi * n_wo_itrp + wi_idx * n_wo_itrp + wo_idx_org
                    ]

    cur_wi_idx = 0
    olaf_cur_lambda_idx = 0
    itrp_cur_lambda_idx = 0

    olaf_curve = ax_olaf.plot(
        xs_olaf,
        olaf_arranged[olaf_cur_lambda_idx, cur_wi_idx, :],
        "o-",
        label="Measured BRDF(Olaf)",
    )
    itrp_curve = ax_itrp.plot(
        xs_itrp,
        itrp_arranged[itrp_cur_lambda_idx, cur_wi_idx, :],
        "o-",
        label="Interpolated BRDF",
    )
    olaf_curve_comp = ax_comp.plot(
        xs_olaf,
        olaf_arranged[olaf_cur_lambda_idx, cur_wi_idx, :],
        "o-",
        label="Measured BRDF(Olaf)",
    )
    itrp_curve_comp = ax_comp.plot(
        xs_itrp,
        itrp_arranged[itrp_cur_lambda_idx, cur_wi_idx, :],
        "o-",
        label="Interpolated BRDF",
    )

    for a in ax:
        a.legend()

    olaf_cur_lambda_text = TextBox(
        plt.axes((0.15, 0.025, 0.06, 0.04)),
        "λ",
        initial=f"{wavelengths_olaf[olaf_cur_lambda_idx]:.4f}",
    )
    button_prev_olaf_lambda = Button(
        plt.axes((0.22, 0.025, 0.06, 0.04)),
        "Prev λ",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )
    button_next_olaf_lambda = Button(
        plt.axes((0.29, 0.025, 0.06, 0.04)),
        "Next λ",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )

    compare_button = Button(
        plt.axes((0.38, 0.025, 0.06, 0.04)),
        "Comp.",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )
    compare_norm_button = Button(
        plt.axes((0.45, 0.025, 0.06, 0.04)),
        "(Comp.)",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )
    prev_wi_button = Button(
        plt.axes((0.53, 0.025, 0.06, 0.04)),
        "Prev ωi",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )
    next_wi_button = Button(
        plt.axes((0.60, 0.025, 0.06, 0.04)),
        "Next ωi",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )
    cur_wi_text = TextBox(
        plt.axes((0.46, 0.90, 0.11, 0.04)),
        "ωi: ",
        initial=f"{format_angle_pair(wi_wo_pairs_itrp[cur_wi_idx][0])}",
    )

    itrp_cur_lambda_text = TextBox(
        plt.axes((0.75, 0.025, 0.06, 0.04)),
        "λ",
        initial=f"{wavelengths_itrp[itrp_cur_lambda_idx]:.4f}",
    )
    button_next_itrp_lambda = Button(
        plt.axes((0.82, 0.025, 0.06, 0.04)),
        "Next λ",
        color="lightgoldenrodyellow",
        hovercolor="0.975",
    )

    def update(olaf_lambda_idx, olaf_wi_idx, itrp_lambda_idx, itrp_wi_idx):
        olaf_curve[0].set_ydata(olaf_arranged[olaf_lambda_idx, olaf_wi_idx, :])
        olaf_cur_lambda_text.set_val(f"{wavelengths_olaf[olaf_lambda_idx]:.4f}")
        ax_olaf.relim()
        ax_olaf.autoscale_view()
        itrp_curve[0].set_ydata(itrp_arranged[itrp_lambda_idx, itrp_wi_idx, :])
        itrp_cur_lambda_text.set_val(f"{wavelengths_itrp[itrp_lambda_idx]:.4f}")
        ax_itrp.relim()
        ax_itrp.autoscale_view()
        fig.canvas.draw_idle()

    def update_comp(
            olaf_lambda_idx, olaf_wi_idx, itrp_lambda_idx, itrp_wi_idx, normalize=False
    ):
        if normalize:
            max_measured = brdf_max_olaf[
                olaf_wi_idx * n_wavelengths_olaf + olaf_lambda_idx
                ]
            olaf_curve_comp[0].set_ydata(
                olaf_arranged[olaf_lambda_idx, olaf_wi_idx, :] / max_measured
            )
            max_interpolated = brdf_max_itrp[
                itrp_wi_idx * n_wavelengths_itrp + itrp_lambda_idx
                ]
            itrp_curve_comp[0].set_ydata(
                itrp_arranged[itrp_lambda_idx, itrp_wi_idx, :] / max_interpolated
            )
        else:
            olaf_curve_comp[0].set_ydata(olaf_arranged[olaf_lambda_idx, olaf_wi_idx, :])
            itrp_curve_comp[0].set_ydata(itrp_arranged[itrp_lambda_idx, itrp_wi_idx, :])
        ax_comp.relim()
        ax_comp.autoscale_view()
        fig.canvas.draw_idle()

    def on_btn_next_olaf_lambda(event):
        nonlocal olaf_cur_lambda_idx
        olaf_cur_lambda_idx = (olaf_cur_lambda_idx + 1) % n_wavelengths_olaf
        update(olaf_cur_lambda_idx, cur_wi_idx, olaf_cur_lambda_idx, cur_wi_idx)

    def on_btn_prev_olaf_lambda(event):
        nonlocal olaf_cur_lambda_idx
        olaf_cur_lambda_idx = (olaf_cur_lambda_idx - 1) % n_wavelengths_olaf
        update(olaf_cur_lambda_idx, cur_wi_idx, olaf_cur_lambda_idx, cur_wi_idx)

    def on_btn_next_itrp_lambda(event):
        nonlocal itrp_cur_lambda_idx
        itrp_cur_lambda_idx = (itrp_cur_lambda_idx + 1) % n_wavelengths_itrp
        update(olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx)

    def on_compare(event):
        update_comp(
            olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx, False
        )

    def on_compare_norm(event):
        update_comp(
            olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx, True
        )

    def on_btn_prev_wi(event):
        nonlocal cur_wi_idx
        cur_wi_idx = (cur_wi_idx - 1) % n_wi
        cur_wi_text.set_val(f"{format_angle_pair(wi_wo_pairs_itrp[cur_wi_idx][0])}")
        update(olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx)

    def on_btn_next_wi(event):
        nonlocal cur_wi_idx
        cur_wi_idx = (cur_wi_idx + 1) % n_wi
        cur_wi_text.set_val(f"{format_angle_pair(wi_wo_pairs_itrp[cur_wi_idx][0])}")
        update(olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx)

    button_next_olaf_lambda.on_clicked(on_btn_next_olaf_lambda)
    button_prev_olaf_lambda.on_clicked(on_btn_prev_olaf_lambda)
    button_next_itrp_lambda.on_clicked(on_btn_next_itrp_lambda)
    compare_button.on_clicked(on_compare)
    compare_norm_button.on_clicked(on_compare_norm)
    prev_wi_button.on_clicked(on_btn_prev_wi)
    next_wi_button.on_clicked(on_btn_next_wi)
    plt.show()


def new_polar_brdf_plot():
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    configure_polar_brdf_plot(ax)
    return fig, ax


def configure_polar_brdf_plot(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_thetagrids(range(-90, 91, 15))
    # Custom grid
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    # Hide spines
    ax.spines['polar'].set_visible(False)


def custom_polar_brdf_plot_radial_ticks(ax, ymax):
    # Custom radial ticks and labels
    delta = np.round(ymax * 1.2, 0) / 4
    rticks = np.arange(0, ymax + delta, delta)
    ax.set_rticks(rticks)
    ax.set_rlabel_position(-22.5)  # Position of the radial labels


def add_polar_brdf_plot_phi_o_text(ax, ymax, phi_o_deg, phi_o_deg_opp):
    ax.text(np.radians(-85), ymax * 0.95, fr'$\phi_o={int(round(phi_o_deg_opp))}\degree$',
            horizontalalignment='center',
            verticalalignment='center', fontsize=14)
    ax.text(np.radians(85), ymax, fr'$\phi_o={int(round(phi_o_deg))}\degree$',
            horizontalalignment='center',
            verticalalignment='center', fontsize=14)


def plot_brdf_slice(phi_o_deg, phi_o_deg_opp, brdf_slices: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
                    legend=False, cmap='tab10', scale=1.0, use_log=False):
    sns.set_theme(style="whitegrid", color_codes=True)

    fig_polar, ax_polar = new_polar_brdf_plot()

    cm = plt.get_cmap(cmap)

    ymax = 0
    for i, (slice_phi_o, slice_phi_o_opp, theta, wavelengths, label) in enumerate(brdf_slices):
        xs = np.append(np.flip(-np.radians(theta)), np.radians(theta))
        if use_log:
            for l in range(1):
                ys = np.log10(np.append(np.flip(slice_phi_o_opp[:, l]), slice_phi_o[:, l]) * scale)
                ax_polar.plot(xs, ys, label=f"{label}", linewidth=1.8, color=cm(i))
                # λ = {wavelengths[l]:.0f} nm
                ymax = max(ymax, np.max(ys))
        else:
            for l in range(1):
                ys = np.append(np.flip(slice_phi_o_opp[:, l]), slice_phi_o[:, l]) * scale
                ax_polar.plot(xs, ys, label=f"{label}", linewidth=1.8, color=cm(i))
                # λ = {wavelengths[l]:.0f} nm
                ymax = max(ymax, np.max(ys))

    add_polar_brdf_plot_phi_o_text(ax_polar, ymax, phi_o_deg, phi_o_deg_opp)
    custom_polar_brdf_plot_radial_ticks(ax_polar, ymax)

    if legend:
        ax_polar.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('./brdf_slice.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_brdf_slice_in_plane(phi_deg, phi_opp_deg, slices):
    fig_polar, ax_polar = new_polar_brdf_plot()
    ax_polar.set_rlabel_position(0)

    ymax = 0
    for slices_phi, slices_phi_opp, theta_i, theta_o, wavelengths in slices:
        n_spectrum = len(wavelengths)
        xs = np.append(np.flip(-np.radians(np.array(theta_o))), np.radians(np.array(theta_o)))
        for i, (slice_phi, slice_phi_opp, ti) in enumerate(zip(slices_phi, slices_phi_opp, theta_i)):
            if i % 3 == 0:
                slice_phi_o = np.array(slice_phi).reshape((-1, n_spectrum))
                slice_phi_o_opp = np.array(slice_phi_opp).reshape((-1, n_spectrum))
                for l in range(1):
                    ys = np.append(np.flip(slice_phi_o_opp[:, l]), slice_phi_o[:, l])
                    ax_polar.plot(xs, ys, label=fr"$θ_i = {ti:>3.0f}\;\degree$")
                    # , λ = {wavelengths[l]:>3.0f}\;nm
                    ymax = max(ymax, np.max(ys))

    add_polar_brdf_plot_phi_o_text(ax_polar, ymax, phi_deg, phi_opp_deg)
    custom_polar_brdf_plot_radial_ticks(ax_polar, ymax)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()


def calculate_brdf_surface(theta, phi, vals):
    n_theta = len(theta)
    n_phi = len(phi)
    print(f"n_theta: {n_theta}, n_phi: {n_phi}")

    t, p = np.meshgrid(theta, phi)
    x = np.zeros((n_theta, n_phi))
    y = np.zeros((n_theta, n_phi))
    z = np.zeros((n_theta, n_phi))

    for i in range(n_phi):
        for j in range(n_theta):
            r = vals[j, i]
            x[j, i] = r * np.sin(t[i, j]) * np.cos(p[i, j])
            y[j, i] = r * np.sin(t[i, j]) * np.sin(p[i, j])
            z[j, i] = r * np.cos(t[i, j])

    return x, y, z


def plot_brdf_polar_grid(ax):
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d
    r = 0.5
    # Add polar grid on the ground plane (z=0)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)  # Angles for lines

    # Plot concentric circles using Circle class
    circle = Circle((0, 0), r, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5, fill=False)
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0)

    # Plot radial lines
    for angle in angles:
        x_line = [0, np.cos(angle) * r]
        y_line = [0, np.sin(angle) * r]
        z_line = [0, 0]
        ax.plot(x_line, y_line, z_line, color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)
        x_text = r * 1.1 * np.cos(angle)
        y_text = r * 1.1 * np.sin(angle)
        ax.text(x_text, y_text, 0, f'{np.degrees(angle):.0f}°', color='black', fontsize=14, ha='center', va='center',
                alpha=0.5)


def plot_brdf_3d(theta_i, phi_i, theta, phi, vals, cmap='viridis', scale=1.0):
    vals = vals * scale
    x, y, z = calculate_brdf_surface(theta, phi, vals)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_brdf_polar_grid(ax)

    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    cmap=cmap, alpha=0.8,
                    linewidth=0.2,
                    edgecolor='none')

    theta_i_deg = round(np.degrees(theta_i), 0)
    phi_i_deg = round(np.degrees(phi_i), 0)
    ax.view_init(elev=25, azim=phi_i_deg - 25)
    ax.dist = 20

    # val_max = np.max(vals)
    # incident_x = np.sin(theta_i) * np.cos(phi_i) * val_max * 0.6
    # incident_y = np.sin(theta_i) * np.sin(phi_i) * val_max * 0.6
    # incident_z = np.cos(theta_i) * val_max * 0.6
    # Draw incident light using arrow
    # ax.quiver(0, 0, 0, incident_x, incident_y, incident_z, color='k', arrow_length_ratio=0.1, linewidth=2)
    # ax.text(incident_x, incident_y, incident_z * 1.2,
    #         fr'$\theta_i={theta_i_deg}\degree, \phi_i={phi_i_deg}\degree$',
    #         color='k',
    #         fontsize=16)
    # # Draw line from the incident light to the ground plane
    # ax.plot([incident_x, incident_x], [incident_y, incident_y], [0, incident_z], color='k', linestyle='dashed',
    #         linewidth=0.5, alpha=0.5)
    # # Annotate the phi_i angle
    # ax.text(incident_x, incident_y / 4, 0, fr'$\phi_i={phi_i_deg}\degree$', color='k', fontsize=14)
    # # Draw the Z-axis
    # ax.quiver(0, 0, 0, 0, 0, val_max, color='k', arrow_length_ratio=0.1)

    # zlim = np.max(z)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(0, zlim)
    # hide gridlines
    ax.grid(False)
    # hide y and z plane
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # hide x and z plane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # hide axis line
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # # Add color bar which maps values to colors
    # mappable = plt.cm.ScalarMappable(cmap=cmap)
    # mappable.set_array(vals)
    # plt.colorbar(mappable, ax=ax, shrink=0.2, aspect=10)

    plt.tight_layout()
    fig.savefig(f'./brdf_3d_θ{theta_i_deg}φ{phi_i_deg}.pdf', format='pdf', bbox_inches='tight')
    plt.show()


linestyles = ['solid', 'dashed', 'dashdot', 'dotted']


def plot_ndf_slice(phi, phi_opp, ndf_slices: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], ylim):
    # Angles are in radians
    print(f"Plotting NDF slice with wm = ({np.degrees(phi)}, {np.degrees(phi_opp)})")
    deg_ticks = np.arange(-90, 91, 30)
    rad_ticks = np.radians(deg_ticks)
    # each slice is a tuple of (slice, slice_opp, theta) of one measurement
    sns.set_theme(style="whitegrid", color_codes=True)

    figsize = (8, 8) if len(ndf_slices) == 1 else (8, 6)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_aspect('auto')
    ax.set_xlabel(r"$θ_m$", fontsize=18)
    ax.set_ylabel(r"$NDF\;[sr^{-1}]$", fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels([f"{int(deg)}°" for deg in deg_ticks])

    for i, (label, theta, slice, slice_opp) in enumerate(ndf_slices):
        # Combine theta and its filpped negative counterpart for x-axis
        xs = np.append(np.flip(-np.array(theta)), np.array(theta))
        ys = np.append(np.flip(slice_opp), slice)
        if len(ndf_slices) > 1:
            ax.plot(xs, ys, linestyle=linestyles[i], linewidth=2, label=label)
        else:
            ax.plot(xs, ys, color='b', linestyle='-', linewidth=2)

        if i == 0:
            # Annotation
            ax.annotate(fr'$\phi_m={np.degrees(phi_opp):.0f}\degree$', xy=(xs[0], ys[0]), xycoords='data',
                        xytext=(-10, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"),
                        fontsize=14, color='k', fontweight='bold')
            ax.annotate(fr'$\phi_m={np.degrees(phi):.0f}\degree$', xy=(xs[-1], ys[-1]), xycoords='data',
                        xytext=(-40, 20), textcoords='offset points',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.5"),
                        fontsize=14, color='k', fontweight='bold')

    if ylim is not None:
        ax.set_ylim(0, ylim)

    if len(ndf_slices) > 1:
        ax.legend()

    plt.tight_layout()
    # save as pdf
    plt.savefig('./ndf_slice.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def plot_gaf_slice(tm, pm, pv, pv_opp, gaf_slices: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], save=None):
    print(f"Plotting GAF slice with wm = ({np.degrees(tm)}, {np.degrees(pm)}) at pv = {np.degrees(pv)}")
    deg_ticks = np.arange(-90, 91, 30)
    rad_ticks = np.radians(deg_ticks)

    multi = len(gaf_slices) > 1

    figsize = (8, 8) if multi else (8, 6)
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_aspect('auto')
    ax.set_xlabel(r"$θ_v$", fontsize=18)
    ax.set_ylabel(r"$GAF$", fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels([f"{int(deg)}°" for deg in deg_ticks])

    for i, (label, theta, slice, slice_opp) in enumerate(gaf_slices):
        # Combine theta and its filpped negative counterpart for x-axis
        xs = np.append(np.flip(-theta), theta)
        ys = np.append(np.flip(slice_opp), slice)

        if multi:
            ax.plot(xs, ys, linestyle=linestyles[i], linewidth=1.6, label=label)
        else:
            ax.plot(xs, ys, color='b', linestyle='-', linewidth=2)

        if i == 0:
            # Annotation
            ax.text(-1.5, 0.05, fr'$\phi_v={np.degrees(pv_opp):.0f}\degree$', fontsize=20, color='k', fontweight='bold')
            ax.text(1.0, 0.05, fr'$\phi_v={np.degrees(pv):.0f}\degree$', fontsize=20, color='k', fontweight='bold')

    if multi:
        ax.legend()

    plt.tight_layout()
    if save is not None:
        if save.endswith('.pdf'):
            plt.savefig(save, format='pdf', bbox_inches='tight')
        elif save.endswith('.png'):
            plt.savefig(save, format='png', bbox_inches='tight', dpi=100)
    else:
        plt.show()


def plot_brdf_map(images: list[str, Tuple[int, int], np.ndarray], cmap='BuPu', cbar=False, coord=False, diff=False,
                  fc='black', pstep=45, tstep=30, save=None):
    print("params: ", cmap, cbar, coord, diff, fc, pstep, tstep)
    from tone_mapping import tone_mapping

    if diff:
        for i in range(0, len(images), 2):
            (name1, size1, pixels1) = images[i]
            (name2, size2, pixels2) = images[i + 1]
            diff_pixels = np.abs(pixels1 - pixels2)
            mse = np.mean(np.square(diff_pixels))
            fig, ax = tone_mapping(diff_pixels, size1, cmap=cmap, cbar=cbar, coord=coord, cbar_label='Difference',
                                   color=fc, pstep=pstep, tstep=tstep, ac='o')
            ax.text(-size1[0] / 2 + 50, size1[1] / 2 - 10, f'MSE: {mse:.4f}', color=fc, fontsize=14, ha='center',
                    va='center', alpha=0.8)
            if save is not None:
                if save.endswith('.pdf'):
                    fig.savefig(f'{name1}_{name2}_diff.pdf', format='pdf', bbox_inches='tight')
                elif save.endswith('.png'):
                    fig.savefig(f'{name1}_{name2}_diff.png', format='png', bbox_inches='tight', dpi=100)
            plt.show()
    else:
        for i, (name, size, pixels) in enumerate(images):
            fig, ax = tone_mapping(pixels, size, cmap=cmap, cbar=cbar, coord=coord,
                                   cbar_label=r'BRDF [$\mathrm{sr^{-1}}$]', color=fc, pstep=pstep, tstep=tstep, ac='o')
            if save is not None:
                if save.endswith('.pdf'):
                    fig.savefig(f'{name}.pdf', format='pdf', bbox_inches='tight')
                elif save.endswith('.png'):
                    fig.savefig(f'{name}.png', format='png', bbox_inches='tight', dpi=100)
            plt.show()


def downsample_surface(xx, yy, surface, factor):
    xx = xx[::factor, ::factor]
    yy = yy[::factor, ::factor]
    return surface[::factor, ::factor]


def plot_surfaces(surfaces, cmap, ds_factor=4):
    for dv, du, surface, name in surfaces:
        print(f"Plotting surface: {surface.shape}, du: {du}, dv: {dv}")
        cols = surface.shape[1]
        rows = surface.shape[0]
        w = cols * du
        h = rows * dv

        z_downsampled = surface[::ds_factor, ::ds_factor]
        cols_downsampled = int(cols / ds_factor)
        rows_downsampled = int(rows / ds_factor)

        print("Downsampled shape: ", z_downsampled.shape)
        print("Downsampled cols: ", cols_downsampled)
        print("Downsampled rows: ", rows_downsampled)

        x = np.linspace(-w / 2, w / 2, cols_downsampled)
        y = np.linspace(-h / 2, h / 2, rows_downsampled)
        print(f"X: {x.shape}, Y: {y.shape}")
        xx, yy = np.meshgrid(x, y)

        # Compute the min and max of the surface
        min_val = np.min(surface)
        max_val = np.max(surface)
        range = max_val - min_val
        if range == 0:
            range = 1

        z_downsampled = (z_downsampled - min_val) / range / max(w, h)

        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = z_downsampled.flatten()

        tris = Triangulation(x_flat, y_flat)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        print("z max: ", np.max(z_downsampled))
        max_z = np.max(z_downsampled)
        zlim = 0.01 if max_z == 0.0 else max_z * 10.0
        ax.set_zlim(0, zlim)

        # plot surface
        # ax.plot_trisurf(x_flat, y_flat, z_flat, triangles=tris.triangles, cmap=cmap, edgecolor='none', linewidth=0,
        #                 alpha=0.8)

        ax.plot_surface(xx, yy, z_downsampled, color='gray', alpha=0.8, edgecolor='none', linewidth=0)

        # hide gridlines
        ax.grid(False)
        # hide y and z plane
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # hide x and z plane
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # hide axis line
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.xaxis.line.set_color("white")
        ax.yaxis.line.set_color("white")
        ax.zaxis.line.set_color("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=50)

        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        fig.savefig(f'{name}_plot.png', format='png', bbox_inches='tight', dpi=100)
        # plt.show()


def plot_brdf_fitting(samples: np.ndarray, theta_i: np.ndarray, phi_i: np.ndarray, theta_o: np.ndarray,
                      wavelengths: np.ndarray, fitted_bk: np.ndarray, fitted_tr: np.ndarray):
    from matplotlib.widgets import Slider, CheckButtons

    # Two subplots, left is the 3D BRDF, right is the 2D BRDF slice
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.3, top=0.85, right=0.8)

    # the dimension of the samples is (n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_wavelengths)
    (n_theta_i, n_phi_i, n_phi_o, n_theta_o, n_wavelengths) = samples.shape
    n_slice = n_phi_o // 2
    xs = np.append(np.flip(-np.degrees(theta_o)), np.degrees(theta_o))

    # initial values (index) for theta_i, phi_i, phi_o
    init_theta_i = 0
    init_phi_i = 0
    init_wavelength = 0
    is_polar = False

    # Add sliders for theta_i, phi_i, and wavelength
    ax_theta_i = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    ax_phi_i = plt.axes([0.2, 0.15, 0.6, 0.03], facecolor='lightgoldenrodyellow')
    ax_wavelength = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')

    # Use index-based sliders but map them to actual theta_i and phi_i values
    theta_i_slider = Slider(ax_theta_i, 'θi', 0, n_theta_i - 1, valinit=init_theta_i, valstep=1)
    phi_i_slider = Slider(ax_phi_i, 'φi', 0, n_phi_i - 1, valinit=init_phi_i, valstep=1)
    wavelength_slider = Slider(ax_wavelength, 'λ', 0, n_wavelengths - 1, valinit=init_wavelength, valstep=1)

    wi_text = fig.text(0.5, 0.92,
                       f"θi: {np.degrees(theta_i[init_theta_i]):.2f}°, φi: {np.degrees(phi_i[init_phi_i]):.2f}°, λ: {wavelengths[init_wavelength]:.0f} nm",
                       ha='center', va='center', fontsize=14, fontweight='bold')

    def toggle_polar(label):
        nonlocal is_polar
        is_polar = not is_polar
        update(0)

    # update the plot based on selected values
    def update(val):
        ax[1].cla()

        ti = int(theta_i_slider.val)
        pi = int(phi_i_slider.val)
        l = int(wavelength_slider.val)

        theta_i_val = np.degrees(theta_i[ti])
        phi_i_val = np.degrees(phi_i[pi])
        lambda_val = wavelengths[l]

        theta_i_slider.valtext.set_text(f"{theta_i_val:.2f}°")
        phi_i_slider.valtext.set_text(f"{phi_i_val:.2f}°")
        wavelength_slider.valtext.set_text(f"{lambda_val:.0f} nm")
        wi_text.set_text(f"θi: {theta_i_val:.2f}°, φi: {phi_i_val:.2f}°, λ: {lambda_val:.0f} nm")

        for s in range(n_slice):
            s_opp = (s + n_slice) % n_phi_o
            samples_phi = samples[ti, pi, s, :, l]
            samples_phi_opp = samples[ti, pi, s_opp, :, l]
            samples_phi_bk = fitted_bk[ti, pi, s, :, l]
            samples_phi_bk_opp = fitted_bk[ti, pi, s_opp, :, l]
            samples_phi_tr = fitted_tr[ti, pi, s, :, l]
            samples_phi_tr_opp = fitted_tr[ti, pi, s_opp, :, l]
            ys = np.append(np.flip(samples_phi_opp), samples_phi)
            ys_bk = np.append(np.flip(samples_phi_bk_opp), samples_phi_bk)
            ys_tr = np.append(np.flip(samples_phi_tr_opp), samples_phi_tr)
            ax[1].plot(xs, ys, label=f"φo={s}'{s_opp}")
            ax[1].plot(xs, ys_bk, label=f"bk φo={s}'{s_opp}")
            ax[1].plot(xs, ys_tr, label=f"tr φo={s}'{s_opp}")

        ax[1].legend(loc="upper right")
        fig.canvas.draw_idle()

    # Update plot on slider change
    theta_i_slider.on_changed(update)
    phi_i_slider.on_changed(update)
    wavelength_slider.on_changed(update)

    update(0)
    plt.show()

    # for ti in range(n_theta_i):
    #     for pi in range(n_phi_i):
    #         for s in range(n_slice):
    #             s_opp = (s + n_slice) % n_phi_o
    #             # for l in range(n_wavelengths):
    #             l = 0
    #             samples_phi = samples[ti, pi, s, :, l]
    #             samples_phi_opp = samples[ti, pi, s_opp, :, l]
    #             ys = np.append(np.flip(samples_phi_opp), samples_phi)
    #             ax[1].plot(xs, ys, label=f"θi={ti}, φi={pi}, φo={s}'{s_opp}, λ={l}")

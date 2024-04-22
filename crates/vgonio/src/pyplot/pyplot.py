import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np

# Use this to avoid GUI
# mpl.use('Agg')
mpl.use('qtagg')


def plot_err(xstart, xend, xstep, errs):
    # Plot the error, with x-axis from xstart to xend (inclusive) with xstep.
    xs = np.arange(xstart, xend + xstep, xstep)
    x_min = np.argmin(errs)
    plt.plot(xs, errs)
    plt.annotate(f"Min: {xs[x_min]:.4f}", (xs[x_min], errs[x_min]))
    plt.show()


def format_angle_pair(t):
    return f"θ: {np.degrees(t[0]):>4.2f}, φ: {np.degrees(t[1]):>6.2f}"


def plot_brdf_comparison(n_wi, dense,
                         n_wo_itrp, wi_wo_pairs_itrp,
                         n_wo_olaf, wi_wo_pairs_olaf,
                         brdf_itrp, wavelengths_itrp, brdf_max_itrp,
                         brdf_olaf, wavelengths_olaf, brdf_max_olaf):
    n_wavelengths_itrp = len(wavelengths_itrp)
    n_wavelengths_olaf = len(wavelengths_olaf)
    print(
        f"n_wi: {n_wi}, n_wo_itrp: {n_wo_itrp}, n_wo_olaf: {n_wo_olaf}, n_wavelengths_itrp: {n_wavelengths_itrp}, n_wavelengths_olaf: {n_wavelengths_olaf}")
    fig, ax = plt.subplots(1, 3)
    fig.suptitle('BRDF')
    fig.set_figwidth(13)
    fig.set_figheight(6)

    # for i, ((wi_theta, wi_phi), wos) in enumerate(wi_wo_pairs):
    #     print(f"{i} -- θi: {np.degrees(wi_theta):>6.2f}, φi: {np.degrees(wi_phi):>6.2f}")
    #     for j, (wo_theta, wo_phi) in enumerate(wos):
    #         print(f"    {j}  θo: {np.degrees(wo_theta):>6.2f}, φo: {np.degrees(wo_phi):>6.2f}")

    ax_olaf = ax[0]
    ax_comp = ax[1]
    ax_itrp = ax[2]

    print(f"n_wo_olaf: {n_wo_olaf}, n_wo_itrp: {n_wo_itrp}")

    # from -80 to 80, with step 10
    xs_olaf = np.arange(-n_wo_olaf / 2 * 10, (1 + n_wo_olaf / 2) * 10, 10)
    xs_itrp = np.arange(-n_wo_itrp / 2 * 2.5, (1 + n_wo_itrp / 2) * 2.5, 2.5) if dense else xs_olaf
    print(f"len: {len(xs_itrp)}, xs_itrp: {xs_itrp}")
    print(f"len: {len(xs_olaf)}, xs_olaf: {xs_olaf}")

    # rearrange the data to match the xs, NaN for data that is not measured
    # input data is in the form of [λ, ωi, ωo]
    # we want to merge ωo into a single dimension (in-plane brdf)
    olaf_arranged = np.full((n_wavelengths_olaf, n_wi, (n_wo_olaf + 1)), np.nan)
    itrp_arranged = np.full((n_wavelengths_itrp, n_wi, (n_wo_itrp + 1)), np.nan)
    for wi_idx, ((_wi_theta, _wi_phi), wos) in enumerate(wi_wo_pairs_olaf):
        # print(f"wi idx: {wi_idx}, θi: {np.degrees(wi_theta):>6.2f}, φi: {np.degrees(wi_phi):>6.2f}")
        for wo_idx_org, (wo_theta, wo_phi) in enumerate(wos):
            diff = np.abs(xs_olaf - np.degrees(wo_theta)) if wo_phi < np.pi * 0.5 else np.abs(
                xs_olaf + np.degrees(wo_theta))
            wo_idx = np.argmin(diff)
            for k in np.arange(0, n_wavelengths_olaf):
                olaf_arranged[k, wi_idx, wo_idx] = brdf_olaf[k * n_wi * n_wo_olaf + wi_idx * n_wo_olaf + wo_idx_org]

    for wi_idx, ((wi_theta, wi_phi), wos) in enumerate(wi_wo_pairs_itrp):
        # print(f"wi idx: {wi_idx}, θi: {np.degrees(wi_theta):>6.2f}, φi: {np.degrees(wi_phi):>6.2f}")
        for wo_idx_org, (wo_theta, wo_phi) in enumerate(wos):
            diff = np.abs(xs_itrp - np.degrees(wo_theta)) if wo_phi < np.pi * 0.5 else np.abs(
                xs_itrp + np.degrees(wo_theta))
            wo_idx = np.argmin(diff)
            for k in range(n_wavelengths_itrp):
                itrp_arranged[k, wi_idx, wo_idx] = brdf_itrp[k * n_wi * n_wo_itrp + wi_idx * n_wo_itrp + wo_idx_org]

    cur_wi_idx = 0
    olaf_cur_lambda_idx = 0
    itrp_cur_lambda_idx = 0

    olaf_curve = ax_olaf.plot(xs_olaf, olaf_arranged[olaf_cur_lambda_idx, cur_wi_idx, :], 'o-',
                              label='Measured BRDF(Olaf)')
    itrp_curve = ax_itrp.plot(xs_itrp, itrp_arranged[itrp_cur_lambda_idx, cur_wi_idx, :], 'o-',
                              label='Interpolated BRDF')
    olaf_curve_comp = ax_comp.plot(xs_olaf, olaf_arranged[olaf_cur_lambda_idx, cur_wi_idx, :], 'o-',
                                   label='Measured BRDF(Olaf)')
    itrp_curve_comp = ax_comp.plot(xs_itrp, itrp_arranged[itrp_cur_lambda_idx, cur_wi_idx, :], 'o-',
                                   label='Interpolated BRDF')

    for a in ax:
        a.legend()

    olaf_cur_lambda_text = TextBox(plt.axes([0.15, 0.025, 0.06, 0.04]), 'λ',
                                   initial=f"{wavelengths_olaf[olaf_cur_lambda_idx]:.4f}")
    button_prev_olaf_lambda = Button(plt.axes([0.22, 0.025, 0.06, 0.04]), 'Prev λ', color='lightgoldenrodyellow',
                                     hovercolor='0.975')
    button_next_olaf_lambda = Button(plt.axes([0.29, 0.025, 0.06, 0.04]), 'Next λ', color='lightgoldenrodyellow',
                                     hovercolor='0.975')

    compare_button = Button(plt.axes([0.38, 0.025, 0.06, 0.04]), 'Comp.', color='lightgoldenrodyellow',
                            hovercolor='0.975')
    compare_norm_button = Button(plt.axes([0.45, 0.025, 0.06, 0.04]), '(Comp.)', color='lightgoldenrodyellow',
                                 hovercolor='0.975')
    prev_wi_button = Button(plt.axes([0.53, 0.025, 0.06, 0.04]), 'Prev ωi', color='lightgoldenrodyellow',
                            hovercolor='0.975')
    next_wi_button = Button(plt.axes([0.60, 0.025, 0.06, 0.04]), 'Next ωi', color='lightgoldenrodyellow',
                            hovercolor='0.975')
    cur_wi_text = TextBox(plt.axes([0.46, 0.90, 0.11, 0.04]), 'ωi: ',
                          initial=f"{format_angle_pair(wi_wo_pairs_itrp[cur_wi_idx][0])}")

    itrp_cur_lambda_text = TextBox(plt.axes([0.75, 0.025, 0.06, 0.04]), 'λ',
                                   initial=f"{wavelengths_itrp[itrp_cur_lambda_idx]:.4f}")
    button_next_itrp_lambda = Button(plt.axes([0.82, 0.025, 0.06, 0.04]), 'Next λ', color='lightgoldenrodyellow',
                                     hovercolor='0.975')

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

    def update_comp(olaf_lambda_idx, olaf_wi_idx, itrp_lambda_idx, itrp_wi_idx, normalize=False):
        if normalize:
            max_measured = brdf_max_olaf[olaf_wi_idx * n_wavelengths_olaf + olaf_lambda_idx]
            olaf_curve_comp[0].set_ydata(olaf_arranged[olaf_lambda_idx, olaf_wi_idx, :] / max_measured)
            max_interpolated = brdf_max_itrp[itrp_wi_idx * n_wavelengths_itrp + itrp_lambda_idx]
            itrp_curve_comp[0].set_ydata(itrp_arranged[itrp_lambda_idx, itrp_wi_idx, :] / max_interpolated)
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
        update_comp(olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx, False)

    def on_compare_norm(event):
        update_comp(olaf_cur_lambda_idx, cur_wi_idx, itrp_cur_lambda_idx, cur_wi_idx, True)

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

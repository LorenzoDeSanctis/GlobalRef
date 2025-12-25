import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu


def set_global_figure_style():
    """
    Configure global Matplotlib parameters for publication-quality figures.

    The style is tailored for 1.5-column figures with small font sizes,
    suitable for journal submissions.
    """
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 12,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })


def plot_transform_dashed(ax, T, length=5):
    """
    Plot a dashed SE(3) reference frame on a 3D axis.

    The x, y, and z axes are plotted as dashed red, green, and blue lines,
    respectively.

    Parameters
    ----------
    ax : matplotlib axis
        3D axis on which to plot the frame.
    T : ndarray of shape (4, 4)
        Homogeneous transformation matrix representing the frame.
    length : float, optional
        Length of the axis arrows (default is 5).
    """
    p = T[:3, 3]
    R = T[:3, :3]

    x_axis = p + length * R[:, 0]
    y_axis = p + length * R[:, 1]
    z_axis = p + length * R[:, 2]

    ax.plot(
        [p[0], x_axis[0]], [p[1], x_axis[1]], [p[2], x_axis[2]],
        linestyle="--", color="r", linewidth=2
    )
    ax.plot(
        [p[0], y_axis[0]], [p[1], y_axis[1]], [p[2], y_axis[2]],
        linestyle="--", color="g", linewidth=2
    )
    ax.plot(
        [p[0], z_axis[0]], [p[1], z_axis[1]], [p[2], z_axis[2]],
        linestyle="--", color="b", linewidth=2
    )


def plot_transform_validation(
    T_known,
    T_est_stereo,
    T_est_mono,
    output_path,
    unit="mm"
):
    """
    Plot a side-by-side comparison between known and estimated transformations.

    Two panels are shown: one for the stereo estimate and one for the
    monocular estimate. The known transformation is plotted as a solid
    frame, while the estimated transformations are shown as dashed frames.

    Parameters
    ----------
    T_known : ndarray of shape (4, 4)
        Ground-truth transformation.
    T_est_stereo : ndarray of shape (4, 4)
        Estimated transformation from the stereo pipeline.
    T_est_mono : ndarray of shape (4, 4)
        Estimated transformation from the monocular pipeline.
    output_path : str
        Path where the figure will be saved.
    unit : str, optional
        Unit label for the axes (default is "mm").
    """
    set_global_figure_style()

    fig = plt.figure(figsize=(5.51, 3.2))

    # Stereo panel
    ax1 = pu.make_3d_axis(1, 121, unit=unit)
    ax1.set_title("(a)", pad=2)

    plot_transform_dashed(ax1, T_est_stereo)
    pt.plot_transform(ax=ax1, A2B=T_known, s=5)
    ax1.view_init(elev=20, azim=25)

    # Monocular panel
    ax2 = pu.make_3d_axis(1, 122, unit=unit)
    ax2.set_title("(b)", pad=2)

    plot_transform_dashed(ax2, T_est_mono)
    pt.plot_transform(ax=ax2, A2B=T_known, s=5)
    ax2.view_init(elev=20, azim=25)

    for ax in (ax1, ax2):
        ax.set_xlabel(r"$x$ [{}]".format(unit))
        ax.set_ylabel(r"$y$ [{}]".format(unit))
        ax.set_zlabel("")

    # Axis limits
    all_frames = np.stack([T_known, T_est_stereo, T_est_mono])
    positions = all_frames[:, :3, 3]

    xmin, ymin, _ = positions.min(axis=0) - 5
    xmax, ymax, _ = positions.max(axis=0) + 5

    dx = xmax - xmin
    dy = ymax - ymin
    half = max(dx, dy) / 2

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    for ax in (ax1, ax2):
        ax.set_xlim([xmid - half, xmid + half])
        ax.set_ylim([ymid - half, ymid + half])
        ax.set_zlim([-4, 2.5])

    ax1.legend(
        handles=[
            Line2D([0], [0], linestyle="--", color="k", label="Estimated"),
            Line2D([0], [0], linestyle="-", color="k", label="Known"),
        ],
        loc="upper left",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shah_boxplots(
    ori_data,
    pos_data,
    labels,
    output_path
):
    """
    Plot side-by-side boxplots of Shah orientation and position accuracy metrics.

    Parameters
    ----------
    ori_data : list of ndarray
        Orientation accuracy arrays (e.g., [stereo, monocular]).
    pos_data : list of ndarray
        Position accuracy arrays (e.g., [stereo, monocular]).
    labels : list of str
        Labels corresponding to each method.
    output_path : str
        Path where the figure will be saved.
    """
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 12,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    fig_width = 5.51
    fig_height = 2.2

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "figure.figsize": (fig_width, fig_height),
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # (a) Orientation
    sns.boxplot(
        ax=axes[0],
        data=ori_data,
        width=0.35,
        palette=["#4a4a4a", "#bfbfbf"],
        showcaps=True,
        boxprops={"linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        showfliers=False,
    )

    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Orientation Accuracy")
    axes[0].set_ylim(0.999945, 1)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].text(0.5, 1.1, "(a)", transform=axes[0].transAxes,
                 ha="center", va="bottom", fontsize=12)

    # (b) Position
    sns.boxplot(
        ax=axes[1],
        data=pos_data,
        width=0.35,
        palette=["#4a4a4a", "#bfbfbf"],
        showcaps=True,
        boxprops={"linewidth": 0.8},
        medianprops={"color": "black", "linewidth": 1},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        showfliers=False,
    )

    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Position Accuracy")
    axes[1].set_ylim(0.99999775, 1)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].text(0.5, 1.1, "(b)", transform=axes[1].transAxes,
                 ha="center", va="bottom", fontsize=12)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tau_shah_metrics(
    taus,
    ori_mean_st,
    ori_std_st,
    pos_mean_st,
    pos_std_st,
    ori_mean_m,
    ori_std_m,
    pos_mean_m,
    pos_std_m,
    output_path,
    tau_stereo=None,
    tau_mono=None,
):
    """
    Plot Shah orientation and position accuracy metrics as a function of τ.

    For each modality:
    - solid vertical line  : τ provided in input
    - dashed vertical line : τ maximizing the plotted metric
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------------------------------------------------
    # Style
    # --------------------------------------------------
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.0), sharex=True)

    # τ that maximizes each metric
    tau_ori_st_max = taus[np.argmax(ori_mean_st)]
    tau_ori_m_max  = taus[np.argmax(ori_mean_m)]
    tau_pos_st_max = taus[np.argmax(pos_mean_st)]
    tau_pos_m_max  = taus[np.argmax(pos_mean_m)]

    # ==================================================
    # (a) Orientation accuracy
    # ==================================================
    ax = axes[0]

    ax.plot(taus, ori_mean_st, color="tab:blue", lw=1.2, label="Stereo")
    ax.fill_between(
        taus,
        ori_mean_st - ori_std_st,
        ori_mean_st + ori_std_st,
        color="tab:blue",
        alpha=0.25,
    )

    ax.plot(taus, ori_mean_m, color="tab:red", lw=1.2, label="Monocular")
    ax.fill_between(
        taus,
        ori_mean_m - ori_std_m,
        ori_mean_m + ori_std_m,
        color="tab:red",
        alpha=0.25,
    )

    # solid lines: input τ
    if tau_stereo is not None:
        ax.axvline(-tau_stereo, color="tab:blue", linewidth=1, linestyle="--")
    if tau_mono is not None:
        ax.axvline(-tau_mono, color="tab:red", linewidth=1, linestyle="--")

    # dashed lines: τ maximizing metric
    ax.axvline(tau_ori_st_max, color="tab:blue", linewidth=1)
    ax.axvline(tau_ori_m_max,  color="tab:red", linewidth=1)

    ax.set_ylabel("Orientation accuracy")
    ax.set_ylim(0.99988, 1.0)
    ax.set_xlim(min(taus), max(taus))
    ax.grid(True, alpha=0.25)
    ax.text(
        0.5, 1.1, "(a)",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=12,
    )

    ax.legend(frameon=False, loc=3)

    # ==================================================
    # (b) Position accuracy
    # ==================================================
    ax = axes[1]

    ax.plot(taus, pos_mean_st, color="tab:blue", lw=1.2, label="Stereo")
    ax.fill_between(
        taus,
        pos_mean_st - pos_std_st,
        pos_mean_st + pos_std_st,
        color="tab:blue",
        alpha=0.25,
    )

    ax.plot(taus, pos_mean_m, color="tab:red", lw=1.2, label="Monocular")
    ax.fill_between(
        taus,
        pos_mean_m - pos_std_m,
        pos_mean_m + pos_std_m,
        color="tab:red",
        alpha=0.25,
    )

    # Dashed lines: input τ
    if tau_stereo is not None:
        ax.axvline(-tau_stereo, color="tab:blue", linewidth=0.8, linestyle="--")
    if tau_mono is not None:
        ax.axvline(-tau_mono, color="tab:red", linewidth=0.8, linestyle="--")

    # Solid lines: τ maximizing metric
    ax.axvline(tau_pos_st_max, color="tab:blue", linewidth=0.8)
    ax.axvline(tau_pos_m_max,  color="tab:red", linewidth=0.8)

    ax.set_ylabel("Position accuracy")
    ax.set_ylim(0.999997, 1.0)
    ax.set_xlim(min(taus), max(taus))
    ax.grid(True, alpha=0.25)
    ax.text(
        0.5, 1.1, "(b)",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=12,
    )

    # --------------------------------------------------
    # Common formatting and save
    # --------------------------------------------------
    for ax in axes:
        ax.set_xlabel("Temporal offset τ [s]")

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tau_shah_metrics_2x2(
    taus,
    ori_mean_st,
    ori_std_st,
    pos_mean_st,
    pos_std_st,
    ori_mean_m,
    ori_std_m,
    pos_mean_m,
    pos_std_m,
    output_path,
    tau_stereo=None,
    tau_mono=None,
):
    """
    Plot Shah metrics as a 2x2 figure.

    Top row: orientation accuracy
        (a) Stereo
        (b) Monocular

    Bottom row: position accuracy
        (c) Stereo
        (d) Monocular

    For each panel:
        solid vertical line  : τ provided in input
        dashed vertical line : τ maximizing the plotted metric
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------------------------------------------------
    # Style
    # --------------------------------------------------
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
        },
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.48, 5.6), sharex=True)

    # τ maximizing each metric
    tau_ori_st_max = taus[np.argmax(ori_mean_st)]
    tau_ori_m_max  = taus[np.argmax(ori_mean_m)]
    tau_pos_st_max = taus[np.argmax(pos_mean_st)]
    tau_pos_m_max  = taus[np.argmax(pos_mean_m)]

    # ==================================================
    # (a) Orientation, Stereo
    # ==================================================
    ax = axes[0, 0]

    ax.plot(taus, ori_mean_st, color="tab:blue", lw=1.2)
    ax.fill_between(
        taus,
        ori_mean_st - ori_std_st,
        ori_mean_st + ori_std_st,
        color="tab:blue",
        alpha=0.25,
    )

    if tau_stereo is not None:
        ax.axvline(-tau_stereo, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(tau_ori_st_max, color="black", linewidth=1.0)

    ax.set_ylabel("Orientation accuracy")
    ax.set_ylim(0.99988, 1.0)
    ax.set_xlim(min(taus), max(taus))
    ax.text(0.5, 1.08, "(a)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12)

    # ==================================================
    # (b) Orientation, Monocular
    # ==================================================
    ax = axes[0, 1]

    ax.plot(taus, ori_mean_m, color="tab:red", lw=1.2)
    ax.fill_between(
        taus,
        ori_mean_m - ori_std_m,
        ori_mean_m + ori_std_m,
        color="tab:red",
        alpha=0.25,
    )

    if tau_mono is not None:
        ax.axvline(-tau_mono, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(tau_ori_m_max, color="black", linewidth=1.0)

    ax.set_ylim(0.99988, 1.0)
    ax.set_xlim(min(taus), max(taus))
    ax.text(0.5, 1.08, "(b)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12)

    # ==================================================
    # (c) Position, Stereo
    # ==================================================
    ax = axes[1, 0]

    ax.plot(taus, pos_mean_st, color="tab:blue", lw=1.2)
    ax.fill_between(
        taus,
        pos_mean_st - pos_std_st,
        pos_mean_st + pos_std_st,
        color="tab:blue",
        alpha=0.25,
    )

    if tau_stereo is not None:
        ax.axvline(-tau_stereo, color="black", linewidth=1.2, linestyle="--")
    ax.axvline(tau_pos_st_max, color="black", linewidth=1.0)

    ax.set_ylabel("Position accuracy")
    ax.set_ylim(0.999997, 1.0)
    ax.set_xlim(min(taus), max(taus))
    ax.text(0.5, 1.08, "(c)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12)

    # ==================================================
    # (d) Position, Monocular
    # ==================================================
    ax = axes[1, 1]

    ax.plot(taus, pos_mean_m, color="tab:red", lw=1.2)
    ax.fill_between(
        taus,
        pos_mean_m - pos_std_m,
        pos_mean_m + pos_std_m,
        color="tab:red",
        alpha=0.25,
    )

    if tau_mono is not None:
        ax.axvline(-tau_mono, color="black", linewidth=1.2, linestyle="--",)
    ax.axvline(tau_pos_m_max, color="black", linewidth=1.0)

    ax.set_ylim(0.999997, 1.0)
    ax.set_xlim(min(taus), max(taus))
    ax.text(0.5, 1.08, "(d)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12)

    # --------------------------------------------------
    # Common formatting
    # --------------------------------------------------
    for ax in axes[1, :]:
        ax.set_xlabel("Temporal offset τ [s]")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.25)

    plt.tight_layout(pad=0.6)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # Tau differences (estimated - known)
    # --------------------------------------------------
    print("Tau differences")
    if tau_stereo is not None:
        print(f" Orientation Stereo: {abs(tau_ori_st_max + tau_stereo)*1000:.6f} ms")
        print(f" Position    Stereo: {abs(tau_pos_st_max + tau_stereo)*1000:.6f} ms")

    if tau_mono is not None:
        print(f" Orientation Monocular: {abs(tau_ori_m_max + tau_mono)*1000:.4f} ms")
        print(f" Position    Monocular: {abs(tau_pos_m_max + tau_mono)*1000:.4f} ms")

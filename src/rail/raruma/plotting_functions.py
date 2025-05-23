import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import sigmaclip
from astropy.stats import biweight_location, biweight_scale

def get_nrow_ncol(nfig: int) -> tuple[int, int]:

    shape_dict = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (2, 4),
        8: (2, 4),
        9: (3, 3),
        10: (3, 4),
        11: (3, 4),
        12: (3, 4),
        13: (4, 4),
        14: (4, 4),
        15: (4, 4),
        16: (4, 4),
    }
    try:
        return shape_dict[nfig]
    except KeyError:
        raise ValueError(f"Sorry, Phillipe.  I'm not going to put {nfig} subplots in one figure") from None



def plot_feature_histograms(data, labels: list[str]|None = None) -> Figure:

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = get_nrow_ncol(n_features)    
    axs = fig.subplots(nrow, ncol)

    for ifeature in range(n_features):
        icol = int(ifeature / ncol)
        irow = ifeature % ncol

        axs[icol][irow].hist(data[:,ifeature], bins=100)
        if labels is not None:
            axs[icol][irow].set_xlabel(labels[ifeature])
            
    return fig


def plot_true_nz(targets) -> Figure:
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1)
    ax.hist(targets, bins=100)
    return fig


def plot_pca_hist2d(data, pca_out, labels: list[str]|None = None) -> Figure:

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = n_features, n_features
    axs = fig.subplots(nrow, ncol)

    for irow in range(nrow):
        row_data = data[:,irow]
        for icol in range(ncol):
            col_data = pca_out[:,icol]
            axs[irow][icol].hist2d(row_data, col_data, bins=(100, 100), norm='log')
        if labels is not None:
            axs[irow][0].set_xlabel(labels[ifeature]) 
            
    return fig


def plot_feature_target_hist2d(data, targets, labels: list[str]|None = None) -> Figure:

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = get_nrow_ncol(n_features)    
    axs = fig.subplots(nrow, ncol)

    for ifeature in range(n_features):
        icol = int(ifeature / ncol)
        irow = ifeature % ncol

        axs[icol][irow].hist2d(targets, data[:,ifeature], bins=100)
        if labels is not None:
            axs[icol][irow].set_xlabel(labels[ifeature])
            
    return fig


def plot_features_target_scatter(data, targets, labels: list[str]|None = None) -> Figure:

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = n_features, n_features
    axs = fig.subplots(nrow, ncol)

    for irow in range(nrow):
        row_data = data[irow]
        for icol in range(ncol):
            col_data = pca_out[icol]
            axs[irow][icol].scatter(row_data, col_data, c=targets, cmap='rainbow', marker='.', s=1)
        if labels is not None:
            axs[irow][0].set_xlabel(labels[ifeature]) 
            
    return fig


def plot_true_predict(targets, predictions) -> Figure:

    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1)
    ax.hist2d(targets, predictions, bins=(100, 100), norm='log')
    return fig


def get_biweight_mean_sigma_outlier(subset: np.ndarray, nclip: int=3) -> tuple[float, float, float, float]:
    subset_clip, _, _ = sigmaclip(subset, low=3, high=3)
    for _j in range(nclip):
        subset_clip, _, _ = sigmaclip(subset_clip, low=3, high=3)
        
    mean = biweight_location(subset_clip)
    std = biweight_scale(subset_clip)
    outlier_rate = np.sum(np.abs(subset) > 3 * biweight_scale(subset_clip)) / len(
        subset
    )

    return mean, std / np.sqrt(len(subset_clip)), std, outlier_rate


def plot_true_predict_fancy(targets, predictions) -> Figure:
    z_min = 0.
    z_max = 3.
    figure, axes = plt.subplots(figsize=(7, 6))
    bin_edges = np.linspace(0, 3., 301)
    dz = (predictions - targets) / (1 + targets)
    mean, mean_err, std, outlier_rate = get_biweight_mean_sigma_outlier(dz, nclip=3)
    mean, std, outlier_rate = round(mean, 4), round(std, 4), round(outlier_rate, 4)
    h = axes.hist2d(
        targets,
        predictions,
        bins=(bin_edges, bin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
    )
    axes.plot(
        [z_min - 10, z_max + 10],
        [z_min - 10, z_max + 10],
        "--",
        color="red",
    )
    axes.plot(
        [z_min - 10, z_max + 10],
        [z_min - 10 - 3 * std, z_max + 10 - 3 * std],
        "--",
        color="red",
    )
    axes.plot(
        [z_min - 10, z_max + 10],
        [z_min - 10 + 3 * std, z_max + 10 + 3 * std],
        "--",
        color="red",
    )
    axes.plot(
        [],
        [],
        ".",
        alpha=0.0,
        label=rf"$\Delta z = {mean} $"
        + "\n"
        + rf"$\sigma z = {std} $"
        + "\n"
        + f"outlier rate = {outlier_rate}",
    )

    plt.xlabel("True Redshift")
    plt.ylabel("Estimated Redshift")
    cb = figure.colorbar(h[3], ax=axes)
    cb.set_label("Density")

    plt.legend()

    
def process_data(
    zphot: np.ndarray,
    specz: np.ndarray,
    low: float=0.01,
    high: float=2.,
    nclip: int=3,
    nbin: int=101,
) -> dict[str, list[float]]:
    dz = (zphot - specz) / (1 + specz)
    
    z_bins = np.linspace(low, high, nbin)
    # Bin the data
    bin_indices = np.digitize(zphot, bins=z_bins) - 1  # Assign each point to a bin
    
    biweight_mean: list[float] = []
    biweight_std: list[float] = []
    biweight_sigma: list[float] = []
    biweight_outlier: list[float] = []
    z_mean: list[float] = []
    qt_95_low: list[float] = []
    qt_68_low: list[float] = []
    median: list[float] = []
    qt_68_high: list[float] = []
    qt_95_high: list[float] = []
    for i in range(len(z_bins) - 1):
        subset = dz[bin_indices == i]
        if len(subset) < 1:
            continue
        subset_clip, _, _ = sigmaclip(subset, low=3, high=3)
        for _j in range(nclip):
            subset_clip, _, _ = sigmaclip(subset_clip, low=3, high=3)
            
        biweight_mean.append(biweight_location(subset_clip))
        biweight_std.append(biweight_scale(subset_clip) / np.sqrt(len(subset_clip)))
        biweight_sigma.append(biweight_scale(subset_clip))
        
        outlier_rate = np.sum(
            np.abs(subset) > 3 * biweight_scale(subset_clip)
        ) / len(subset)
        biweight_outlier.append(outlier_rate)

        qt_95_low.append(np.percentile(subset, 2.5))
        qt_68_low.append(np.percentile(subset, 16))
        median.append(np.percentile(subset, 50))
        qt_68_high.append(np.percentile(subset, 84))
        qt_95_high.append(np.percentile(subset, 97.5))
        
        z_mean.append(np.mean(zphot[bin_indices == i]))
    return {
        "z_mean": z_mean,
        "biweight_mean": biweight_mean,
        "biweight_std": biweight_std,
        "biweight_sigma": biweight_sigma,
        "biweight_outlier": biweight_outlier,
        "qt_95_low": qt_95_low,
        "qt_68_low": qt_68_low,
        "median": median,
        "qt_68_high": qt_68_high,
        "qt_95_high": qt_95_high,
    }

    
def plot_biweight_stats_v_redshift(targets, predictions) -> Figure:

    n_zbins = 100
    z_min = 0.
    z_max = 3.
    n_clip = 3

    dz = (predictions - targets) / (1 + targets)
    
    results = process_data(
        predictions,
        targets,
        nbin=n_zbins,
        low=z_min,
        high=z_max,
        nclip=n_clip,
    )
    figure, axes = plt.subplots(2, 1, figsize=(8, 6))

    plt.subplots_adjust(wspace=0.1, hspace=0.0)

    axes[0].errorbar(
        results["z_mean"],
        results["biweight_mean"],
        results["biweight_std"],
        label="Bias",
    )
    
    axes[0].plot(results["z_mean"], results["biweight_sigma"], label=r"$\sigma_z$")

    axes[0].plot(
        results["z_mean"], results["biweight_outlier"], label=r"Outlier rate"
    )
    axes[0].set_title(
        f"Bias, Sigma, and Outlier rates w/ {n_clip} sigma clipping"
    )
    axes[0].set_ylabel("Statistics")
    axes[0].legend()
    axes[0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axes[0].set_xlim(z_min, z_max)

    bin_edges_z = np.linspace(z_min, z_max, 100 + 1)
    bin_edges_dz = np.linspace(-0.3, 0.3, 100 + 1)
    axes[1].hist2d(
        predictions,
        dz,
        bins=(bin_edges_z, bin_edges_dz),
        norm=colors.LogNorm(),
        cmap="gray",
    )

    axes[1].set_xlim(z_min, z_max)
    for qt in ["qt_95_low", "qt_68_low", "median", "qt_68_high", "qt_95_high"]:
        axes[1].plot(
            results["z_mean"], results[qt], "--", color="blue", linewidth=2.0
        )

    axes[1].set_xlabel("Redshift")
    axes[1].set_ylabel(r"$(z_{phot} - z_{spec})/(1+z_{spec})$")
    return figure
    

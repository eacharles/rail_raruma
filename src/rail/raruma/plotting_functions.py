
import matplotlib.pyplot as plt


def get_nrow_ncol(nfig: int) -> tuple[int, int]:

    shape_dict = {
        1: (1, 1),
        2: (1, 2),
        3, (1, 3),
        4, (2, 2),
        5, (2, 3),
        6, (2, 3),
        7, (2, 4),
        8, (2, 4),
        9, (3, 3),
        10, (3, 4),
        11, (3, 4),
        12, (3, 4),
        13, (4, 4),
        14, (4, 4),
        15, (4, 4),
        16, (4, 4),
    }
    try:
        return shape_dict[nfig]
    except KeyError:
        raise ValueError(f"Sorry, Phillipe.  I'm not going to put {nfig} subplots in one figure") from None



def plot_feature_histograms(data, labels: list[str]|None = None) -> matplotlib.Figure:

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


def plot_pca_hist2d(data, pca_out, labels: list[str]|None = None) -> matplotlib.Figure:

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = n_features, n_features
    axs = fig.subplots(nrow, ncol)

    for irow in range(nrow):
        row_data = data[icol]
        for icol in range(ncol):
            col_data = pca_out[icol]
            axs[irow][icol].hist2d(row_data, col_data, bins=(100, 100), norm='log')
        if labels is not None:
            axs[irow][0].set_xlabel(labels[ifeature]) 
            
    return fig


def plot_feature_target_hist2d(data, targets, labels: list[str]|None = None) -> matplotlib.Figure:

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


def plot_features_target_scatter(data, targets, labels: list[str]|None = None) -> matplotlib.Figure:

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


def plot_true_predict(targets, predictions) -> matplotlib.Figure:

    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1)
    ax.hist2d(targets, predictions, bins=(100, 100), norm='log')
    return fig

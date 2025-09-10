import numpy as np
from . import utility_functions as raruma_util


def make_admixture(
    mags: np.ndarray,
    library: np.ndarray,
    admixture: float=0.01,
    zero_points: float = 31.4,
    seed: int | None = None,
) -> np.ndarray:
    """Add an admixture of fluxes to objects

    Parameters
    ----------
    mags:
        Input mags [N_bands, N_objects]

    library:
        Library for adding admixtures  [N_bands, N_bkg_objects]

    admixture:
        Admixture fraction

    zero_points:
        Flux to mag zero-point

    seed:
        Random number seed

    Returns
    -------
    Magnitudes with flux admixtures added
    """
    n_lib = len(library)
    n_obj = len(mags)

    fluxes = raruma_util.mags_to_fluxes(mags, zero_points)
    lib_fluxes = raruma_util.mags_to_fluxes(library, zero_points)
    lib_fluxes = np.nan_to_num(lib_fluxes)

    total_fluxes = fluxes.sum(axis=1)

    if seed is not None:
        np.random.seed(seed)

    picks = np.random.randint(n_lib, size=n_obj)
    pick_fluxes = lib_fluxes[picks]
    pick_totals = pick_fluxes.sum(axis=1)

    pick_weights = total_fluxes / pick_totals
    contamination = (pick_fluxes.T * (pick_weights * admixture)).T

    new_fluxes = fluxes + contamination
    new_mags = raruma_util.fluxes_to_mags(new_fluxes, zero_points)

    return new_mags


from rail.raruma import utility_functions as raruma_util


def make_admixture(
    mags: np.ndarray,
    library: np.ndarray,
    admixture: float,
    zero_points: float=31.4,
    seed:int|None=None,
) -> np.ndarray:

    n_lib = len(library)
    n_obj = len(mags)
    
    fluxes = raruma_util.mags_to_fluxes(mags, zero_points)
    
    total_fluxes = fluxes.sum(axis=1)

    if seed is not None:
        np.random.seed(seed)

    picks = np.random.randint(n_lib, size=(n_obj))
    pick_fluxes = fluxes[picks]
    pick_totals = pick_fluxes.sum(axis=1)

    pick_weights = total_fluxes / pick_totals
    contamination = (pick_fluxes.T*(pick_weights*admixture)).T
        
    new_fluxes = fluxes + contamination
    new_mags = raruma_util.fluxes_to_mags(new_fluxes, zero_points)
    
    return new_mags
    
    



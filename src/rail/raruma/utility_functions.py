


def make_band_names(template: str, bands: list[str]) -> list[str]:
    return [template.format(band=band_) for band_ in bands]


def extract_data_to_2d_array(data: TableLike, column_names: list[str]) -> np.ndarray:
    column_data = [data[column_] for column_ in column_names]
    return np.vstack(column_data).T


def fluxes_to_mags(fluxes: np.ndarray, zero_points: float|np.ndarray) -> np.ndarray:    
    return -2.5 * np.log10(fluxes) + zero_points
        

def mags_to_fluxes(mags: np.ndarray, zero_points: float|np.ndarray) -> np.ndarray:    
    return np.power(10, (zero_points - mags)/2.5)


def adjacent_band_colors(mags: np.ndarray) -> np.ndarray:
    n_bands = mags.shape[-1]
    colors = [mags[i+1] - mags[i] for i in range(n_bands-1)]
    return np.vstack(colors)
    

def ref_band_colors(mags: np.ndarray, ref_band_index: int) -> np.ndarray:
    ref_mags = mags[:,ref_band_index]
    n_bands = mags.shape[-1]
    colors_list = []
    for i in range(n_bands):
        if i == ref_band_index:
            continue
        colors_list.append(mags[:,i] - ref_mags)
    return np.vstack(colors_list).T
        

def run_regression(
    regerssor,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    regerssor.fit(train_features, train_targets)
    return regerssor.predict(test_features)

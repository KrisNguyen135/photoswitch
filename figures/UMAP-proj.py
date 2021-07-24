import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib.colors import LinearSegmentedColormap

RANDOM_SEED: int = 42

METRIC: str = 'jaccard'
N_NEIGHBORS: int = 7

SMILE_KEY: str = 'Smiles'
FP_KEY: str = 'Morgan fingerprint (2048 bits)'
TARGET_KEY: str = '$λ^{max}$(nm)'
COREID_KEY: str = 'Core ID'
# Change me
INPUT_CSV: str = 'UMAP-AllHits.csv'
X_KEY: str = 'x'
Y_KEY: str = 'y'

MARKER_SIZE: float = 6.0
EDGE_SIZE: float = .1
# Common choices are 'white', 'black' or 'none'
EDGE_COLOR: str = 'none'
OUTPUT_FIGURE_FORMAT: str = 'pdf'
OUTPUT_FIGURE_NAME: str = 'umap_morganfp4.{ext}'.format(ext=OUTPUT_FIGURE_FORMAT)


def wavelength_to_rgb(wavelength):
    '''
    Based on wikipedia visible light article
    '''
    wavelength = float(wavelength)
    if 400 <= wavelength <= 450:
        return '#5d1cb7' #af88e8' #b79fda' #ccb0f4'
    elif 450 <= wavelength <= 485:
        return '#0b54b4' #8888f0' #9e9edc' #b0b0f4'
    elif 485 <= wavelength <= 500:
        return '#0ce2f0' #85f3f3' #9edcdc' #b0f4f4'
    elif 500 <= wavelength <= 565:
        return '#58cd21' #8ef28e' #9edc93' #b0f4b0'
    elif 565 <= wavelength <= 590:
        return '#f2f10c' #d6d66d' #dcdc9e' #f4f4b0'
    elif 590 <= wavelength <= 625:
        return '#f4a408' #f29f68' #dcb89e' #f4ccb0'
    elif 625 <= wavelength <= 700:
        return '#e43c24' #d66969' #dc9e9e' #f4b0b0'
    else:
        return '#969691' #babab0' #cfcfc4'


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, index_col=False,
                     dtype={SMILE_KEY: str,
                            COREID_KEY: int,
                            FP_KEY: str,
                            TARGET_KEY: np.float32})
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # fp column entries are just strings, need to convert to an array of ints
    # could also go straight to bools here if only using distance metrics for binary data
    df[FP_KEY] = df[FP_KEY].map(lambda x: np.array(list(x), dtype=int))
    return df


def project_data(df: pd.DataFrame) -> pd.DataFrame:
    stacked_fps = np.stack(df[FP_KEY])
    reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, metric=METRIC, random_state=RANDOM_SEED)
    projection = reducer.fit_transform(stacked_fps)
    df[X_KEY] = projection[:, 0]
    df[Y_KEY] = projection[:, 1]
    return df


def create_figure(df: pd.DataFrame) -> None:
    # Swap next two lines to change colorbar ticks and coloring
    clim = (df[TARGET_KEY].min(), df[TARGET_KEY].max())
    # clim = (300, 700)
    norm = plt.Normalize(*clim)
    wl = np.arange(clim[0], clim[1] + 1, 2)
    colorlist = list(zip(norm(wl), [wavelength_to_rgb(w) for w in wl]))
    spectralmap = LinearSegmentedColormap.from_list("spectrum", colorlist)

    sns.set_theme()
    points = plt.scatter(df[X_KEY],
                         df[Y_KEY],
                         s=MARKER_SIZE,
                         linewidth=EDGE_SIZE,
                         c=df[TARGET_KEY],
                         edgecolor=EDGE_COLOR,
                         cmap=spectralmap,
                          alpha=0.7)
    cb = plt.colorbar(points)
    cb.set_label(TARGET_KEY)

    plt.axis('off')
    plt.grid(b=None)

    plt.savefig(OUTPUT_FIGURE_NAME, bbox_inches='tight', format=OUTPUT_FIGURE_FORMAT)
    plt.close()


def main():
    df = load_raw_data()
    df = process_data(df)
    df = project_data(df)
    create_figure(df)


if __name__ == '__main__':
    main()

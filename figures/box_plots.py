from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CURRENT_FOLDER_PATH = Path(__file__).parent.resolve()
FIGURES_FOLDER_PATH = Path(CURRENT_FOLDER_PATH, 'figures')
HISTS_FOLDER_PATH = Path(FIGURES_FOLDER_PATH, 'hists')


SMILE_KEY: str = 'Smiles'
FP_KEY: str = 'Morgan fingerprint (2048 bits)'
TARGET_KEY: str = '$λ^{max}$(nm)'
COREID_KEY: str = 'Core ID'

INPUT_NAME: str = 'UMAP-AllHits.csv'


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_NAME, index_col=0,
                     dtype={SMILE_KEY: str,
                            COREID_KEY: int,
                            FP_KEY: str,
                            TARGET_KEY: np.float32})
    return df


def plot_absorbance_by_coreid_boxplot(df: pd.DataFrame) -> None:
    FIGURES_FOLDER_PATH.mkdir(exist_ok=True, parents=True)
    grouped = df.loc[:, [COREID_KEY, TARGET_KEY]].groupby([COREID_KEY])
    iqr = grouped.quantile(.75) - grouped.quantile(.25)

    iqr_target_key = '{} IQR'.format(TARGET_KEY)
    iqr = iqr.rename({TARGET_KEY: iqr_target_key}, axis=1)

    iqr = iqr.sort_values(by=iqr_target_key)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[COREID_KEY], y=df[TARGET_KEY], color=sns.color_palette()[0])
    plt.savefig(Path(FIGURES_FOLDER_PATH, 'box.png'))
    plt.close()


def plot_absorbance_values_by_coreid_hists(df: pd.DataFrame) -> None:
    HISTS_FOLDER_PATH.mkdir(exist_ok=True, parents=True)
    core_ids = df[COREID_KEY].unique()
    for core_id in core_ids:
        filtered_df = df[df[COREID_KEY] == core_id]
        sns.histplot(x=filtered_df[TARGET_KEY])
        plt.savefig(Path(HISTS_FOLDER_PATH, 'hist_of_core_id_{}.png'.format(core_id)), bbox_inches='tight', format='png')
        plt.close()


def plot_absorbance_stds_by_coreid_barchart(df: pd.DataFrame) -> None:
    FIGURES_FOLDER_PATH.mkdir(exist_ok=True, parents=True)
    std_df = df.groupby(COREID_KEY)[TARGET_KEY].std().reset_index()
    std_target = '{} Standard Deviation'.format(TARGET_KEY)
    std_df = std_df.rename({TARGET_KEY: std_target}, axis=1)
    sns.barplot(data=std_df,
                x=COREID_KEY,
                y=std_target,
                order=std_df.sort_values(std_target)[COREID_KEY],
                color=sns.color_palette()[0])
    plt.savefig(Path(FIGURES_FOLDER_PATH, 'stds.png'), bbox_inches='tight',
                format='png')
    plt.close()


def main():
    sns.set(style="white", font_scale=1.5)

    sns.set_context("notebook")
    sns.despine()
    df = load_raw_data()
    plot_absorbance_by_coreid_boxplot(df)
    plot_absorbance_values_by_coreid_hists(df)
    plot_absorbance_stds_by_coreid_barchart(df)


if __name__ == '__main__':
    main()

from typing import Union, Tuple

import numpy as np
import pandas as pd


def saptial_figsize(sample_df, scaling_factor: Union[int, float]=1) -> Tuple[int, int]:
    """
    Calculate the figure size for spatial-based plot according to the points and the span of x and y.
    :param sample_df: pd.DataFrame, the sample data.
    :param scale_factor: float, the scale factor control the size of spatial-based plots. The larger the scale factor,
    the larger the figure size.
    :return: tuple[int, int], the figure size.
    """

    n_points = sample_df[['x', 'y']].dropna().shape[0]
    # debug(f'n_points: {n_points}')

    x_span = sample_df['x'].dropna().max() - sample_df['x'].dropna().min()
    y_span = sample_df['y'].dropna().max() - sample_df['y'].dropna().min()
    # debug(f'x_span: {x_span}')
    # debug(f'y_span: {y_span}')

    points_density = n_points / x_span / y_span * 10_000

    # debug(f'points density: {points_density}')

    fig_width = x_span / 2_000 * scaling_factor * np.sqrt(points_density) + .5  # Adding 2 for colorbar space
    fig_height = y_span / 2_000 * scaling_factor * np.sqrt(points_density) + .2  # Adding 1.5 for title space

    # debug(f'scale_factor: {scale_factor}')
    # debug(f'fig_width: {fig_width}')
    # debug(f'fig_height: {fig_height}')

    return fig_width, fig_height


def gini(array: Union[np.ndarray, pd.Series]) -> float:
    """Calculate the Gini coefficient of a numpy array.
    :param array: np.ndarray or pd.Series, the array for calculating Gini coefficient.
    :return: float, the Gini coefficient.
    """
    #
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    if isinstance(array, pd.Series):
        array = np.array(array)
    array = array.flatten()  # type: ignore
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)  # type: ignore
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)  # type: ignore
    # Number of array elements:
    n = array.shape[0]  # type: ignore
    # Index per array element:
    index = np.arange(1, n + 1)  # type: ignore
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # type: ignore
